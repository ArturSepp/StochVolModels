"""Offline research benchmark, not a public SVI API or production selection rule.

Run this file to compare six fitters on frozen synthetic training/held-out quotes.
Outputs, including failed jobs, exact quotes, diagnostics and source hashes, go to
``local_path.get_output_path()/smile_fitter_benchmark/<UTC run>``. No vendor data.
Research criteria are fixed below before the first comparison; see the local S4
protocol and the surface diagnostics guide for their scope and limitations.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from typing import Callable
import warnings

import numpy as np
from numba.typed import List
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from stochvolmodels import OptionChain, fitters, local_path
from stochvolmodels.pricers.gmm_pricer import GmmPricer
from stochvolmodels.pricers.heston_pricer import (
    HestonParams, HestonPricer, heston_chain_pricer,
)
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pricer import (
    ConstraintsType, LogSVPricer, LogsvModelCalibrationType,
)

from _svi_benchmark import fit_raw_svi, fit_raw_svi_legacy, fit_ssvi, raw_svi
from _svi_wing_benchmark import fit_raw_svi_constrained, svi_prediction_domain


# Keep the original fitter in the default comparison: the bounded candidate's
# fresh-case convergence gains did not establish acceptable wing stability.
MODELS = ("approx_logsv", "gmm2", "raw_svi_legacy", "ssvi", "logsv", "heston")
AVAILABLE_MODELS = (*MODELS, "raw_svi", "raw_svi_constrained", "raw_svi_price",
                    "raw_svi_price_seeded")
CASES = ("flat", "skewed", "sparse_noisy", "short")
STABILITY_FAMILIES = ("flat", "left", "right", "short_sparse")
STABILITY_SEEDS = (301, 302, 303, 304, 305)
WING_SEEDS = (401, 402, 403, 404, 405)
FOLDS = ("interior", "wings")
PRICE_FLOOR = 0.01
DYNAMICS_DEADLINE_SECONDS = 75.0
CRITERIA = {
    "p95_spread_error": 1.0, "coverage": 0.90, "iv_rmse": 0.005,
    "p95_perturbation": 1.0, "smile_warm_seconds": 10.0,
    "dynamics_warm_seconds": 60.0, "relative_error_improvement": 0.20,
    "coverage_improvement": 0.05,
}


def black_prices(forward, strikes, ttm, vols, discount, optiontypes):
    """Independent vectorized SciPy Black formula in USD per underlying unit."""
    root = np.asarray(vols) * np.sqrt(ttm)
    d1 = np.log(forward / strikes) / root + root / 2
    d2 = d1 - root
    call = discount * (forward * norm.cdf(d1) - strikes * norm.cdf(d2))
    put = discount * (strikes * norm.cdf(-d2) - forward * norm.cdf(-d1))
    return np.where(optiontypes == "C", call, put)


def _forward(ttm):
    """USD 100 spot, rate 2%, proportional dividend yield 1%."""
    return 100 * np.exp(0.01 * ttm)


def _discount(ttm):
    """Continuously compounded rate 2%."""
    return np.exp(-0.02 * ttm)


def _quote_chain(ttms, ks, ivs):
    """Build the same finite bid/ask observations for every model."""
    strikes = [np.asarray(_forward(t) * np.exp(k), dtype=float) for t, k in zip(ttms, ks)]
    cps = [np.where(k < 0, "P", "C") for k in ks]
    chain = OptionChain(
        ttms=np.asarray(ttms, dtype=float), forwards=_forward(ttms),
        discfactors=_discount(ttms), ids=np.array([f"T{i}" for i in range(len(ttms))]),
        strikes_ttms=List(strikes), optiontypes_ttms=List(cps),
        bid_ivs=List([v - 0.005 for v in ivs]), ask_ivs=List([v + 0.005 for v in ivs]),
    )
    for side in ("bid", "ask"):
        setattr(chain, f"{side}_prices", List([
            black_prices(f, k, t, v, d, cp)
            for f, k, t, v, d, cp in zip(
                chain.forwards, strikes, ttms, getattr(chain, f"{side}_ivs"),
                chain.discfactors, cps,
            )
        ]))
    return chain


@dataclass
class Sample:
    """One immutable-by-convention synthetic quote universe."""

    name: str
    ttms: np.ndarray
    z: np.ndarray
    ks: list[np.ndarray]
    ivs: list[np.ndarray]
    perturbation_seed: int | None = None

    def chain(self, mask=None, perturb=False):
        """Select training/test rows; perturbations never change a held-out target."""
        if mask is None:
            mask = np.ones(len(self.z), dtype=bool)
        seed = (20261100 + CASES.index(self.name) if self.perturbation_seed is None
                else self.perturbation_seed)
        rng = np.random.default_rng(seed)
        # Draw the same full-universe shocks for all folds before selecting training rows.
        ivs = [v + rng.normal(0, 0.0005, len(v)) if perturb else v.copy() for v in self.ivs]
        return _quote_chain(self.ttms, [k[mask] for k in self.ks], [v[mask] for v in ivs])

    def training_mask(self, fold):
        """Fixed interior interpolation and full two-wing extrapolation holdouts."""
        if fold == "interior":
            return (np.arange(len(self.z)) % 3 != 1) | np.isclose(self.z, 0)
        if fold == "wings":
            return np.abs(self.z) <= 1.5
        raise ValueError(fold)


def make_sample(name):
    """Generate flat or martingale-mixture quotes with independent Black inversion."""
    if name not in CASES:
        raise ValueError(name)
    ttms = np.array([7, 14, 30]) / 365 if name == "short" else np.array([0.1, 0.4, 1.0])
    z = np.linspace(-2.25, 2.25, 13 if name == "sparse_noisy" else 21)
    ks = [z * 0.28 * np.sqrt(t) for t in ttms]
    rng = np.random.default_rng(20261012)
    ivs = []
    for t, k in zip(ttms, ks):
        if name == "flat":
            iv = np.full(len(k), 0.28)
        else:
            weights = np.array([0.35, 0.65])
            shifts = np.exp(np.array([-0.12, 0.03]) * t)
            shifts /= weights @ shifts
            vols = np.array([0.40, 0.18])
            strikes, cps = _forward(t) * np.exp(k), np.where(k < 0, "P", "C")
            prices = sum(
                w * black_prices(_forward(t) * s, strikes, t, v, _discount(t), cps)
                for w, s, v in zip(weights, shifts, vols)
            )
            iv = np.array([
                brentq(
                    lambda v: float(black_prices(_forward(t), strike, t, v,
                                                 _discount(t), cp)) - price,
                    0.005, 3.0, xtol=1e-13,
                ) for strike, cp, price in zip(strikes, cps, prices)
            ])
            if name == "sparse_noisy":
                iv += rng.normal(0, 0.001, len(iv))
        ivs.append(iv)
    return Sample(name, ttms, z, ks, ivs)


def make_stability_sample(family, seed):
    """Frozen follow-up cases: new tenors, three-state mixtures and five noise seeds.

    These cases were specified before the raw-SVI conditioning follow-up and are
    evaluated only after freezing its settings. Reusing them for later tuning
    would make them development data. The initial S4 cases already have that role.
    """
    if family not in STABILITY_FAMILIES or seed not in STABILITY_SEEDS:
        raise ValueError("Unknown stability family or seed")
    short = family == "short_sparse"
    ttms = np.array([5, 12, 25]) / 365 if short else np.array([0.08, 0.3, 0.8])
    z = np.linspace(-2.5, 2.5, 15 if short else 25)
    ks = [z * 0.30 * np.sqrt(t) for t in ttms]
    rng = np.random.default_rng(seed)
    ivs = []
    for t, k in zip(ttms, ks):
        if family == "flat":
            iv = np.full(len(k), 0.24)
        else:
            weights, vols = np.array([0.25, 0.50, 0.25]), np.array([0.50, 0.24, 0.12])
            drift = np.array([-0.20, 0.02, 0.12]) * (-1 if family == "right" else 1)
            shifts = np.exp(t * drift)
            shifts /= weights @ shifts
            strikes, cps = _forward(t) * np.exp(k), np.where(k < 0, "P", "C")
            prices = sum(w * black_prices(_forward(t) * s, strikes, t, v, _discount(t), cps)
                         for w, s, v in zip(weights, shifts, vols))
            iv = np.array([brentq(
                lambda v: float(black_prices(_forward(t), strike, t, v, _discount(t), cp))
                - price, 0.005, 3.0, xtol=1e-13,
            ) for strike, cp, price in zip(strikes, cps, prices)])
        ivs.append(iv + rng.normal(0, 0.0005 if family == "flat" else 0.001, len(k)))
    return Sample(f"{family}_{seed}", ttms, z, ks, ivs, perturbation_seed=seed + 10000)


def make_wing_sample(family, seed):
    """Predeclared wing-study cases; separate from the observed stability suite."""
    if family not in STABILITY_FAMILIES or seed not in WING_SEEDS:
        raise ValueError("Unknown wing-study family or seed")
    short = family == "short_sparse"
    ttms = np.array([3, 9, 21]) / 365 if short else np.array([0.06, 0.25, 0.65])
    z = np.linspace(-2.75, 2.75, 17 if short else 29)
    ks = [z * 0.30 * np.sqrt(t) for t in ttms]
    rng = np.random.default_rng(seed)
    ivs = []
    for t, k in zip(ttms, ks):
        if family == "flat":
            iv = np.full(len(k), 0.22)
        else:
            weights, vols = np.array([0.20, 0.55, 0.25]), np.array([0.45, 0.22, 0.10])
            drift = np.array([-0.16, 0.015, 0.10]) * (-1 if family == "right" else 1)
            shifts = np.exp(t * drift)
            shifts /= weights @ shifts
            strikes, cps = _forward(t) * np.exp(k), np.where(k < 0, "P", "C")
            prices = sum(w * black_prices(_forward(t) * s, strikes, t, v, _discount(t), cps)
                         for w, s, v in zip(weights, shifts, vols))
            iv = np.array([brentq(
                lambda v: float(black_prices(_forward(t), strike, t, v, _discount(t), cp))
                - price, 0.005, 3.0, xtol=1e-13,
            ) for strike, cp, price in zip(strikes, cps, prices)])
        ivs.append(iv + rng.normal(0, 0.0005 if family == "flat" else 0.001, len(k)))
    return Sample(f"{family}_{seed}", ttms, z, ks, ivs, perturbation_seed=seed + 10000)


@dataclass
class Fitted:
    """Research adapter around final parameters and frozen pricing settings."""

    predict: Callable
    continuous: bool
    parameters: object
    optimizer: object
    domains: tuple = ()


class _CalibrationDeadline:
    """Stop a costly native calibration at objective boundaries, without changing it."""

    def __init__(self):
        self.deadline = perf_counter() + DYNAMICS_DEADLINE_SECONDS

    def compute_model_ivols_for_chain(self, *args, **kwargs):
        """A single numerical evaluation can overrun; elapsed time remains recorded."""
        self._check_deadline()
        result = super().compute_model_ivols_for_chain(*args, **kwargs)
        self._check_deadline()
        return result

    def _check_deadline(self):
        """Propagate an explicit failure instead of returning partially fitted parameters."""
        if perf_counter() > self.deadline:
            raise TimeoutError(f"Native calibration exceeded {DYNAMICS_DEADLINE_SECONDS:g}s "
                               "at an objective boundary")


class _LimitedLogSV(_CalibrationDeadline, LogSVPricer):
    """Native LogSV objective with a research-runner wall-time guard."""


class _LimitedHeston(_CalibrationDeadline, HestonPricer):
    """Native Heston objective with the same wall-time guard."""


def _iv_prediction(iv_callback):
    """Convert final smile IVs to independent Black prices, without clipping."""
    def predict(chain):
        ivs = [iv_callback(np.log(k / f), t) for k, f, t in zip(
            chain.strikes_ttms, chain.forwards, chain.ttms)]
        prices = [black_prices(f, k, t, v, d, cp) for f, k, t, v, d, cp in zip(
            chain.forwards, chain.strikes_ttms, chain.ttms, ivs,
            chain.discfactors, chain.optiontypes_ttms)]
        return prices, ivs
    return predict


def _slice_index(ttms, t):
    """Reject implicit interpolation of independently fitted slices."""
    match = np.flatnonzero(np.isclose(ttms, t, rtol=0, atol=1e-12))
    if len(match) != 1:
        raise ValueError("This fitter has no maturity interpolation")
    return int(match[0])


def enforce_prediction_domain(fit, chain, evidence):
    """Retain every requested slice before rejecting any unsupported prediction."""
    domains = getattr(fit, "domains", ())
    if not domains:
        evidence["domain"] = dict(policy="not_declared_by_this_comparator", supported=None)
        return
    report = dict(policy="reject", coordinate="log(K/F)", supported=True,
                  outside_quotes=0, slices=[])
    evidence["domain"] = report
    times = np.array([item[0] for item in domains])
    for t, forward, strikes in zip(chain.ttms, chain.forwards, chain.strikes_ttms):
        index = _slice_index(times, t)
        item = svi_prediction_domain(np.log(strikes / forward), domains[index][1:])
        report["slices"].append(dict(ttm=float(t), **item))
        report["outside_quotes"] += item["outside_quotes"]
        report["supported"] &= item["supported"]
    if not report["supported"]:
        raise ValueError(f"{report['outside_quotes']} quotes outside SVI prediction domain")


def fit_model(model, chain):
    """Fit training quotes; the raw_svi_price variants use half-spread price MSE."""
    ks = [np.log(k / f) for k, f in zip(chain.strikes_ttms, chain.forwards)]
    ivs = [0.5 * (b + a) for b, a in zip(chain.bid_ivs, chain.ask_ivs)]
    times = chain.ttms
    raw_fitters = {"raw_svi": fit_raw_svi, "raw_svi_legacy": fit_raw_svi_legacy,
                   "raw_svi_constrained": fit_raw_svi_constrained,
                   "raw_svi_price": fit_raw_svi_constrained,
                   "raw_svi_price_seeded": fit_raw_svi_constrained}
    if model in raw_fitters:
        fitter = raw_fitters[model]
        if model in ("raw_svi_price", "raw_svi_price_seeded"):
            if chain.bid_prices is None or chain.ask_prices is None:
                raise ValueError(f"{model} requires original training price quotes")
            fits = [fitter(k, v, t, price_quotes=dict(
                forward=chain.forwards[i], discount=chain.discfactors[i],
                optiontypes=chain.optiontypes_ttms[i],
                bid=chain.bid_prices[i], ask=chain.ask_prices[i]),
                iv_seed=model == "raw_svi_price_seeded")
                for i, (k, v, t) in enumerate(zip(ks, ivs, times))]
        else:
            fits = [fitter(k, v, t) for k, v, t in zip(ks, ivs, times)]
        params = [p for p, _ in fits]
        domains = (tuple((float(t), *e["wing_check"]["k_domain"]) for t, (_, e)
                         in zip(times, fits))
                   if model in ("raw_svi_constrained", "raw_svi_price", "raw_svi_price_seeded")
                   else ())

        def iv_at(k, t):
            index = _slice_index(times, t)
            if domains and not svi_prediction_domain(k, domains[index][1:])["supported"]:
                raise ValueError("Requested coordinates outside SVI prediction domain")
            return np.sqrt(raw_svi(k, params[index])[0] / t)

        return Fitted(_iv_prediction(iv_at), False, [asdict(p) for p in params],
                      [e for _, e in fits], domains)
    if model == "ssvi":
        fit, evidence = fit_ssvi(ks, ivs, times)
        return Fitted(_iv_prediction(lambda k, t: np.sqrt(fit.variance(k, t) / t)),
                      True, asdict(fit), evidence)
    if model == "approx_logsv":
        params = [fitters.fit_logsv_ivols(k, v, t, is_vega_weights=False)
                  for k, v, t in zip(ks, ivs, times)]

        def iv_at(k, t):
            return fitters.calc_logsv_ivols(k, **params[_slice_index(times, t)])

        return Fitted(_iv_prediction(iv_at), False, params, "native curve_fit defaults")
    if model == "gmm2":
        pricer = GmmPricer()
        params = [pricer.calibrate_model_params_to_chain_slice(
            OptionChain.get_slices_as_chain(chain, ids=[id_]),
            n_mixtures=2, is_vega_weighted=False,
        ) for id_ in chain.ids]

        def predict(target):
            prices = [np.asarray(pricer.price_chain(
                OptionChain.get_slices_as_chain(target, ids=[id_]),
                params[_slice_index(times, t)],
            )[0]) for id_, t in zip(target.ids, target.ttms)]
            return prices, target.compute_model_ivols_from_chain_data(List(prices))

        return Fitted(predict, False, [asdict(p) for p in params], "native SLSQP defaults")
    atm = float(np.interp(0, ks[0], ivs[0]))
    if model == "logsv":
        pricer = _LimitedLogSV()
        params = pricer.calibrate_model_params_to_chain(
            chain, params0=LogSvParams(sigma0=atm, theta=atm, kappa1=2.0,
                                      kappa2=None, beta=-0.2, volvol=0.5),
            is_vega_weighted=False, model_calibration_type=LogsvModelCalibrationType.PARAMS5,
            constraints_type=ConstraintsType.MMA_MARTINGALE,
        )
        scaler = pricer.set_vol_scaler(chain)

        def predict(target):
            prices = pricer.price_chain(target, params, vol_scaler=scaler)
            return prices, target.compute_model_ivols_from_chain_data(prices)

        return Fitted(predict, True, asdict(params),
                      {"optimizer": "native SLSQP defaults", "vol_scaler": scaler})
    if model == "heston":
        pricer = _LimitedHeston()
        params = pricer.calibrate_model_params_to_chain(
            chain, params0=HestonParams(v0=atm**2, theta=atm**2, kappa=2,
                                       rho=-0.2, volvol=0.3), is_vega_weighted=False,
        )
        scaler = min(0.3, np.sqrt(params.v0 * times[0]))

        def predict(target):
            prices = heston_chain_pricer(
                **asdict(params), ttms=target.ttms, forwards=target.forwards,
                discfactors=target.discfactors, strikes_ttms=target.strikes_ttms,
                optiontypes_ttms=target.optiontypes_ttms, vol_scaler=scaler,
            )
            return prices, target.compute_model_ivols_from_chain_data(prices)

        return Fitted(predict, True, asdict(params),
                      {"optimizer": "native SLSQP defaults; Feller", "vol_scaler": scaler})
    raise ValueError(model)


def _validate_predictions(chain, prices, ivs):
    """Do not let a finite-subset metric conceal any invalid held-out repricing."""
    for expected, p, v in zip(chain.strikes_ttms, prices, ivs):
        if (np.shape(p) != expected.shape or np.shape(v) != expected.shape
                or not np.all(np.isfinite(p)) or not np.all(np.isfinite(v))
                or np.any(np.asarray(p) < 0) or np.any(np.asarray(v) <= 0)):
            raise ValueError("Invalid model price/IV or mismatched evaluation universe")
    if len(prices) != len(chain.ttms) or len(ivs) != len(chain.ttms):
        raise ValueError("Model omitted an expiry")


def evaluate(fit, sample, test_mask):
    """Recompute final fit and finite-domain diagnostics after all constraints."""
    target = sample.chain(test_mask)
    prices, ivs = fit.predict(target)
    _validate_predictions(target, prices, ivs)
    report = fitters.diagnose_fit(
        target, model_prices=prices, model_ivols=ivs,
        price_unit="USD per unit", spread_floor=PRICE_FLOOR,
    )
    quotes = report.table("quotes")
    summary = report.table("summary").iloc[0]
    common = min(float(np.max(np.abs(k))) for k in sample.ks)

    def iv_at(strikes, t):
        k = np.log(strikes / _forward(t))
        grid = _quote_chain(np.array([t]), [k], [np.full(len(k), 0.28)])
        p, v = fit.predict(grid)
        _validate_predictions(grid, p, v)
        return np.asarray(v[0])

    surface = fitters.SurfaceEvaluator(
        iv=iv_at, forward=_forward, discount=_discount,
        ttm_domain=(float(sample.ttms[0]), float(sample.ttms[-1])),
        k_domain=(-common, common), underlying_id="SYNTH_SPOT",
        carry_model="proportional", interpolation=(
            "linear ATM total variance" if isinstance(fit.parameters, dict)
            and "eta" in fit.parameters else "native dynamics" if fit.continuous else "none"
        ),
    )
    extras = (sample.ttms[:-1] + sample.ttms[1:]) / 2 if fit.continuous else ()
    surface_report = fitters.diagnose_surface(
        surface, sample.ttms, np.linspace(-common, common, 61), extra_ttms=extras,
        price_unit="USD per unit", tolerance=1e-8, density_tolerance=1e-8,
    )
    wing_checks = []
    for t, k in zip(sample.ttms, sample.ks):
        grid = np.linspace(float(k[0]), float(k[-1]), 101)
        strikes = _forward(t) * np.exp(grid)
        iv = iv_at(strikes, t)
        call = black_prices(_forward(t), strikes, t, iv, _discount(t), "C")
        wing_checks.append(fitters.check_call_slice(
            strikes, call, forward=_forward(t), discount=_discount(t),
            price_unit="USD per unit", tolerance=1e-8, density_tolerance=1e-8,
        ).to_dict())
    checks = surface_report.table("checks")
    wing_failures = sum(int(fitters.DiagnosticReport.from_dict(item).table("checks")
                            .status.eq("fail").sum()) for item in wing_checks)
    metrics = {
        "n_test": len(quotes), "iv_rmse": float(summary.iv_rmse),
        "p95_spread_error": float(np.percentile(np.abs(quotes.spread_scaled_error), 95)),
        "spread_rmse": float(np.sqrt(np.mean(quotes.spread_scaled_error**2))),
        "coverage": float(summary.price_inside_bid_ask_fraction),
        "surface_failures": int(checks.status.eq("fail").sum()) + wing_failures,
        "surface_unchecked": int(checks.status.eq("not_checked").sum()),
        "continuous": fit.continuous,
    }
    return metrics, {
        "fit": report.to_dict(), "surface": surface_report.to_dict(), "wings": wing_checks,
    }, np.concatenate(prices)


def _timed_fit(model, chain):
    """Preserve native solver logs and warnings separately from wall time."""
    log = StringIO()
    start = perf_counter()
    with redirect_stdout(log), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = fit_model(model, chain)
    elapsed = perf_counter() - start
    return fit, elapsed, {"stdout": log.getvalue(), "warnings": [str(w.message) for w in caught]}


def run_job(model, sample, fold, repeats=2):
    """Retain a row on failure; a failed perturbation also fails the complete job."""
    train_mask = sample.training_mask(fold)
    row = {"model": model, "case": sample.name, "fold": fold, "status": "failed",
           "n_train": int(sum(train_mask) * len(sample.ttms)),
           "n_test": int(sum(~train_mask) * len(sample.ttms))}
    evidence = {"timings": [], "logs": [],
                "training_indices": np.flatnonzero(train_mask),
                "held_out_indices": np.flatnonzero(~train_mask)}
    start = perf_counter()
    try:
        for _ in range(repeats):
            row["stage"] = "base_fit"
            fit, elapsed, log = _timed_fit(model, sample.chain(train_mask))
            evidence["timings"].append(elapsed)
            evidence["logs"].append(log)
        row["warm_seconds"] = float(np.median(evidence["timings"]))
        evidence.update(parameters=fit.parameters, optimizer=fit.optimizer)
        row["stage"] = "base_domain"
        enforce_prediction_domain(fit, sample.chain(), evidence)
        diagnostics_start = perf_counter()
        row["stage"] = "base_diagnostics"
        metrics, reports, base_prices = evaluate(fit, sample, ~train_mask)
        row.update(metrics, diagnostics_seconds=perf_counter() - diagnostics_start)
        evidence["base"] = reports
        row["stage"] = "perturbed_fit"
        perturbed, seconds, log = _timed_fit(model, sample.chain(train_mask, perturb=True))
        evidence.update(perturbed_seconds=seconds, perturbed_log=log,
                        perturbed_parameters=perturbed.parameters,
                        perturbed_optimizer=perturbed.optimizer)
        row["stage"] = "perturbed_domain"
        domain_evidence = {}
        evidence["perturbed_domain"] = domain_evidence
        enforce_prediction_domain(perturbed, sample.chain(), domain_evidence)
        row["stage"] = "perturbed_diagnostics"
        perturbed_metrics, reports, shifted_prices = evaluate(perturbed, sample, ~train_mask)
        evidence["perturbed"] = reports
        row["perturbed_surface_failures"] = perturbed_metrics["surface_failures"]
        target = sample.chain(~train_mask)
        half_spreads = np.concatenate([
            np.maximum(0.5 * (a - b), PRICE_FLOOR)
            for a, b in zip(target.ask_prices, target.bid_prices)
        ])
        row["p95_perturbation"] = float(np.percentile(
            np.abs(shifted_prices - base_prices) / half_spreads, 95,
        ))
        row["status"] = "ok"
        row["stage"] = "complete"
        row["error"] = ""
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    row["total_seconds"] = perf_counter() - start
    row["screen_pass"] = passes_screen(row)
    return row, evidence


def passes_screen(row):
    """Apply the fixed research screen; missing/failing fits cannot pass."""
    budget = CRITERIA["dynamics_warm_seconds" if row["model"] in ("logsv", "heston")
                      else "smile_warm_seconds"]
    return bool(
        row["status"] == "ok" and row["p95_spread_error"] <= CRITERIA["p95_spread_error"]
        and row["coverage"] >= CRITERIA["coverage"] and row["iv_rmse"] <= CRITERIA["iv_rmse"]
        and row["p95_perturbation"] <= CRITERIA["p95_perturbation"]
        and row["surface_failures"] == 0 and row["perturbed_surface_failures"] == 0
        and row["warm_seconds"] <= budget
    )


def summarize(rows):
    """Show both conditional metrics and failure-inclusive screen coverage."""
    frame = pd.DataFrame(rows)
    result = []
    for model, group in frame.groupby("model", sort=False):
        valid = group[group.status.eq("ok")]
        row = {"model": model, "attempted": len(group), "failed": int(len(group) - len(valid)),
               "screen_passed": int(group.screen_pass.sum())}
        for name in ("p95_spread_error", "coverage", "iv_rmse", "p95_perturbation",
                     "warm_seconds", "surface_failures"):
            row[f"mean_{name}_successful_only"] = (
                float(valid[name].mean()) if len(valid) and name in valid else None
            )
        result.append(row)
    return result


def _json_safe(value):
    """Strict JSON output preserves missing values without nonstandard NaNs."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _save(path, payload):
    """Write auditable data, never serializing invalid JSON numbers."""
    path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")


def source_identity():
    """Record dirty source identity and numerical environment for native defaults."""
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "src/stochvolmodels").rglob("*.py"))
    paths += [Path(__file__), Path(__file__).with_name("_svi_benchmark.py"),
              Path(__file__).with_name("_svi_wing_benchmark.py")]
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        ).strip(),
        "dirty": subprocess.check_output(["git", "status", "--porcelain"], cwd=root,
                                          text=True).splitlines(),
        "sha256": {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest()
                   for p in paths},
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {p: version(p) for p in ("numpy", "scipy", "numba", "pandas",
                                              "stochvolmodels", "vanilla-option-pricers")},
    }


def run_benchmark(output=None, models=None, cases=None, repeats=2, suite="original"):
    """Run the fixed offline comparison; write every job before the next begins."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if suite not in ("original", "stability", "wings"):
        raise ValueError("Unknown benchmark suite")
    if models is None:
        models = (MODELS if suite == "original" else ("raw_svi_legacy", "raw_svi")
                  if suite == "stability" else
                  ("raw_svi_legacy", "raw_svi", "raw_svi_constrained"))
    if cases is None:
        cases = CASES if suite == "original" else STABILITY_FAMILIES
    if not models or any(m not in AVAILABLE_MODELS for m in models):
        raise ValueError("Unknown or empty model list")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output) if output is not None else (
        Path(local_path.get_output_path()) / "smile_fitter_benchmark" / stamp
    )
    output.mkdir(parents=True, exist_ok=False)
    seeds = STABILITY_SEEDS if suite == "stability" else WING_SEEDS
    factory = make_stability_sample if suite == "stability" else make_wing_sample
    samples = ([make_sample(name) for name in cases] if suite == "original" else
               [factory(name, seed) for name in cases for seed in seeds])
    metadata = {
        "started_utc": stamp, "source": source_identity(), "criteria": CRITERIA,
        "models": models, "cases": cases, "folds": FOLDS, "repeats": repeats,
        "suite": suite, "seeds": seeds if suite != "original" else None,
        "raw_svi_method": "scaled analytic Jacobian; training-relative shape bounds",
        "raw_svi_legacy_method": "original S4 finite-difference calibration",
        "raw_svi_constrained_method": "scaled SLSQP; sampled density and call-slope constraints",
        "raw_svi_price_method": "same constraints; mean squared half-spread price residuals",
        "raw_svi_price_seeded_method": "same price loss; fourth start from constrained IV fit",
        "constrained_prediction_domain": "reject outside training midpoint +/- twice half-range",
        "dynamics_objective_deadline_seconds": DYNAMICS_DEADLINE_SECONDS,
        "weights": "uniform IV SSE except raw_svi_price variants: half-spread price MSE",
        "price_unit": "USD per unit", "spread_floor": PRICE_FLOOR,
        "setup": {}, "timing_note": "warm runtime, fresh deterministic parameters each repeat",
        "limitations": ["synthetic only", "unequal optimizer budgets; seeded price adds an IV fit",
                        "finite grids are not global certificates", "no maturity extrapolation"],
    }
    _save(output / "metadata.json", metadata)
    archive = output / "run_sources"
    archive.mkdir()
    for name in (Path(__file__).name, "_svi_benchmark.py", "_svi_wing_benchmark.py"):
        (archive / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    snapshots = []
    for sample in samples:
        chain = sample.chain()
        snapshots.append({**asdict(sample), "quotes": {
            name: getattr(chain, name) for name in (
                "forwards", "discfactors", "bid_ivs", "ask_ivs", "strikes_ttms",
                "optiontypes_ttms", "bid_prices", "ask_prices",
            )
        }})
        for name, value in snapshots[-1]["quotes"].items():
            snapshots[-1]["quotes"][name] = list(value)
    _save(output / "quotes.json", snapshots)
    metadata["quotes_sha256"] = sha256((output / "quotes.json").read_bytes()).hexdigest()
    for model in models:
        print(f"Warmup {model}", flush=True)
        try:
            warmup = _quote_chain(
                np.array([0.2, 0.6, 1.2]), [np.linspace(-0.15, 0.15, 11)] * 3,
                [np.full(11, 0.31)] * 3,
            )
            _, elapsed, log = _timed_fit(model, warmup)
            metadata["setup"][model] = {"seconds": elapsed, "status": "ok", "log": log}
        except Exception as error:
            metadata["setup"][model] = {"status": "failed", "error": str(error)}
        _save(output / "metadata.json", metadata)
    rows = []
    for sample in samples:
        for fold in FOLDS:
            for model in models:
                print(f"Fit {sample.name}/{fold}/{model}", flush=True)
                row, evidence = run_job(model, sample, fold, repeats=repeats)
                rows.append(row)
                _save(output / f"{sample.name}_{fold}_{model}.json", {"row": row, **evidence})
                pd.DataFrame(rows).to_csv(output / "results.csv", index=False)
                print(f"  {row['status']}: {row.get('error', '')}", flush=True)
    summary = summarize(rows)
    _save(output / "summary.json", summary)
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"Saved {output}", flush=True)
    return output, rows


class Locals(Enum):
    """The full predeclared six-model research comparison."""

    BENCHMARK = 1


def run_local(local: Locals):
    """Run the reproducible research comparison without optional data services."""
    if local != Locals.BENCHMARK:
        raise ValueError(local)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="fresh directory for all run artifacts")
    parser.add_argument("--suite", choices=("original", "stability", "wings"), default="original")
    parser.add_argument("--models", nargs="+", choices=AVAILABLE_MODELS)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    return run_benchmark(output=args.output, suite=args.suite, models=args.models,
                         repeats=args.repeats)


if __name__ == "__main__":
    run_local(local=Locals.BENCHMARK)
