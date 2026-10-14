"""Training-only refit sensitivity; repository-only, not a reliability certificate.

Run on synthetic quotes with --draws 16, or use the optional historical runner.
Outputs retain every shock and failure. No default warning threshold is asserted.
"""

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
from numba.typed import List

from stochvolmodels import OptionChain, local_path
from compare_smile_fitters import (
    _json_safe, _validate_predictions, black_prices, enforce_prediction_domain,
    fit_model, make_sample, source_identity,
)


MODELS = ("raw_svi_constrained", "raw_svi_shared_rho")


def _copy_training(chain):
    changes = {name: np.asarray(getattr(chain, name)).copy()
               for name in ("ttms", "forwards", "discfactors", "ids")}
    for name in ("strikes_ttms", "optiontypes_ttms", "bid_ivs", "ask_ivs",
                 "bid_prices", "ask_prices"):
        values = getattr(chain, name)
        if values is not None:
            changes[name] = List([np.asarray(v).copy() for v in values])
    return replace(chain, **changes)


def _validate_training(chain):
    if (not len(chain.ttms) or np.any(np.diff(chain.ttms) <= 0)
            or chain.bid_ivs is None or chain.ask_ivs is None):
        raise ValueError("Require ordered training maturities and two-sided IVs")
    for values in (chain.ttms, chain.forwards, chain.discfactors):
        if not np.all(np.isfinite(values) & (values > 0)):
            raise ValueError("Training maturities, forwards and discounts must be finite positive")
    for k, cp, bid, ask in zip(chain.strikes_ttms, chain.optiontypes_ttms,
                               chain.bid_ivs, chain.ask_ivs):
        if (len(k) < 5 or np.any(np.diff(k) <= 0) or not np.all(np.isfinite(k) & (k > 0))
                or bid.shape != k.shape or ask.shape != k.shape or cp.shape != k.shape
                or not np.all(np.isfinite(bid) & (bid > 0))
                or not np.all(np.isfinite(ask) & (ask >= bid))
                or not np.all(np.isin(cp, ["C", "P"]))):
            raise ValueError("Require five ordered standard-option strikes and finite positive IVs")


def _perturb(chain, shocks):
    changed = _copy_training(chain)
    for i, shock in enumerate(shocks):
        changed.bid_ivs[i] += shock
        changed.ask_ivs[i] += shock
        if not np.all(np.isfinite(np.r_[changed.bid_ivs[i], changed.ask_ivs[i]])
                      & (np.r_[changed.bid_ivs[i], changed.ask_ivs[i]] > 0)):
            raise ValueError("nonpositive or nonfinite perturbed training IV; no clipping")
    for side in ("bid", "ask"):
        prices = [black_prices(f, k, t, v, d, cp) for f, k, t, v, d, cp in zip(
            changed.forwards, changed.strikes_ttms, changed.ttms,
            getattr(changed, f"{side}_ivs"), changed.discfactors, changed.optiontypes_ttms)]
        setattr(changed, f"{side}_prices", List(prices))
    return changed


def _prediction_grid(chain, fit, points):
    domains = fit.domains
    if len(domains) != len(chain.ttms):
        raise ValueError("Reliability requires an explicit finite domain for every fitted tenor")
    ks = []
    for t, f, strike, (time, low, high) in zip(chain.ttms, chain.forwards,
                                              chain.strikes_ttms, domains):
        observed = np.log(strike/f)
        if (abs(time-t) > 1e-12 or not np.all(np.isfinite([low, high])) or low >= high
                or low > observed[0]+1e-12 or high < observed[-1]-1e-12):
            raise ValueError("Declared prediction domain must include the training range")
        ks.append(np.linspace(low, high, points))
    target = OptionChain(ttms=chain.ttms.copy(), forwards=chain.forwards.copy(),
                         discfactors=chain.discfactors.copy(), ids=chain.ids.copy(),
                         strikes_ttms=List([f*np.exp(k) for f, k in zip(chain.forwards, ks)]),
                         optiontypes_ttms=List([np.full(points, "C") for _ in ks]))
    return ks, target


def _summarize(chain, ks, base_prices, base_ivs, successes, shock_std, complete):
    slices = []
    for i, (k, t, f, d, strike) in enumerate(zip(ks, chain.ttms, chain.forwards,
                                                chain.discfactors, chain.strikes_ttms)):
        observed = np.log(strike/f)
        left, right = k < observed[0]-1e-12, k > observed[-1]+1e-12
        masks = {"left_extrapolation": left, "training_range": ~(left | right),
                 "right_extrapolation": right, "combined_wings": left | right}
        grid = dict(k=k, base_iv=base_ivs[i], base_call_price=base_prices[i],
                    region=np.where(left, "left_extrapolation",
                                    np.where(right, "right_extrapolation", "training_range")))
        fields = ("iv_q05", "iv_q50", "iv_q95", "rms_iv_change", "max_abs_iv_change",
                  "rms_normalized_call_change")
        grid.update({field: None for field in fields})
        if successes:
            vols = np.asarray([a["ivs"][i] for a in successes])
            prices = np.asarray([a["prices"][i] for a in successes])
            delta = vols - base_ivs[i]
            for field, values in zip(fields[:3], np.quantile(vols, [.05, .5, .95], axis=0)):
                grid[field] = values
            grid["rms_iv_change"] = np.sqrt(np.mean(delta**2, axis=0))
            grid["max_abs_iv_change"] = np.max(abs(delta), axis=0)
            grid["rms_normalized_call_change"] = np.sqrt(
                np.mean(((prices-base_prices[i])/(f*d))**2, axis=0))
        regions = {}
        for name, mask in masks.items():
            score = (float(np.max(grid["rms_iv_change"][mask])/shock_std)
                     if complete and np.any(mask) else None)
            regions[name] = dict(points=int(mask.sum()), max_rms_amplification=score)
        slices.append(dict(expiry_id=str(chain.ids[i]), ttm=float(t), forward=float(f),
                           discount=float(d), training_k_range=[float(observed[0]),
                                                                float(observed[-1])],
                           grid=grid, regions=regions))
    return slices


def diagnose_wing_reliability(training, *, model="raw_svi_constrained", draws=16,
                              seed=5401, shock_std=0.0005, grid_points=81):
    """Measure fitted-curve sensitivity using training quotes alone.

    Parameters
    ----------
    training : OptionChain
        Training-only standard Black quotes. F/D are fixed, including when they
        were reconstructed upstream from a larger snapshot.
    model : str
        Independent constrained raw SVI or shared-rho raw SVI research comparator.
    draws, seed, shock_std, grid_points : int, int, float, int
        At least two independent shocks, nonnegative seed, positive decimal-IV
        standard deviation and an odd grid size of at least nine.

    Returns
    -------
    dict
        Base fit, every attempted shock/refit, pointwise sensitivity statistics
        and regional maximum RMS amplification. Failed draws make status
        incomplete and suppress regional scores; descriptive successful-only
        pointwise statistics remain explicitly labelled. Base failures retain
        their stage/error. Invalid configuration or training inputs raise.

    Notes
    -----
    RMS measures movement from the unperturbed curve, not standard deviation
    about the refit mean. Empirical refit quantiles are not confidence intervals.
    No warning threshold, calibration adjustment or accuracy guarantee is supplied.
    """
    if (model not in MODELS or isinstance(draws, bool) or not isinstance(draws, int) or draws < 2
            or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            or not np.isscalar(shock_std) or not np.isfinite(shock_std) or shock_std <= 0
            or isinstance(grid_points, bool) or not isinstance(grid_points, int)
            or grid_points < 9 or grid_points % 2 != 1):
        raise ValueError("Invalid reliability model, draws, seed, shock scale or grid size")
    _validate_training(training)
    snapshot = _copy_training(training)
    training_record = {name: getattr(snapshot, name) for name in
                       ("ttms", "forwards", "discfactors", "ids", "strikes_ttms",
                        "optiontypes_ttms", "bid_ivs", "ask_ivs", "bid_prices", "ask_prices")}
    training_record = {name: (value if value is None or isinstance(value, np.ndarray)
                              else list(value))
                       for name, value in training_record.items()}
    report = dict(status="failed", stage="base_fit", requested=draws, successful=0, failed=0,
                  attempts=[], slices=[], training=training_record,
                  metadata=dict(
                      model=model, seed=seed, shock_std=float(shock_std), grid_points=grid_points,
                      warning_threshold=None, statistics_scope="successful_draws_only",
                      score_policy="all requested draws must succeed",
                      price_change_unit="normalized call price C/(discount*forward)",
                      scope="training-only conditional refit sensitivity; not confidence intervals",
                      arbitrage_certificate=False, accuracy_certificate=False))
    started = perf_counter()
    base_log = StringIO()
    with warnings.catch_warnings(record=True) as caught, redirect_stdout(base_log):
        warnings.simplefilter("always")
        try:
            fit = fit_model(model, _copy_training(training))
            report["base"] = dict(parameters=fit.parameters, optimizer=fit.optimizer)
            report["stage"] = "base_grid"
            ks, target = _prediction_grid(training, fit, grid_points)
            base_prices, base_ivs = fit.predict(target)
            _validate_predictions(target, base_prices, base_ivs)
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
        report["base_warnings"] = list(dict.fromkeys(str(w.message) for w in caught))
    report.update(base_seconds=perf_counter()-started, base_stdout=base_log.getvalue())
    if "error" in report:
        return report
    rng = np.random.default_rng(seed)
    successes = []
    for index in range(draws):
        shocks = [rng.normal(0, shock_std, len(k)) for k in training.strikes_ttms]
        attempt = dict(index=index, shocks=shocks, status="failed", stage="perturb")
        started, log = perf_counter(), StringIO()
        with warnings.catch_warnings(record=True) as caught, redirect_stdout(log):
            warnings.simplefilter("always")
            try:
                changed = _perturb(training, shocks)
                attempt["stage"] = "fit"
                fitted = fit_model(model, changed)
                attempt.update(parameters=fitted.parameters, optimizer=fitted.optimizer)
                attempt["stage"] = "prediction_domain"
                enforce_prediction_domain(fitted, target, attempt)
                attempt["stage"] = "prediction"
                prices, ivs = fitted.predict(target)
                _validate_predictions(target, prices, ivs)
                attempt.update(prices=prices, ivs=ivs, status="completed", stage="completed")
                successes.append(attempt)
            except Exception as error:
                attempt["error"] = f"{type(error).__name__}: {error}"
            attempt["warnings"] = list(dict.fromkeys(str(w.message) for w in caught))
        attempt.update(seconds=perf_counter()-started, stdout=log.getvalue())
        report["attempts"].append(attempt)
    complete = len(successes) == draws
    report.update(status="complete" if complete else "incomplete", stage="completed",
                  successful=len(successes), failed=draws-len(successes))
    report["slices"] = _summarize(training, ks, base_prices, base_ivs,
                                  successes, shock_std, complete)
    return report


class Locals(Enum):
    """Repository-only training sensitivity example."""

    SYNTHETIC = 1


def run_local(local: Locals = Locals.SYNTHETIC):
    """Write synthetic sensitivity evidence without using held-out quotes."""
    if local != Locals.SYNTHETIC:
        raise ValueError(local)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default=MODELS[0])
    parser.add_argument("--case", choices=("flat", "skewed", "sparse_noisy", "short"),
                        default="skewed")
    parser.add_argument("--fold", choices=("interior", "wings"), default="wings")
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--seed", type=int, default=5401)
    parser.add_argument("--shock-std", type=float, default=0.0005)
    parser.add_argument("--research-policy", type=Path,
                        help="Caller-frozen JSON cutoff artifact; no installed default")
    parser.add_argument("--research-noise-assumption", choices=("known_iid_gaussian_iv",),
                        help="Explicit caller declaration; never inferred from bid/ask spreads")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.research_noise_assumption and args.research_policy is None:
        parser.error("--research-noise-assumption requires --research-policy")
    policy = None
    if args.research_policy is not None:
        from _wing_research_policy import load_policy
        policy, policy_hash = load_policy(args.research_policy)
        policy_bytes = args.research_policy.read_bytes()
        if sha256(policy_bytes).hexdigest() != policy_hash:
            raise ValueError("Research artifact changed while loading")
    sample = make_sample(args.case)
    training = sample.chain(sample.training_mask(args.fold))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path(local_path.get_output_path())/"wing_reliability"/stamp
    output.mkdir(parents=True, exist_ok=False)
    source = source_identity()
    root = Path(__file__).resolve().parents[2]
    source["sha256"][Path(__file__).relative_to(root).as_posix()] = sha256(
        Path(__file__).read_bytes()).hexdigest()
    if policy is not None:
        helper = Path(__file__).with_name("_wing_research_policy.py")
        source["sha256"][helper.relative_to(root).as_posix()] = sha256(
            helper.read_bytes()).hexdigest()
    archive = output/"run_sources"
    archive.mkdir()
    for name in source["sha256"]:
        if name.startswith("examples/calibration/"):
            (archive/Path(name).name).write_bytes((root/name).read_bytes())
    metadata = dict(started_utc=stamp, case=args.case, fold=args.fold, source=source)
    result = diagnose_wing_reliability(training, model=args.model, draws=args.draws,
                                      seed=args.seed, shock_std=args.shock_std)
    if policy is not None:
        from _wing_research_policy import assess_research_policy
        metadata["research_policy"] = dict(artifact=policy, sha256=policy_hash,
                                            noise_assumption=args.research_noise_assumption)
        (output/"research_policy.json").write_bytes(policy_bytes)
        result["research_assessment"] = assess_research_policy(
            result, policy, noise_assumption=args.research_noise_assumption)
    (output/"report.json").write_text(json.dumps(_json_safe(dict(metadata=metadata, result=result)),
                                                 allow_nan=False, indent=2), encoding="utf-8")
    print(result["status"], output/"report.json")


if __name__ == "__main__":
    run_local()
