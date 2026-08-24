"""Experiments for the discrete-versus-continuous TGARCH study.

All volatilities and rates are annualised, time is measured in years, and
returns are log returns.  This repository-only module intentionally keeps
simulation and reporting outside the public :mod:`stochvolmodels` package.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numba import njit
from scipy.integrate import quad
from scipy.optimize import brentq, minimize
from scipy.stats import gaussian_kde, geninvgauss, invgamma

import stochvolmodels as svm
from stochvolmodels.products.payoffs import EuropeanOptionPayoff
from stochvolmodels.valuation import PathEstimator, PathValuationResult, value_paths
from volatility_book.ch_discrete_vol.sim import (
    M1,
    S1,
    LimitParams,
    Measure,
    SimulationResult,
    TgarchParams,
    derived_limit_params,
    filter_discrete_returns,
    run_unit_checks,
    simulate_discrete_path,
    simulate_stationary_sigma,
    simulate_terminal,
    simulate_two_shock_limit_path,
)

DT_GRID = (1.0 / 52.0, 1.0 / 252.0, 1.0 / 1008.0, 1.0 / 4032.0, 1.0 / 16128.0)
MATURITIES = (1.0 / 12.0, 1.0 / 4.0)
MONEYNESS_MULTIPLIERS = np.array((-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0))
KAPPA2_MOMENT_GRID = (0.0, 0.5, 1.0, 2.0, 4.25)
MOMENT_POWERS = (1.25, 1.5)
BASE_SEED = 20260823


class StudyProfile(str, Enum):
    """Execution size for development, reference analysis, or the full brief."""

    SMOKE = "smoke"
    REFERENCE = "reference"
    FULL = "full"


@dataclass(frozen=True)
class StudyConfig:
    """Numerical workload selected by :class:`StudyProfile`."""

    profile: StudyProfile
    dt_grid: tuple[float, ...]
    maturities: tuple[float, ...]
    unit_paths: int
    moment_check_draws: int
    e4_paths: int
    bootstrap_replications: int
    stationary_burn_years: float
    stationary_sample_years: float
    e5_sample_years: float
    e6_years: float
    e7_replications: int
    e7_parameter_sets: tuple[str, ...]

    def pricing_paths(self, dt: float) -> int:
        """Return the brief's path count, or its profile-specific development analogue."""
        if self.profile is StudyProfile.FULL:
            return 2**19 if dt >= 1.0 / 1008.0 - 1.0e-15 else 2**17
        if self.profile is StudyProfile.REFERENCE:
            return 2**15 if dt >= 1.0 / 1008.0 - 1.0e-15 else 2**14
        return 2**11


@dataclass
class StudyResults:
    """In-memory results used by the Markdown, JSON, and PDF writers."""

    profile: str
    config: dict[str, Any]
    parameters: dict[str, Any]
    checks: dict[str, Any]
    experiments: dict[str, Any]
    provenance: dict[str, Any]
    warnings: list[str]


def make_study_config(profile: StudyProfile) -> StudyConfig:
    """Create a validated workload configuration."""
    if not isinstance(profile, StudyProfile):
        raise ValueError("profile must be a StudyProfile")
    if profile is StudyProfile.FULL:
        return StudyConfig(
            profile=profile,
            dt_grid=DT_GRID,
            maturities=MATURITIES,
            unit_paths=2**16,
            moment_check_draws=2**20,
            e4_paths=2**17,
            bootstrap_replications=400,
            stationary_burn_years=50.0,
            stationary_sample_years=500.0,
            e5_sample_years=1000.0,
            e6_years=40.0,
            e7_replications=200,
            e7_parameter_sets=("crypto", "equity"),
        )
    if profile is StudyProfile.REFERENCE:
        return StudyConfig(
            profile=profile,
            dt_grid=DT_GRID,
            maturities=MATURITIES,
            unit_paths=2**13,
            moment_check_draws=2**18,
            e4_paths=2**14,
            bootstrap_replications=200,
            stationary_burn_years=15.0,
            stationary_sample_years=150.0,
            e5_sample_years=300.0,
            e6_years=20.0,
            e7_replications=32,
            e7_parameter_sets=("crypto", "equity"),
        )
    return StudyConfig(
        profile=profile,
        dt_grid=(1.0 / 52.0, 1.0 / 1008.0),
        maturities=(1.0 / 4.0,),
        unit_paths=2**10,
        moment_check_draws=2**14,
        e4_paths=2**10,
        bootstrap_replications=40,
        stationary_burn_years=2.0,
        stationary_sample_years=8.0,
        e5_sample_years=20.0,
        e6_years=3.0,
        e7_replications=2,
        e7_parameter_sets=("crypto",),
    )


def parameter_sets() -> dict[str, TgarchParams]:
    """Return the two parameter sets specified by the brief."""
    shared = {
        "r": 0.0,
        "gamma0": 0.0,
        "gamma1": 0.5,
        "eta0": 0.0,
        "eta1": -0.38,
        "spot0": 1.0,
    }
    return {
        "crypto": TgarchParams(
            theta=0.6,
            kappa1=3.0,
            kappa2=3.0,
            beta=1.0,
            eps=1.5,
            sigma0=0.6,
            **shared,
        ),
        "equity": TgarchParams(
            theta=0.2,
            kappa1=4.0,
            kappa2=4.0,
            beta=-1.0,
            eps=1.0,
            sigma0=0.2,
            **shared,
        ),
    }


def _as_jsonable(value: Any) -> Any:
    """Convert NumPy and dataclass values to JSON-compatible Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: _as_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_normalized_text(path: Path) -> str | None:
    """Hash text using the repository's LF-normalized Git-content convention."""
    if not path.exists():
        return None
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def _git_output(repository: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def collect_provenance(output_dir: Path, profile: StudyProfile) -> dict[str, Any]:
    """Collect source hashes, versions, seeds, and the enclosing repository revision."""
    repo = Path(__file__).resolve().parents[2]
    code_dir = Path(__file__).resolve().parent
    notes_dir = code_dir / "notes"
    source_files = ("sim.py", "experiments.py", "reporting.py", "run_study.py")
    packages = ("numpy", "scipy", "matplotlib", "numba", "pandas", "stochvolmodels")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "script": "python -m volatility_book.ch_discrete_vol.run_study",
        "profile": profile.value,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "package_versions": versions,
        "repository_head": _git_output(repo, "rev-parse", "HEAD"),
        "repository_describe": _git_output(repo, "describe", "--tags", "--always", "--dirty"),
        "repository_status": _git_output(repo, "status", "--short"),
        "study_folder_git_tag": "N/A: the Volatility Book folder is not a Git repository",
        "code_sha256": {name: _sha256(code_dir / name) for name in source_files},
        "input_sha256": {
            "brief": _sha256_normalized_text(
                notes_dir / "SOL_BRIEF_discrete_vs_continuous_tgarch_study.md"
            ),
            "note": _sha256_normalized_text(notes_dir / "tgarch_quadratic_drift_note.tex"),
        },
        "seeds": {f"E{index}": BASE_SEED + index for index in range(1, 8)},
    }


def _limit_with_kappa2(base: LimitParams, kappa2_hat: float) -> LimitParams:
    """Hold ``d0`` and ``d1_hat`` fixed while changing the quadratic coefficient."""
    if not np.isfinite(kappa2_hat) or kappa2_hat < 0.0:
        raise ValueError("kappa2_hat must be finite and non-negative")
    return LimitParams.from_drift_coefficients(
        d0=base.d0,
        d1_hat=base.d1_hat,
        kappa2_hat=kappa2_hat,
        lambda0_bar=base.lambda0_bar,
        lambda1_bar=base.lambda1_bar,
        vartheta=base.vartheta,
    )


def _pair_arrays(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 1 or values.size % 2 != 0:
        raise ValueError("antithetic values must be a one-dimensional even-length array")
    half = values.size // 2
    return values[:half], values[half:]


def _antithetic_group_ids(n_paths: int) -> np.ndarray:
    """Declare the simulator's first-half/second-half antithetic provenance."""
    if isinstance(n_paths, (bool, np.bool_)) or not isinstance(n_paths, (int, np.integer)):
        raise ValueError("n_paths must be an integer")
    if n_paths < 4 or n_paths % 2:
        raise ValueError("antithetic valuation requires an even n_paths of at least four")
    return np.tile(np.arange(n_paths // 2, dtype=np.int64), 2)


def _discounted_pair_values(
    values: np.ndarray,
    discount: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return discounted antithetic pair contributions, optionally raw-LR weighted."""
    left, right = _pair_arrays(np.asarray(values, dtype=np.float64))
    left = discount * left
    right = discount * right
    if weights is not None:
        weight_left, weight_right = _pair_arrays(np.asarray(weights, dtype=np.float64))
        if weight_left.shape != left.shape:
            raise ValueError("weights must have the same length as values")
        left = weight_left * left
        right = weight_right * right
    return 0.5 * (left + right)


def _controlled_call_pair_values(
    call_values: np.ndarray,
    spot_values: np.ndarray,
    discount: float,
    expected_discounted_spot: float,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Return antithetic pair contributions after the terminal-spot control variate."""
    pair_payoff = _discounted_pair_values(call_values, discount, weights)
    pair_spot = _discounted_pair_values(spot_values, discount, weights)
    spot_variance = float(np.var(pair_spot, ddof=1))
    if spot_variance <= 1.0e-30:
        coefficient = 0.0
    else:
        coefficient = float(np.cov(pair_payoff, pair_spot, ddof=1)[0, 1] / spot_variance)
    adjusted = pair_payoff - coefficient * (pair_spot - expected_discounted_spot)
    return adjusted, coefficient


def _mean_and_se(pair_values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(pair_values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("at least two antithetic pair values are required")
    return float(np.mean(values)), float(np.std(values, ddof=1) / math.sqrt(values.size))


def _assert_close(actual: float, expected: float, label: str) -> None:
    """Enforce a tight machine-roundoff invariant between adopted and local estimators."""
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise RuntimeError(f"{label} must be finite: actual={actual!r}, expected={expected!r}")
    if not math.isclose(actual, expected, rel_tol=2.0e-13, abs_tol=2.0e-15):
        raise RuntimeError(f"{label} mismatch: actual={actual!r}, expected={expected!r}")


def _value_european_leg(
    *,
    simulated: SimulationResult,
    payoff: EuropeanOptionPayoff,
    discount: float,
    independent_group_ids: np.ndarray,
    expected_estimator: PathEstimator,
) -> tuple[np.ndarray, PathValuationResult]:
    """Evaluate one package-owned payoff and validate its raw valuation contract."""
    payoff_values = payoff(simulated.paths)
    result = value_paths(
        paths=simulated.paths,
        payoff=payoff,
        discount_factor=discount,
        independent_group_ids=independent_group_ids,
    )
    expected_groups = simulated.n_paths // 2
    if result.estimator is not expected_estimator:
        raise RuntimeError(
            f"unexpected estimator: {result.estimator.value}, expected {expected_estimator.value}"
        )
    if result.standard_error_basis != "independent_groups":
        raise RuntimeError("antithetic valuation must use independent-group standard errors")
    if (
        result.n_paths != simulated.n_paths
        or result.n_independent_groups != expected_groups
        or result.group_size != 2
    ):
        raise RuntimeError("valuation did not preserve the declared antithetic grouping")
    if result.settlement_unit != payoff.settlement_unit:
        raise RuntimeError("valuation settlement unit differs from the payoff unit")
    if result.recenter_shift is not None:
        raise RuntimeError("chapter valuations must not apply legacy forward recentering")
    if expected_estimator is PathEstimator.MONTE_CARLO:
        if result.mean_likelihood_ratio is not None or result.log_mean_likelihood_ratio is not None:
            raise RuntimeError("direct Monte Carlo valuation unexpectedly used likelihood ratios")
        _assert_close(result.path_effective_sample_size, float(simulated.n_paths), "path ESS")
        _assert_close(result.group_effective_sample_size, float(expected_groups), "group ESS")
    else:
        if expected_estimator is not PathEstimator.RAW_LIKELIHOOD_RATIO:
            raise RuntimeError("chapter valuation supports only raw likelihood-ratio weighting")
        if result.mean_likelihood_ratio is None or result.log_mean_likelihood_ratio is None:
            raise RuntimeError("raw likelihood-ratio diagnostics are missing")
        if not (
            0.0 < result.path_effective_sample_size <= simulated.n_paths * (1.0 + 1.0e-12)
            and 0.0
            < result.group_effective_sample_size
            <= expected_groups * (1.0 + 1.0e-12)
        ):
            raise RuntimeError("raw likelihood-ratio ESS lies outside its valid range")
    _assert_close(
        result.path_ess_fraction,
        result.path_effective_sample_size / simulated.n_paths,
        "path ESS fraction",
    )
    _assert_close(
        result.group_ess_fraction,
        result.group_effective_sample_size / expected_groups,
        "group ESS fraction",
    )
    return payoff_values, result


def _assert_pair_valuation(
    result: PathValuationResult,
    pair_values: np.ndarray,
    label: str,
) -> None:
    """Cross-check package value and grouped SE against chapter-local pair contributions."""
    local_value, local_standard_error = _mean_and_se(pair_values)
    _assert_close(result.value, local_value, f"{label} value")
    _assert_close(result.standard_error, local_standard_error, f"{label} standard error")


def _assert_control_identity(
    *,
    adjusted_value: float,
    call_result: PathValuationResult,
    spot_result: PathValuationResult,
    coefficient: float,
    expected_discounted_spot: float,
    label: str,
) -> None:
    """Check the raw fitted-control identity without normalizing likelihood weights."""
    expected = call_result.value - coefficient * (
        spot_result.value - expected_discounted_spot
    )
    _assert_close(adjusted_value, expected, f"{label} controlled value")


def _ivols_and_se(
    *,
    maturity: float,
    forward: float,
    discount: float,
    strikes: np.ndarray,
    prices: np.ndarray,
    price_se: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    optiontypes = np.full(strikes.size, "C")
    ivols = svm.infer_bsm_ivols_from_slice_prices(
        ttm=maturity,
        forward=forward,
        discfactor=discount,
        strikes=strikes,
        optiontypes=optiontypes,
        model_prices=prices,
    )
    vegas = np.array(
        [
            svm.compute_bsm_vanilla_vega(
                ttm=maturity,
                forward=forward,
                strike=float(strike),
                vol=float(ivol),
                discfactor=discount,
            )
            for strike, ivol in zip(strikes, ivols)
        ]
    )
    ivol_se = np.divide(
        price_se,
        vegas,
        out=np.full_like(price_se, np.nan),
        where=vegas > 1.0e-12,
    )
    return ivols, ivol_se


def _affine_slice(
    *,
    params: TgarchParams,
    limit: LimitParams,
    maturity: float,
    strikes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model_params = svm.LogSvParams(
        sigma0=params.sigma0,
        theta=limit.theta_hat,
        kappa1=limit.kappa1_hat,
        kappa2=limit.kappa2_hat,
        beta=params.beta,
        volvol=params.eps,
    )
    forward = params.spot0 * math.exp(params.r * maturity)
    discount = math.exp(-params.r * maturity)
    return svm.LogSVPricer().price_slice(
        params=model_params,
        ttm=maturity,
        forward=forward,
        strikes=strikes,
        optiontypes=np.full(strikes.size, "C"),
        discfactor=discount,
    )


def run_e1(
    params_by_name: dict[str, TgarchParams],
    config: StudyConfig,
) -> dict[str, Any]:
    """E1: compare exact discrete option IVs with the continuous affine expansion."""
    records: list[dict[str, Any]] = []
    seed = BASE_SEED + 1
    for name, params in params_by_name.items():
        limit = derived_limit_params(params)
        for maturity in config.maturities:
            strikes = params.spot0 * np.exp(
                params.sigma0 * math.sqrt(maturity) * MONEYNESS_MULTIPLIERS
            )
            affine_prices, affine_ivols = _affine_slice(
                params=params,
                limit=limit,
                maturity=maturity,
                strikes=strikes,
            )
            discount = math.exp(-params.r * maturity)
            forward = params.spot0 * math.exp(params.r * maturity)
            spot_payoff = EuropeanOptionPayoff(
                asset_id="spot",
                expiry=maturity,
                strike=0.0,
                option_type="C",
                unit="spot units",
            )
            call_payoffs = tuple(
                EuropeanOptionPayoff(
                    asset_id="spot",
                    expiry=maturity,
                    strike=float(strike),
                    option_type="C",
                    unit="spot units",
                )
                for strike in strikes
            )
            for dt in config.dt_grid:
                n_paths = config.pricing_paths(dt)
                simulated = simulate_terminal(
                    params=params,
                    measure=Measure.Q_EXACT,
                    maturity=maturity,
                    max_dt=dt,
                    n_paths=n_paths,
                    seed=seed,
                )
                group_ids = _antithetic_group_ids(n_paths)
                spot_values, spot_result = _value_european_leg(
                    simulated=simulated,
                    payoff=spot_payoff,
                    discount=discount,
                    independent_group_ids=group_ids,
                    expected_estimator=PathEstimator.MONTE_CARLO,
                )
                _assert_pair_valuation(
                    spot_result,
                    _discounted_pair_values(spot_values, discount),
                    "E1 discounted spot",
                )
                coefficients: list[float] = []
                prices = np.empty(strikes.size)
                price_se = np.empty(strikes.size)
                for index, call_payoff in enumerate(call_payoffs):
                    call_values, call_result = _value_european_leg(
                        simulated=simulated,
                        payoff=call_payoff,
                        discount=discount,
                        independent_group_ids=group_ids,
                        expected_estimator=PathEstimator.MONTE_CARLO,
                    )
                    _assert_pair_valuation(
                        call_result,
                        _discounted_pair_values(call_values, discount),
                        "E1 call",
                    )
                    adjusted, coefficient = _controlled_call_pair_values(
                        call_values=call_values,
                        spot_values=spot_values,
                        discount=discount,
                        expected_discounted_spot=params.spot0,
                    )
                    coefficients.append(coefficient)
                    prices[index], price_se[index] = _mean_and_se(adjusted)
                    _assert_control_identity(
                        adjusted_value=prices[index],
                        call_result=call_result,
                        spot_result=spot_result,
                        coefficient=coefficient,
                        expected_discounted_spot=params.spot0,
                        label="E1",
                    )
                mc_ivols, mc_ivol_se = _ivols_and_se(
                    maturity=maturity,
                    forward=forward,
                    discount=discount,
                    strikes=strikes,
                    prices=prices,
                    price_se=price_se,
                )
                error_bp = 1.0e4 * (mc_ivols - affine_ivols)
                records.append(
                    {
                        "parameter_set": name,
                        "maturity": maturity,
                        "dt": dt,
                        "n_steps": simulated.n_steps,
                        "n_paths": n_paths,
                        "seed": seed,
                        "strikes": strikes,
                        "mc_prices": prices,
                        "mc_price_se": price_se,
                        "mc_ivols": mc_ivols,
                        "mc_ivol_se": mc_ivol_se,
                        "affine_prices": affine_prices,
                        "affine_ivols": affine_ivols,
                        "error_bp": error_bp,
                        "mean_abs_error_bp": float(np.nanmean(np.abs(error_bp))),
                        "max_abs_error_bp": float(np.nanmax(np.abs(error_bp))),
                        "mean_ivol_se_bp": float(1.0e4 * np.nanmean(mc_ivol_se)),
                        "max_ivol_se_bp": float(1.0e4 * np.nanmax(mc_ivol_se)),
                        "control_coefficients": np.asarray(coefficients),
                        "floor_hits": simulated.floor_hits,
                        "kernel_invalid_hits": 0,
                    }
                )

    summaries: list[dict[str, Any]] = []
    finest_table: list[dict[str, Any]] = []
    for name in params_by_name:
        for maturity in config.maturities:
            group = sorted(
                (
                    row
                    for row in records
                    if row["parameter_set"] == name and row["maturity"] == maturity
                ),
                key=lambda row: row["dt"],
                reverse=True,
            )
            within_noise = True
            for coarse, fine in zip(group[:-1], group[1:]):
                noise = 2.0 * (coarse["mean_ivol_se_bp"] + fine["mean_ivol_se_bp"])
                within_noise &= fine["mean_abs_error_bp"] <= coarse["mean_abs_error_bp"] + noise
            finest = group[-1]
            band_pass = bool(
                np.all(
                    np.abs(finest["error_bp"])
                    <= 50.0 + 2.0e4 * finest["mc_ivol_se"]
                )
            )
            summaries.append(
                {
                    "parameter_set": name,
                    "maturity": maturity,
                    "monotone_within_mc_noise": bool(within_noise),
                    "finest_within_50bp_plus_2strike_se_band": bool(band_pass),
                    "acceptance_pass": bool(within_noise and band_pass),
                    "finest_mean_abs_error_bp": finest["mean_abs_error_bp"],
                    "finest_max_abs_error_bp": finest["max_abs_error_bp"],
                    "finest_mean_ivol_se_bp": finest["mean_ivol_se_bp"],
                }
            )
            for index, strike in enumerate(finest["strikes"]):
                finest_table.append(
                    {
                        "parameter_set": name,
                        "maturity": maturity,
                        "dt": finest["dt"],
                        "strike": float(strike),
                        "mc_ivol": float(finest["mc_ivols"][index]),
                        "mc_ivol_se": float(finest["mc_ivol_se"][index]),
                        "affine_ivol": float(finest["affine_ivols"][index]),
                        "error_bp": float(finest["error_bp"][index]),
                    }
                )
    return {
        "claim": "Q_EXACT option IVs converge to the continuous affine-expansion benchmark",
        "seed": seed,
        "records": records,
        "summaries": summaries,
        "finest_table": finest_table,
        "acceptance_pass": bool(all(row["acceptance_pass"] for row in summaries)),
        "acceptance_criterion": (
            "Mean absolute errors must decline within Monte Carlo noise, and each "
            "finest-step strike error must lie within 50 bp plus two of its own MC SE."
        ),
    }


def _raw_likelihood_weights(log_weights: np.ndarray | None) -> np.ndarray:
    """Restore raw likelihood ratios for chapter-local paired-control contributions."""
    if not isinstance(log_weights, np.ndarray):
        raise ValueError("log_weights must be a NumPy array")
    logs = np.asarray(log_weights, dtype=np.float64)
    if logs.ndim != 1 or not np.all(np.isfinite(logs)):
        raise ValueError("log_weights must be a finite one-dimensional array")
    maximum = float(np.max(logs))
    scaled = np.exp(logs - maximum)
    scale = math.exp(maximum)
    return scale * scaled


def run_e2(
    params_by_name: dict[str, TgarchParams],
    config: StudyConfig,
) -> dict[str, Any]:
    """E2: compare direct Q, P-to-Q weighting, and the hatted limit recursion."""
    maturity = max(config.maturities)
    seed = BASE_SEED + 2
    records: list[dict[str, Any]] = []
    for name, params in params_by_name.items():
        limit = derived_limit_params(params)
        strike = params.spot0
        discount = math.exp(-params.r * maturity)
        call_payoff = EuropeanOptionPayoff(
            asset_id="spot",
            expiry=maturity,
            strike=strike,
            option_type="C",
            unit="spot units",
        )
        spot_payoff = EuropeanOptionPayoff(
            asset_id="spot",
            expiry=maturity,
            strike=0.0,
            option_type="C",
            unit="spot units",
        )
        for dt in config.dt_grid:
            n_paths = config.pricing_paths(dt)
            exact = simulate_terminal(
                params=params,
                measure=Measure.Q_EXACT,
                maturity=maturity,
                max_dt=dt,
                n_paths=n_paths,
                seed=seed,
            )
            physical = simulate_terminal(
                params=params,
                measure=Measure.P,
                maturity=maturity,
                max_dt=dt,
                n_paths=n_paths,
                seed=seed,
                track_log_weights=True,
            )
            limit_sim = simulate_terminal(
                params=params,
                measure=Measure.Q_LIMIT,
                maturity=maturity,
                max_dt=dt,
                n_paths=n_paths,
                seed=seed,
                limit_params=limit,
            )
            group_ids = _antithetic_group_ids(n_paths)
            weights = _raw_likelihood_weights(physical.log_weights)
            exact_call_values, exact_call_result = _value_european_leg(
                simulated=exact,
                payoff=call_payoff,
                discount=discount,
                independent_group_ids=group_ids,
                expected_estimator=PathEstimator.MONTE_CARLO,
            )
            exact_spot_values, exact_spot_result = _value_european_leg(
                simulated=exact,
                payoff=spot_payoff,
                discount=discount,
                independent_group_ids=group_ids,
                expected_estimator=PathEstimator.MONTE_CARLO,
            )
            weighted_call_values, weighted_call_result = _value_european_leg(
                simulated=physical,
                payoff=call_payoff,
                discount=discount,
                independent_group_ids=group_ids,
                expected_estimator=PathEstimator.RAW_LIKELIHOOD_RATIO,
            )
            weighted_spot_values, weighted_spot_result = _value_european_leg(
                simulated=physical,
                payoff=spot_payoff,
                discount=discount,
                independent_group_ids=group_ids,
                expected_estimator=PathEstimator.RAW_LIKELIHOOD_RATIO,
            )
            limit_call_values, limit_call_result = _value_european_leg(
                simulated=limit_sim,
                payoff=call_payoff,
                discount=discount,
                independent_group_ids=group_ids,
                expected_estimator=PathEstimator.MONTE_CARLO,
            )
            limit_spot_values, limit_spot_result = _value_european_leg(
                simulated=limit_sim,
                payoff=spot_payoff,
                discount=discount,
                independent_group_ids=group_ids,
                expected_estimator=PathEstimator.MONTE_CARLO,
            )
            _assert_pair_valuation(
                exact_call_result,
                _discounted_pair_values(exact_call_values, discount),
                "E2 exact call",
            )
            _assert_pair_valuation(
                exact_spot_result,
                _discounted_pair_values(exact_spot_values, discount),
                "E2 exact spot",
            )
            _assert_pair_valuation(
                weighted_call_result,
                _discounted_pair_values(weighted_call_values, discount, weights),
                "E2 weighted call",
            )
            _assert_pair_valuation(
                weighted_spot_result,
                _discounted_pair_values(weighted_spot_values, discount, weights),
                "E2 weighted spot",
            )
            _assert_pair_valuation(
                limit_call_result,
                _discounted_pair_values(limit_call_values, discount),
                "E2 limit call",
            )
            _assert_pair_valuation(
                limit_spot_result,
                _discounted_pair_values(limit_spot_values, discount),
                "E2 limit spot",
            )
            exact_pairs, exact_cv = _controlled_call_pair_values(
                call_values=exact_call_values,
                spot_values=exact_spot_values,
                discount=discount,
                expected_discounted_spot=params.spot0,
            )
            weighted_pairs, weighted_cv = _controlled_call_pair_values(
                call_values=weighted_call_values,
                spot_values=weighted_spot_values,
                discount=discount,
                expected_discounted_spot=params.spot0,
                weights=weights,
            )
            limit_pairs, limit_cv = _controlled_call_pair_values(
                call_values=limit_call_values,
                spot_values=limit_spot_values,
                discount=discount,
                expected_discounted_spot=params.spot0,
            )
            exact_price, exact_se = _mean_and_se(exact_pairs)
            weighted_price, weighted_se = _mean_and_se(weighted_pairs)
            limit_price, limit_se = _mean_and_se(limit_pairs)
            exact_minus_weighted = exact_pairs - weighted_pairs
            limit_minus_exact = limit_pairs - exact_pairs
            difference_ab, difference_ab_se = _mean_and_se(exact_minus_weighted)
            difference_ca, difference_ca_se = _mean_and_se(limit_minus_exact)
            _assert_control_identity(
                adjusted_value=exact_price,
                call_result=exact_call_result,
                spot_result=exact_spot_result,
                coefficient=exact_cv,
                expected_discounted_spot=params.spot0,
                label="E2 exact",
            )
            _assert_control_identity(
                adjusted_value=weighted_price,
                call_result=weighted_call_result,
                spot_result=weighted_spot_result,
                coefficient=weighted_cv,
                expected_discounted_spot=params.spot0,
                label="E2 weighted raw-LR",
            )
            _assert_control_identity(
                adjusted_value=limit_price,
                call_result=limit_call_result,
                spot_result=limit_spot_result,
                coefficient=limit_cv,
                expected_discounted_spot=params.spot0,
                label="E2 limit",
            )
            mean_weight = weighted_call_result.mean_likelihood_ratio
            spot_mean_weight = weighted_spot_result.mean_likelihood_ratio
            if mean_weight is None or spot_mean_weight is None:
                raise RuntimeError("E2 raw likelihood-ratio mean diagnostic is missing")
            _assert_close(spot_mean_weight, mean_weight, "E2 call/spot mean likelihood ratio")
            _assert_close(
                weighted_spot_result.path_effective_sample_size,
                weighted_call_result.path_effective_sample_size,
                "E2 call/spot path ESS",
            )
            ess = weighted_call_result.path_effective_sample_size
            if physical.effective_sample_size is None or physical.ess_fraction is None:
                raise RuntimeError("E2 simulator likelihood-ratio diagnostics are missing")
            _assert_close(ess, physical.effective_sample_size, "E2 valuation/simulator path ESS")
            _assert_close(
                weighted_call_result.path_ess_fraction,
                physical.ess_fraction,
                "E2 valuation/simulator ESS fraction",
            )
            records.append(
                {
                    "parameter_set": name,
                    "maturity": maturity,
                    "strike": strike,
                    "dt": dt,
                    "n_paths": n_paths,
                    "seed": seed,
                    "exact_price": exact_price,
                    "exact_se": exact_se,
                    "weighted_price": weighted_price,
                    "weighted_se": weighted_se,
                    "limit_price": limit_price,
                    "limit_se": limit_se,
                    "exact_minus_weighted": difference_ab,
                    "exact_minus_weighted_se": difference_ab_se,
                    "limit_minus_exact": difference_ca,
                    "limit_minus_exact_se": difference_ca_se,
                    "mean_likelihood_weight": mean_weight,
                    "ess": ess,
                    "ess_fraction": ess / n_paths,
                    "ess_warning": ess / n_paths < 0.2,
                    "ab_within_3se": abs(difference_ab) <= 3.0 * difference_ab_se,
                    "control_coefficients": {
                        "exact": exact_cv,
                        "weighted": weighted_cv,
                        "limit": limit_cv,
                    },
                    "floor_hits": {
                        "exact": exact.floor_hits,
                        "physical": physical.floor_hits,
                        "limit": limit_sim.floor_hits,
                    },
                }
            )

    rates: list[dict[str, Any]] = []
    for name in params_by_name:
        group = sorted(
            (row for row in records if row["parameter_set"] == name),
            key=lambda row: row["dt"],
        )
        x = np.log(np.array([row["dt"] for row in group]))
        y_raw = np.array([abs(row["limit_minus_exact"]) for row in group])
        noise = np.array([row["limit_minus_exact_se"] for row in group])
        usable = y_raw > noise
        if np.count_nonzero(usable) >= 2:
            slope, intercept = np.polyfit(x[usable], np.log(y_raw[usable]), 1)
        else:
            slope, intercept = np.nan, np.nan
        rates.append(
            {
                "parameter_set": name,
                "fitted_rate": float(slope),
                "fitted_intercept": float(intercept),
                "usable_points": int(np.count_nonzero(usable)),
                "sqrt_dt_consistent": bool(np.isfinite(slope) and 0.2 <= slope <= 0.8),
            }
        )
    return {
        "claim": "kernel algebra gives the same Q law and Q_LIMIT closes at sqrt(dt) order",
        "seed": seed,
        "maturity": maturity,
        "records": records,
        "rates": rates,
        "acceptance_pass": bool(
            all(row["ab_within_3se"] for row in records)
            and all(row["sqrt_dt_consistent"] for row in rates)
        ),
    }


def _gig_distribution(limit: LimitParams, vartheta2: float) -> Any:
    lam = 2.0 * limit.d1_hat / vartheta2 - 1.0
    chi = 4.0 * limit.d0 / vartheta2
    psi = 4.0 * limit.kappa2_hat / vartheta2
    if psi <= 0.0:
        shape = 1.0 - 2.0 * limit.d1_hat / vartheta2
        scale = 2.0 * limit.d0 / vartheta2
        return invgamma(a=shape, scale=scale)
    return geninvgauss(
        p=lam,
        b=math.sqrt(chi * psi),
        scale=math.sqrt(chi / psi),
    )


def _density_diagnostic(
    samples: np.ndarray,
    distribution: Any,
    max_kde_samples: int = 30_000,
) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size < 100:
        raise ValueError("at least 100 positive stationary samples are required")
    lower = float(distribution.ppf(0.001))
    upper = float(distribution.ppf(0.999))
    grid = np.geomspace(max(lower, 1.0e-8), upper, 350)
    stride = max(1, values.size // max_kde_samples)
    kde_values = values[::stride][:max_kde_samples]
    kde = gaussian_kde(kde_values)
    estimated_density = kde(grid)
    theoretical_density = distribution.pdf(grid)
    sup_error = float(np.max(np.abs(estimated_density - theoretical_density)))
    max_density = float(np.max(theoretical_density))
    probabilities = np.linspace(0.01, 0.99, 99)
    sample_quantiles = np.quantile(values, probabilities)
    theoretical_quantiles = distribution.ppf(probabilities)
    qq_relative_rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    (sample_quantiles - theoretical_quantiles)
                    / np.maximum(theoretical_quantiles, 1.0e-12)
                )
            )
        )
    )
    return {
        "n_samples": values.size,
        "grid": grid,
        "estimated_density": estimated_density,
        "theoretical_density": theoretical_density,
        "sup_density_error": sup_error,
        "relative_sup_density_error": sup_error / max_density,
        "probabilities": probabilities,
        "sample_quantiles": sample_quantiles,
        "theoretical_quantiles": theoretical_quantiles,
        "qq_relative_rmse": qq_relative_rmse,
    }


def run_e3(
    params_by_name: dict[str, TgarchParams],
    config: StudyConfig,
) -> dict[str, Any]:
    """E3: compare the stationary Q_LIMIT recursion with its GIG law."""
    seed = BASE_SEED + 3
    records: list[dict[str, Any]] = []
    comparison_dts = (
        (1.0 / 252.0, 1.0 / 1008.0)
        if config.profile is not StudyProfile.SMOKE
        else (1.0 / 252.0,)
    )
    for name, params in params_by_name.items():
        base = derived_limit_params(params)
        variants = (
            ("baseline", base),
            ("kappa2_5pct", _limit_with_kappa2(base, 0.05 * base.kappa2_hat)),
        )
        for variant_name, limit in variants:
            dts = comparison_dts if variant_name == "baseline" else (1.0 / 252.0,)
            distribution = _gig_distribution(limit, params.vartheta**2)
            for dt in dts:
                simulated = simulate_stationary_sigma(
                    params=params,
                    measure=Measure.Q_LIMIT,
                    dt=dt,
                    burn_years=config.stationary_burn_years,
                    sample_years=config.stationary_sample_years,
                    sample_interval=1.0 / 252.0,
                    seed=seed,
                    limit_params=limit,
                )
                diagnostic = _density_diagnostic(simulated.samples, distribution)
                records.append(
                    {
                        "parameter_set": name,
                        "variant": variant_name,
                        "dt": dt,
                        "burn_years": config.stationary_burn_years,
                        "sample_years": config.stationary_sample_years,
                        "seed": seed,
                        "kappa2_hat": limit.kappa2_hat,
                        "theta_hat": limit.theta_hat,
                        "kappa1_hat": limit.kappa1_hat,
                        "floor_hits": simulated.floor_hits,
                        **diagnostic,
                    }
                )
    baseline_records = [row for row in records if row["variant"] == "baseline"]
    return {
        "claim": "the Q_LIMIT stationary law converges to the stated GIG density",
        "seed": seed,
        "records": records,
        "acceptance_pass": bool(
            all(row["relative_sup_density_error"] <= 0.15 for row in baseline_records)
            and all(row["floor_hits"] == 0 for row in records)
        ),
        "small_error_threshold": "relative sup-density error <= 0.15 (study convention)",
        "acceptance_criterion": (
            "The 0.15 threshold applies to the baseline daily and 1/1008 runs. The "
            "kappa2_hat-at-5% rows are a separate inverse-gamma degeneration stress test."
        ),
    }


def _bootstrap_mean_ci(
    pair_values: np.ndarray,
    *,
    seed: int,
    replications: int,
) -> tuple[float, float]:
    values = np.asarray(pair_values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("bootstrap input must contain at least two pair values")
    if values.size > 4_096:
        block_count = 1_024
        usable = (values.size // block_count) * block_count
        values = values[:usable].reshape(block_count, -1).mean(axis=1)
    generator = np.random.Generator(np.random.PCG64(seed))
    estimates = np.empty(replications)
    for index in range(replications):
        selected = generator.integers(0, values.size, size=values.size)
        estimates[index] = np.mean(values[selected])
    lower, upper = np.quantile(estimates, (0.025, 0.975))
    return float(lower), float(upper)


def _power_pair_values(log_spot: np.ndarray, power: float) -> tuple[np.ndarray, np.ndarray]:
    left, right = _pair_arrays(np.asarray(log_spot, dtype=np.float64))
    with np.errstate(over="ignore", invalid="ignore"):
        left_values = np.exp(power * left)
        right_values = np.exp(power * right)
    return 0.5 * (left_values + right_values), np.concatenate((left_values, right_values))


def _top_share(values: np.ndarray, fraction: float = 0.001) -> float:
    positive = np.asarray(values, dtype=np.float64)
    if positive.ndim != 1 or positive.size == 0 or not np.all(positive >= 0.0):
        return float("nan")
    if not np.all(np.isfinite(positive)):
        return 1.0
    count = max(1, int(math.ceil(fraction * positive.size)))
    total = float(np.sum(positive))
    if total <= 0.0:
        return float("nan")
    selected = np.partition(positive, positive.size - count)[-count:]
    return float(np.sum(selected) / total)


def _spot_moment_sufficient_boundary(params: TgarchParams, power: float) -> float:
    """Return the exponential-supersolution sufficient curve for E[S_T**p]."""
    return params.beta * power + params.vartheta * math.sqrt(power * (power - 1.0))


def run_e4(
    params: TgarchParams,
    config: StudyConfig,
) -> dict[str, Any]:
    """E4: probe heavy spot moments near the sufficient continuous-time curve.

    The brief does not specify which recursion to vary.  We use ``Q_LIMIT`` and
    hold ``d0`` and ``d1_hat`` fixed as ``kappa2_hat`` changes.  This avoids the
    pathwise ``b_k < 1/2`` failure that a positive affine exact-kernel loading
    would create when targeting small ``kappa2_hat``.  Consequently this is an
    Euler proxy for, not a proof of, the open uniform-integrability conjecture.
    """
    seed = BASE_SEED + 4
    maturity = 1.0 / 4.0
    discount = math.exp(-params.r * maturity)
    spot_payoff = EuropeanOptionPayoff(
        asset_id="spot",
        expiry=maturity,
        strike=0.0,
        option_type="C",
        unit="spot units",
    )
    base = derived_limit_params(params)
    records: list[dict[str, Any]] = []
    for dt in config.dt_grid:
        for kappa2_hat in KAPPA2_MOMENT_GRID:
            limit = _limit_with_kappa2(base, kappa2_hat)
            simulated = simulate_terminal(
                params=params,
                measure=Measure.Q_LIMIT,
                maturity=maturity,
                max_dt=dt,
                n_paths=config.e4_paths,
                seed=seed,
                limit_params=limit,
            )
            group_ids = _antithetic_group_ids(config.e4_paths)
            spot_values, spot_result = _value_european_leg(
                simulated=simulated,
                payoff=spot_payoff,
                discount=discount,
                independent_group_ids=group_ids,
                expected_estimator=PathEstimator.MONTE_CARLO,
            )
            _assert_pair_valuation(
                spot_result,
                _discounted_pair_values(spot_values, discount),
                "E4 discounted spot diagnostic",
            )
            for power in MOMENT_POWERS:
                pair_values, path_values = _power_pair_values(simulated.terminal_log_spot, power)
                if np.all(np.isfinite(pair_values)):
                    estimate, standard_error = _mean_and_se(pair_values)
                    ci_lower, ci_upper = _bootstrap_mean_ci(
                        pair_values,
                        seed=seed + 100_000 + int(100 * power) + int(1000 * kappa2_hat),
                        replications=config.bootstrap_replications,
                    )
                else:
                    estimate = standard_error = ci_lower = ci_upper = float("inf")
                boundary = _spot_moment_sufficient_boundary(params, power)
                records.append(
                    {
                        "parameter_set": "crypto",
                        "maturity": maturity,
                        "dt": dt,
                        "n_paths": config.e4_paths,
                        "seed": seed,
                        "kappa2_hat": kappa2_hat,
                        "power": power,
                        "moment": estimate,
                        "standard_error": standard_error,
                        "bootstrap_ci_lower": ci_lower,
                        "bootstrap_ci_upper": ci_upper,
                        "top_0_1pct_share": _top_share(path_values),
                        "sufficient_boundary": boundary,
                        "inside_sufficient_region": kappa2_hat >= boundary,
                        "floor_hits": simulated.floor_hits,
                    }
                )
    verdicts: list[dict[str, Any]] = []
    for power in MOMENT_POWERS:
        boundary = _spot_moment_sufficient_boundary(params, power)
        for kappa2_hat in KAPPA2_MOMENT_GRID:
            group = sorted(
                (
                    row
                    for row in records
                    if row["power"] == power and row["kappa2_hat"] == kappa2_hat
                ),
                key=lambda row: row["dt"],
                reverse=True,
            )
            coarse, fine = group[0], group[-1]
            combined_se = math.sqrt(coarse["standard_error"] ** 2 + fine["standard_error"] ** 2)
            if kappa2_hat >= boundary:
                empirical_pass = abs(fine["moment"] - coarse["moment"]) <= 3.0 * combined_se
                expected_pattern = "stability"
            else:
                empirical_pass = (
                    fine["moment"] > coarse["moment"] + combined_se
                    and fine["top_0_1pct_share"] >= coarse["top_0_1pct_share"]
                )
                expected_pattern = "growth and increasing tail concentration (conjectural)"
            verdicts.append(
                {
                    "power": power,
                    "kappa2_hat": kappa2_hat,
                    "sufficient_boundary": boundary,
                    "expected_pattern": expected_pattern,
                    "empirical_pattern_pass": bool(empirical_pass),
                }
            )
    return {
        "claim": "spot moments stabilise inside a sufficient supersolution region",
        "seed": seed,
        "maturity_assumption": maturity,
        "measure_choice": "Q_LIMIT with fixed d0 and d1_hat",
        "boundary_classification": (
            "sufficient all-horizons exponential-supersolution condition; the paper does not "
            "prove necessity or divergence outside it"
        ),
        "records": records,
        "verdicts": verdicts,
        "acceptance_pass": bool(all(row["empirical_pattern_pass"] for row in verdicts)),
    }


def _hill_curve(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    values = np.sort(np.asarray(samples, dtype=np.float64))
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size < 1_000:
        raise ValueError("Hill estimation requires at least 1,000 positive samples")
    maximum_k = min(5_000, values.size // 10)
    k_grid = np.unique(np.geomspace(50, maximum_k, 45).astype(int))
    tail_indices = np.empty(k_grid.size)
    for index, k in enumerate(k_grid):
        threshold = values[-k - 1]
        log_excess = np.log(values[-k:]) - math.log(threshold)
        tail_indices[index] = 1.0 / np.mean(log_excess)
    selected_k = int(round(math.sqrt(values.size)))
    selected_k = min(max(selected_k, 50), maximum_k)
    threshold = values[-selected_k - 1]
    selected_alpha = float(
        1.0 / np.mean(np.log(values[-selected_k:]) - math.log(threshold))
    )
    return k_grid, tail_indices, selected_k, selected_alpha


def _kesten_tail_index(params: TgarchParams, dt: float) -> float:
    """Solve the Kesten equation by adaptive, piecewise Gaussian quadrature.

    The absolute-news term is non-smooth at zero.  Fixed-order Hermite rules
    leave an error comparable with the order-``dt`` signal as the step shrinks,
    so the integral is split at zero and evaluated adaptively.
    """
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite number")
    root_dt = math.sqrt(dt)
    normalizer = 1.0 / math.sqrt(2.0 * math.pi)

    def moment(power: float) -> float:
        def integrand(z: float) -> float:
            w = (abs(z) - M1) / S1
            innovation = params.beta * z + params.eps * w
            coefficient = abs(1.0 - params.kappa1 * dt + root_dt * innovation)
            return coefficient**power * normalizer * math.exp(-0.5 * z * z)

        negative = quad(
            integrand,
            -math.inf,
            0.0,
            epsabs=2.0e-12,
            epsrel=2.0e-12,
            limit=500,
        )[0]
        positive = quad(
            integrand,
            0.0,
            math.inf,
            epsabs=2.0e-12,
            epsrel=2.0e-12,
            limit=500,
        )[0]
        return float(negative + positive)

    def equation(power: float) -> float:
        return moment(power) - 1.0

    lower = 1.0e-6
    if equation(lower) >= 0.0:
        return float("nan")
    upper = 2.0
    while upper <= 100.0 and equation(upper) <= 0.0:
        upper *= 2.0
    if upper > 100.0:
        return float("nan")
    return float(brentq(equation, lower, upper, xtol=1.0e-11))


def run_e5(params: TgarchParams, config: StudyConfig) -> dict[str, Any]:
    """E5: estimate fixed-step Kesten tails and expose the brief's false limit claim."""
    seed = BASE_SEED + 5
    linear_params = replace(params, kappa2=0.0)
    dts = (1.0 / 52.0, 1.0 / 252.0, 1.0 / 1008.0)
    if config.profile is StudyProfile.SMOKE:
        dts = (1.0 / 52.0, 1.0 / 252.0)
    continuous_alpha = 1.0 + 2.0 * linear_params.kappa1 / linear_params.vartheta**2
    records: list[dict[str, Any]] = []
    for dt in dts:
        simulated = simulate_stationary_sigma(
            params=linear_params,
            measure=Measure.P,
            dt=dt,
            burn_years=config.stationary_burn_years,
            sample_years=config.e5_sample_years,
            sample_interval=max(dt, 1.0 / 252.0),
            seed=seed,
        )
        k_grid, hill_alpha, selected_k, selected_alpha = _hill_curve(simulated.samples)
        kesten_alpha = _kesten_tail_index(linear_params, dt)
        records.append(
            {
                "parameter_set": "crypto_linear_P",
                "dt": dt,
                "burn_years": config.stationary_burn_years,
                "sample_years": config.e5_sample_years,
                "seed": seed,
                "n_samples": simulated.samples.size,
                "k_grid": k_grid,
                "hill_alpha": hill_alpha,
                "selected_k": selected_k,
                "selected_hill_alpha": selected_alpha,
                "kesten_alpha": kesten_alpha,
                "continuous_inverse_gamma_alpha": continuous_alpha,
                "floor_hits": simulated.floor_hits,
            }
        )
    convergence_error = [abs(row["kesten_alpha"] - continuous_alpha) for row in records]
    corrected_convergence = all(
        fine <= coarse + 1.0e-8
        for coarse, fine in zip(convergence_error[:-1], convergence_error[1:])
    )
    return {
        "claim": "fixed-step Kesten tails converge to the continuous inverse-gamma tail",
        "seed": seed,
        "records": records,
        "continuous_tail_index": continuous_alpha,
        "brief_acceptance_pass": False,
        "corrected_acceptance_pass": bool(corrected_convergence),
        "contradiction": (
            "The brief says the limit has all moments and the tail index should diverge. "
            "Remark 19 and the published stationary density instead imply an inverse-gamma "
            f"survival-tail exponent 1+2*kappa1/vartheta^2={continuous_alpha:.6f}."
        ),
    }


def _observation_indices(times: np.ndarray, dt_observation: float) -> np.ndarray:
    final_time = float(times[-1])
    target = np.arange(int(math.floor(final_time / dt_observation)) + 1) * dt_observation
    indices = np.rint(target / (times[1] - times[0])).astype(int)
    indices = np.clip(indices, 0, times.size - 1)
    return np.unique(indices)


def _forgetting_rate(times: np.ndarray, difference: np.ndarray) -> float:
    mask = (times > 0.0) & (times <= min(5.0, times[-1])) & (difference > 1.0e-8)
    if np.count_nonzero(mask) < 5:
        return float("nan")
    slope, _ = np.polyfit(times[mask], np.log(difference[mask]), 1)
    return float(-slope)


def run_e6(
    params_by_name: dict[str, TgarchParams],
    config: StudyConfig,
) -> dict[str, Any]:
    """E6: apply the exact discrete filter to paths of the two-shock diffusion limit."""
    seed = BASE_SEED + 6
    fine_dt = 1.0 / 16128.0
    observation_dts = (1.0 / 52.0, 1.0 / 252.0, 1.0 / 1008.0, 1.0 / 4032.0)
    if config.profile is StudyProfile.SMOKE:
        observation_dts = (1.0 / 52.0, 1.0 / 1008.0)
    records: list[dict[str, Any]] = []
    self_checks: list[dict[str, Any]] = []
    forgetting_series: list[dict[str, Any]] = []
    for name, params in params_by_name.items():
        self_path = simulate_discrete_path(
            params=params,
            measure=Measure.P,
            dt=1.0 / 252.0,
            years=min(config.e6_years, 2.0),
            seed=seed,
        )
        self_filtered = filter_discrete_returns(
            log_prices=self_path.log_prices,
            observation_times=self_path.times,
            params=params,
            sigma0=params.sigma0,
        )
        self_checks.append(
            {
                "parameter_set": name,
                "max_abs_filter_error_own_model": float(
                    np.max(np.abs(self_filtered - self_path.sigmas))
                ),
                "pass": bool(np.max(np.abs(self_filtered - self_path.sigmas)) <= 1.0e-10),
            }
        )
        source = simulate_two_shock_limit_path(
            params=params,
            dt=fine_dt,
            years=config.e6_years,
            seed=seed,
        )
        for dt_observation in observation_dts:
            indices = _observation_indices(source.times, dt_observation)
            times = source.times[indices]
            log_prices = source.log_prices[indices]
            true_sigma = source.sigmas[indices]
            filtered = filter_discrete_returns(
                log_prices=log_prices,
                observation_times=times,
                params=params,
                sigma0=params.sigma0,
            )
            wrong = filter_discrete_returns(
                log_prices=log_prices,
                observation_times=times,
                params=params,
                sigma0=1.5 * params.sigma0,
            )
            burn_mask = times >= min(1.0, 0.1 * times[-1])
            rmse = float(np.sqrt(np.mean(np.square(filtered[burn_mask] - true_sigma[burn_mask]))))
            wrong_rmse = float(
                np.sqrt(np.mean(np.square(wrong[burn_mask] - true_sigma[burn_mask])))
            )
            difference = np.abs(wrong - filtered)
            forgetting_rate = _forgetting_rate(times, difference)
            linearised_rate = params.kappa1 + params.kappa2 * params.theta
            forgetting_rate_ratio = forgetting_rate / linearised_rate
            forgetting_rate_roughly_mean_reversion = bool(
                math.isfinite(forgetting_rate_ratio)
                and 0.5 <= forgetting_rate_ratio <= 2.0
            )
            effective_dts = np.diff(times)
            records.append(
                {
                    "parameter_set": name,
                    "source_dt": fine_dt,
                    "dt_observation_nominal": dt_observation,
                    "dt_observation_mean": float(np.mean(effective_dts)),
                    "dt_observation_min": float(np.min(effective_dts)),
                    "dt_observation_max": float(np.max(effective_dts)),
                    "years": config.e6_years,
                    "seed": seed,
                    "rmse_correct_start": rmse,
                    "rmse_wrong_start": wrong_rmse,
                    "forgetting_rate": forgetting_rate,
                    "linearised_mean_reversion_rate": linearised_rate,
                    "forgetting_rate_ratio": forgetting_rate_ratio,
                    "forgetting_rate_roughly_mean_reversion": (
                        forgetting_rate_roughly_mean_reversion
                    ),
                    "source_floor_hits": source.floor_hits,
                }
            )
            stride = max(1, times.size // 600)
            forgetting_series.append(
                {
                    "parameter_set": name,
                    "dt_observation_nominal": dt_observation,
                    "times": times[::stride],
                    "wrong_minus_correct_abs": difference[::stride],
                }
            )
    scale_verdicts: list[dict[str, Any]] = []
    for name in params_by_name:
        group = sorted(
            (row for row in records if row["parameter_set"] == name),
            key=lambda row: row["dt_observation_nominal"],
            reverse=True,
        )
        decreasing = all(
            fine["rmse_correct_start"] <= coarse["rmse_correct_start"]
            for coarse, fine in zip(group[:-1], group[1:])
        )
        forgetting_pass = all(
            row["forgetting_rate_roughly_mean_reversion"] for row in group
        )
        scale_verdicts.append(
            {
                "parameter_set": name,
                "rmse_decreases_with_dt": bool(decreasing),
                "forgetting_rate_roughly_mean_reversion": bool(forgetting_pass),
                "minimum_forgetting_rate_ratio": min(
                    row["forgetting_rate_ratio"] for row in group
                ),
                "maximum_forgetting_rate_ratio": max(
                    row["forgetting_rate_ratio"] for row in group
                ),
                "coarse_rmse": group[0]["rmse_correct_start"],
                "fine_rmse": group[-1]["rmse_correct_start"],
            }
        )
    return {
        "claim": "the exact one-shock filter consistently recovers a two-shock limit path",
        "seed": seed,
        "records": records,
        "self_model_checks": self_checks,
        "forgetting_series": forgetting_series,
        "scale_verdicts": scale_verdicts,
        "acceptance_pass": bool(
            all(row["pass"] for row in self_checks)
            and all(row["rmse_decreases_with_dt"] for row in scale_verdicts)
            and all(
                row["forgetting_rate_roughly_mean_reversion"]
                for row in scale_verdicts
            )
        ),
        "acceptance_criterion": (
            "RMSE must decrease at finer observation steps, and the fitted wrong-start "
            "decay rate must be within a factor of two of the linearised mean-reversion "
            "rate (study convention)."
        ),
        "interpretation_caveat": (
            "Exactness holds for the one-shock discrete recursion. A path generated with an "
            "independent second Brownian shock is deliberately not the filter's own model."
        ),
    }


@njit(cache=True)
def _qmle_observations(
    transformed: np.ndarray,
    log_returns: np.ndarray,
    dt: float,
    r: float,
) -> np.ndarray:
    kappa1 = math.exp(transformed[0])
    kappa2 = math.exp(transformed[1])
    theta = math.exp(transformed[2])
    beta = transformed[3]
    eps = math.exp(transformed[4])
    gamma1 = transformed[5]
    sigma = theta
    root_dt = math.sqrt(dt)
    observations = np.empty(log_returns.size)
    constant = math.log(2.0 * math.pi * dt)
    for index in range(log_returns.size):
        if sigma <= 1.0e-8 or not math.isfinite(sigma):
            observations[:] = -1.0e12
            return observations
        conditional_mean = (r + gamma1 * sigma * sigma - 0.5 * sigma * sigma) * dt
        z = (log_returns[index] - conditional_mean) / (sigma * root_dt)
        observations[index] = -0.5 * (constant + 2.0 * math.log(sigma) + z * z)
        w = (abs(z) - M1) / S1
        drift = (kappa1 + kappa2 * sigma) * (theta - sigma)
        sigma = sigma + drift * dt + sigma * root_dt * (beta * z + eps * w)
    return observations


def _to_transformed(natural: np.ndarray) -> np.ndarray:
    if natural.shape != (6,):
        raise ValueError("natural parameter vector must have length six")
    if np.any(natural[[0, 1, 2, 4]] <= 0.0):
        raise ValueError("kappa1, kappa2, theta, and eps must be positive")
    return np.array(
        (
            math.log(natural[0]),
            math.log(natural[1]),
            math.log(natural[2]),
            natural[3],
            math.log(natural[4]),
            natural[5],
        ),
        dtype=np.float64,
    )


def _to_natural(transformed: np.ndarray) -> np.ndarray:
    return np.array(
        (
            math.exp(transformed[0]),
            math.exp(transformed[1]),
            math.exp(transformed[2]),
            transformed[3],
            math.exp(transformed[4]),
            transformed[5],
        ),
        dtype=np.float64,
    )


def _fit_qmle(
    log_returns: np.ndarray,
    *,
    dt: float,
    r: float,
    start_natural: np.ndarray,
) -> tuple[np.ndarray, float, bool, float]:
    transformed_start = _to_transformed(start_natural)
    bounds = (
        (math.log(0.1), math.log(20.0)),
        (math.log(0.01), math.log(20.0)),
        (math.log(0.03), math.log(2.5)),
        (-3.0, 3.0),
        (math.log(0.05), math.log(4.0)),
        (-2.0, 2.0),
    )

    def objective(vector: np.ndarray) -> float:
        values = _qmle_observations(vector, log_returns, dt, r)
        result = -float(np.mean(values))
        return result if np.isfinite(result) else 1.0e12

    fitted = minimize(
        objective,
        transformed_start,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 350, "ftol": 1.0e-10, "gtol": 1.0e-6},
    )
    vector = np.asarray(fitted.x, dtype=np.float64)
    observations = _qmle_observations(vector, log_returns, dt, r)
    scores = np.empty((log_returns.size, vector.size))
    for column in range(vector.size):
        step = 1.0e-4 * (1.0 + abs(vector[column]))
        plus = vector.copy()
        minus = vector.copy()
        plus[column] += step
        minus[column] -= step
        scores[:, column] = (
            _qmle_observations(plus, log_returns, dt, r)
            - _qmle_observations(minus, log_returns, dt, r)
        ) / (2.0 * step)
    information = scores.T @ scores
    covariance = np.linalg.pinv(information, rcond=1.0e-10)
    gamma1_se = float(math.sqrt(max(covariance[5, 5], 0.0)))
    return _to_natural(vector), gamma1_se, bool(fitted.success), float(np.sum(observations))


def run_e7(
    params_by_name: dict[str, TgarchParams],
    config: StudyConfig,
) -> dict[str, Any]:
    """E7: run the exact prediction-error QMLE identification experiment."""
    seed = BASE_SEED + 7
    dt = 1.0 / 252.0
    sample_years = (5, 10, 20, 40)
    if config.profile is StudyProfile.SMOKE:
        sample_years = (5, 10)
    parameter_names = ("kappa1", "kappa2", "theta", "beta", "eps", "gamma1")
    estimates: list[dict[str, Any]] = []
    for name in config.e7_parameter_sets:
        params = params_by_name[name]
        truth = np.array(
            (params.kappa1, params.kappa2, params.theta, params.beta, params.eps, params.gamma1)
        )
        start = truth * np.array((0.85, 1.15, 1.05, 0.8, 1.1, 0.5))
        if abs(start[3]) < 0.05:
            start[3] = -0.1
        for replication in range(config.e7_replications):
            replication_seed = seed + replication
            path = simulate_discrete_path(
                params=params,
                measure=Measure.P,
                dt=dt,
                years=float(max(sample_years)),
                seed=replication_seed,
            )
            all_returns = np.diff(path.log_prices)
            for years in sample_years:
                count = int(round(years / dt))
                fitted, gamma1_se, converged, log_likelihood = _fit_qmle(
                    all_returns[:count],
                    dt=dt,
                    r=params.r,
                    start_natural=start,
                )
                estimates.append(
                    {
                        "parameter_set": name,
                        "years": years,
                        "replication": replication,
                        "seed": replication_seed,
                        "estimate": fitted,
                        "gamma1_se": gamma1_se,
                        "gamma1_tstat": (
                            fitted[5] / gamma1_se if gamma1_se > 0.0 else float("nan")
                        ),
                        "converged": converged,
                        "log_likelihood": log_likelihood,
                        "floor_hits": path.floor_hits,
                    }
                )
    summaries: list[dict[str, Any]] = []
    identification: list[dict[str, Any]] = []
    for name in config.e7_parameter_sets:
        params = params_by_name[name]
        truth = np.array(
            (params.kappa1, params.kappa2, params.theta, params.beta, params.eps, params.gamma1)
        )
        for years in sample_years:
            group = [
                row
                for row in estimates
                if row["parameter_set"] == name and row["years"] == years
            ]
            matrix = np.vstack([row["estimate"] for row in group])
            for index, parameter in enumerate(parameter_names):
                errors = matrix[:, index] - truth[index]
                summaries.append(
                    {
                        "parameter_set": name,
                        "years": years,
                        "parameter": parameter,
                        "truth": float(truth[index]),
                        "mean_estimate": float(np.mean(matrix[:, index])),
                        "bias": float(np.mean(errors)),
                        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                    }
                )
            tstats = np.array([abs(row["gamma1_tstat"]) for row in group])
            kappa2_lower_hits = int(np.count_nonzero(np.isclose(matrix[:, 1], 0.01)))
            kappa2_upper_hits = int(np.count_nonzero(np.isclose(matrix[:, 1], 20.0)))
            identification.append(
                {
                    "parameter_set": name,
                    "years": years,
                    "median_abs_gamma1_tstat": float(np.nanmedian(tstats)),
                    "gamma1_identified_at_2": bool(np.nanmedian(tstats) >= 2.0),
                    "kappa1_kappa2_estimate_correlation": float(
                        np.corrcoef(matrix[:, 0], matrix[:, 1])[0, 1]
                        if matrix.shape[0] >= 3
                        else np.nan
                    ),
                    "convergence_fraction": float(np.mean([row["converged"] for row in group])),
                    "kappa2_lower_bound_hits": kappa2_lower_hits,
                    "kappa2_upper_bound_hits": kappa2_upper_hits,
                    "kappa2_any_bound_fraction": float(
                        (kappa2_lower_hits + kappa2_upper_hits) / matrix.shape[0]
                    ),
                }
            )
    threshold_years: dict[str, int | None] = {}
    for name in config.e7_parameter_sets:
        candidates = [
            row["years"]
            for row in identification
            if row["parameter_set"] == name and row["gamma1_identified_at_2"]
        ]
        threshold_years[name] = min(candidates) if candidates else None
    unreached = [name for name, years in threshold_years.items() if years is None]
    correlation_values = [
        row["kappa1_kappa2_estimate_correlation"] for row in identification
    ]
    five_year_rows = [row for row in identification if row["years"] == 5]
    findings = [
        (
            "Median absolute gamma1 t-statistic did not reach 2 through 40 years for "
            + ", ".join(unreached)
            + "."
            if unreached
            else "Median absolute gamma1 t-statistic reached 2 within the simulated horizon."
        ),
        (
            "kappa1/kappa2 estimates remain poorly separated: correlations range from "
            f"{min(correlation_values):.3f} to {max(correlation_values):.3f}; five-year "
            "kappa2 bound-hit fractions are "
            + ", ".join(
                f"{row['parameter_set']}={row['kappa2_any_bound_fraction']:.1%}"
                for row in five_year_rows
            )
            + "."
        ),
    ]
    return {
        "claim": "QMLE sample-size experiment measures structural identification",
        "seed_rule": f"{seed} + replication index",
        "dt": dt,
        "replications": config.e7_replications,
        "estimates": estimates,
        "summaries": summaries,
        "identification": identification,
        "gamma1_median_abs_tstat_2_years": threshold_years,
        "acceptance_criterion": "none: negative identification results are findings",
        "findings": findings,
    }


def _parameter_header(params: TgarchParams) -> dict[str, Any]:
    limit = derived_limit_params(params)
    return {
        "physical": asdict(params),
        "derived": asdict(limit),
        "lambda0_bar": params.gamma1,
        "lambda1_bar": -(M1 / S1) * params.eta1,
        "vartheta": params.vartheta,
    }


def run_all_experiments(
    output_dir: Path,
    profile: StudyProfile = StudyProfile.FULL,
) -> StudyResults:
    """Run mandatory checks followed by E1--E7 in the brief's priority order."""
    output = Path(output_dir)
    if not output.exists() or not output.is_dir():
        raise ValueError("output_dir must be an existing directory")
    config = make_study_config(profile)
    params_by_name = parameter_sets()
    print(f"[TGARCH] profile={profile.value}: mandatory simulator checks", flush=True)
    checks = run_unit_checks(
        parameter_sets=params_by_name,
        dt_grid=config.dt_grid,
        maturity=max(config.maturities),
        seed=BASE_SEED,
        n_paths=config.unit_paths,
        moment_draws=config.moment_check_draws,
    )
    if not checks.get("all_passed", False):
        raise RuntimeError(
            "mandatory simulator checks failed; experiments were not started: "
            + json.dumps(_as_jsonable(checks), indent=2)[:4_000]
        )

    def timed_run(identifier: str, function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        print(f"[TGARCH] {identifier}: start", flush=True)
        started = time.perf_counter()
        result = function()
        result["runtime_seconds"] = time.perf_counter() - started
        print(
            f"[TGARCH] {identifier}: done in {result['runtime_seconds']:.2f}s",
            flush=True,
        )
        return result

    experiments: dict[str, Any] = {}
    experiments["E1"] = timed_run("E1", lambda: run_e1(params_by_name, config))
    experiments["E2"] = timed_run("E2", lambda: run_e2(params_by_name, config))
    experiments["E3"] = timed_run("E3", lambda: run_e3(params_by_name, config))
    experiments["E4"] = timed_run("E4", lambda: run_e4(params_by_name["crypto"], config))
    experiments["E5"] = timed_run("E5", lambda: run_e5(params_by_name["crypto"], config))
    experiments["E6"] = timed_run("E6", lambda: run_e6(params_by_name, config))
    experiments["E7"] = timed_run("E7", lambda: run_e7(params_by_name, config))
    warnings: list[str] = []
    for row in experiments["E2"]["records"]:
        if row["ess_warning"]:
            warnings.append(
                f"E2 ESS/n below 0.2 for {row['parameter_set']} at dt={row['dt']:.8g}"
            )
    if any(
        row.get("floor_hits", 0) != 0
        for experiment in experiments.values()
        for row in experiment.get("records", [])
        if isinstance(row.get("floor_hits", 0), (int, np.integer))
    ):
        warnings.append("At least one reported simulation recorded volatility-floor hits.")
    warnings.extend(
        (
            "Q_EXACT antithetic base normals do not produce equal |z| within a pair when the "
            "conditional Gaussian has a nonzero, state-dependent mean.",
            "E4 uses Q_LIMIT because the brief does not define how to vary kappa2_hat in the "
            "exact kernel without violating b_k < 1/2 on unbounded volatility states.",
            "E5's stated all-moments-finite limit is contradicted by the inverse-gamma law.",
            "E6 weekly observation times require alternating fine-grid spacings because "
            "16128/52 is not an integer.",
        )
    )
    return StudyResults(
        profile=profile.value,
        config=_as_jsonable(config),
        parameters={name: _parameter_header(params) for name, params in params_by_name.items()},
        checks=_as_jsonable(checks),
        experiments=experiments,
        provenance=collect_provenance(output, profile),
        warnings=warnings,
    )


__all__ = [
    "StudyConfig",
    "StudyProfile",
    "StudyResults",
    "collect_provenance",
    "make_study_config",
    "parameter_sets",
    "run_all_experiments",
]
