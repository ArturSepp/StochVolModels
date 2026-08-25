"""Package-owned numerical pipeline for the regime-switching LogSV chapter.

The production path in this module uses :mod:`stochvolmodels` for the
equilibrium, induced-Q state, Fourier transform, implied volatilities, and
risk-neutral Monte Carlo.  The frozen chapter monolith is loaded lazily, after
an exact hash check, only for the physical-measure Feynman--Kac validation that
does not yet have a package API.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence

import numpy as np

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.regime_logsv import (
    CrraRiskPremia,
    EquilibriumClosure,
    EquilibriumSolution,
    Regime,
    RegimeLogSvDynamics,
    RegimeRiskPremiaScales,
    RegimeSwitchLogSvParams,
    RegimeTransition,
    evaluate_risk_neutral_state,
    solve_regime_switch_equilibrium,
)
from stochvolmodels.models.regime_logsv_simulation import (
    simulate_regime_switch_logsv_terminal,
)
from stochvolmodels.pricers.logsv.affine_expansion import ExpansionOrder
from stochvolmodels.pricers.regime_switch_logsv_pricer import (
    RegimeSwitchLogSVPricer,
    StateConditionalOptionChain,
    compute_regime_switch_log_mgf_grid,
)
from stochvolmodels.utils.config import VariableType
from stochvolmodels.utils.mc_payoffs import compute_mc_vars_payoff

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ACCEPTANCE_MANIFEST = SCRIPT_DIR / "acceptance_manifest.json"
SOURCE_PROVENANCE = SCRIPT_DIR / "source_provenance.json"
FROZEN_ORACLE = SCRIPT_DIR / "regime_switch_logsv.py"
EXPECTED_FROZEN_ORACLE_SHA256 = "f197e10149cab121d1cdecec2a85b5311143d69dbde00311d4d8b90f1d9b9e5a"
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "outputs" / "volatility_book" / "ch_lognormal_sv_risk_premia"
)
PAYLOAD_FILENAME = "numerical_payload.json"
ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"
FORBID_RECOMPUTE_ENV = "STOCHVOLMODELS_REGIME_SV_FORBID_RECOMPUTE"
PAYLOAD_SCHEMA_VERSION = 1
FOURIER_POINTS = 1_601
FOURIER_REFINEMENT_POINTS = 3_201

SMILE_LOG_MONEYNESS = np.linspace(-0.30, 0.20, 31)
SMILE_MATURITIES = np.array([1.0 / 12.0, 0.25, 0.5, 1.0])
VALIDATION_LOG_MONEYNESS = np.linspace(-0.30, 0.20, 16)
SIGMA_GRID = np.linspace(0.08, 0.36, 180)
UTILITY_POWER_GRID = np.linspace(-0.9, 0.2, 180)
EQUILIBRIUM_HORIZON_GRID = np.linspace(0.0, 3.0, 151)
FK_HORIZONS = np.array([1.0, 3.0])
EXPECTED_RECORD_NAMES = frozenset(
    {
        "closure.implied_volatility",
        "closure.implied_volatility_difference_bp",
        "closure.table",
        "premia.jump_arithmetic_mean",
        "premia.log_timing_ratio",
        "premia.physical_drift",
        "premia.physical_jump_arithmetic_mean",
        "premia.physical_transition_intensity",
        "premia.risk_neutral_drift",
        "premia.transition_intensity",
        "premia.volatility_loading",
        "smiles.channel_implied_volatility",
        "smiles.channel_log_mgf",
        "smiles.channel_prices",
        "smiles.initial_regime_implied_volatility",
        "smiles.risk_aversion_implied_volatility",
        "smiles.term_structure_implied_volatility",
        "validation.analytic_implied_volatility",
        "validation.analytic_prices",
        "validation.equilibrium_coefficients",
        "validation.forward_martingale",
        "validation.fourier_refinement_prices",
        "validation.mc_prices",
        "validation.mc_standard_errors",
        "validation.physical_feynman_kac",
        "validation.table",
    }
)


class ChapterProfile(str, Enum):
    """Supported numerical workloads for the chapter pipeline."""

    SMOKE = "smoke"
    CANONICAL = "canonical"


@dataclass(frozen=True, slots=True)
class ChapterConfig:
    """Numerical workload attached to a chapter profile."""

    profile: ChapterProfile
    monte_carlo_paths: int
    steps_per_year: int


def as_profile(value: ChapterProfile | str) -> ChapterProfile:
    """Return a validated chapter profile."""

    if isinstance(value, ChapterProfile):
        return value
    try:
        return ChapterProfile(str(value))
    except ValueError as error:
        choices = ", ".join(item.value for item in ChapterProfile)
        raise ValueError(f"profile must be one of: {choices}") from error


def profile_config(profile: ChapterProfile | str) -> ChapterConfig:
    """Return the deterministic workload for ``profile``."""

    selected = as_profile(profile)
    if selected is ChapterProfile.SMOKE:
        return ChapterConfig(selected, monte_carlo_paths=4_000, steps_per_year=360)
    return ChapterConfig(selected, monte_carlo_paths=120_000, steps_per_year=1_440)


def default_output_directory(profile: ChapterProfile | str) -> Path:
    """Return the ignored profile-specific artifact directory."""

    return DEFAULT_OUTPUT_ROOT / as_profile(profile).value


def validate_output_directory(path: Path | str) -> Path:
    """Resolve output and confine repo-local artifacts to the ignored root."""

    output = Path(path).expanduser().resolve()
    repository = REPOSITORY_ROOT.resolve()
    try:
        repository.relative_to(output)
    except ValueError:
        pass
    else:
        raise ValueError("output directory must not equal or contain the repository root")
    try:
        output.relative_to(repository)
    except ValueError:
        return output
    try:
        output.relative_to(DEFAULT_OUTPUT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            f"repository-local regime-SV artifacts must be written below {DEFAULT_OUTPUT_ROOT}"
        ) from error
    return output


def equity_params(
    *,
    utility_power: float = -0.5,
    initial_regime: Regime = Regime.GROWTH,
    agent_horizon: float = 3.0,
    closure: EquilibriumClosure = EquilibriumClosure.LOG_LINEAR,
) -> RegimeSwitchLogSvParams:
    """Build the chapter's transparent equity illustration with package types.

    The growth/stress volatility levels, transition clocks, and arithmetic jump
    sizes are the original chapter inputs.  The combined parameter set remains
    an illustration rather than an empirical calibration.
    """

    common = dict(
        kappa1=2.6949,
        kappa2=10.0107,
        beta=-1.5082,
        volvol=0.8503,
    )
    return RegimeSwitchLogSvParams(
        sigma0=0.15,
        regimes=(
            RegimeLogSvDynamics(theta=0.15, **common),
            RegimeLogSvDynamics(theta=0.225, **common),
        ),
        transitions=(
            RegimeTransition(intensity=0.1, mean_log_jump=-(0.25 / 0.75)),
            RegimeTransition(intensity=1.0, mean_log_jump=0.15 / 1.15),
        ),
        risk_premia=CrraRiskPremia(
            utility_power=utility_power,
            agent_horizon=agent_horizon,
            closure=closure,
        ),
        initial_regime=initial_regime,
    )


class _AnalyticsCache:
    """Reuse every equilibrium and coupled state-conditional chain exactly once."""

    def __init__(self) -> None:
        self._pricer = RegimeSwitchLogSVPricer()
        self._equilibria: dict[RegimeSwitchLogSvParams, EquilibriumSolution] = {}
        self._chains: dict[
            tuple[
                RegimeSwitchLogSvParams,
                RegimeRiskPremiaScales,
                tuple[float, ...],
                tuple[float, ...],
                int,
            ],
            StateConditionalOptionChain,
        ] = {}

    def equilibrium(self, params: RegimeSwitchLogSvParams) -> EquilibriumSolution:
        if params not in self._equilibria:
            self._equilibria[params] = solve_regime_switch_equilibrium(params)
        return self._equilibria[params]

    def chain(
        self,
        params: RegimeSwitchLogSvParams,
        maturities: Sequence[float],
        log_moneyness: Sequence[float],
        *,
        scales: RegimeRiskPremiaScales = RegimeRiskPremiaScales(),
        max_phi: int = FOURIER_POINTS,
    ) -> StateConditionalOptionChain:
        maturity_key = tuple(float(value) for value in maturities)
        moneyness_key = tuple(float(value) for value in log_moneyness)
        key = (params, scales, maturity_key, moneyness_key, int(max_phi))
        if key in self._chains:
            return self._chains[key]

        ttms = np.asarray(maturity_key, dtype=float)
        log_grid = np.asarray(moneyness_key, dtype=float)
        strikes = np.exp(log_grid)
        optiontypes = np.where(strikes < 1.0, "P", "C")
        chain = OptionChain(
            ttms=ttms,
            forwards=np.ones(ttms.size),
            discfactors=np.ones(ttms.size),
            strikes_ttms=tuple(strikes.copy() for _ in ttms),
            optiontypes_ttms=tuple(optiontypes.copy() for _ in ttms),
            ids=np.asarray([f"{ttm:.12g}" for ttm in ttms]),
        )
        result = self._pricer.compute_state_conditional_prices_with_vols(
            chain,
            params,
            equilibrium=self.equilibrium(params),
            scales=scales,
            expansion_order=ExpansionOrder.SECOND,
            max_phi=max_phi,
        )
        self._chains[key] = result
        return result


def _conditional_array(
    conditional: StateConditionalOptionChain,
    attribute: str,
) -> np.ndarray:
    output = []
    for regime in Regime:
        prices, ivols = conditional.for_regime(regime)
        values = prices if attribute == "prices" else ivols
        output.append(np.stack([np.asarray(value, dtype=float) for value in values]))
    return np.stack(output)


def _json_axis_values(values: Iterable[Any]) -> list[Any]:
    output: list[Any] = []
    for value in values:
        if isinstance(value, (np.integer, int)) and not isinstance(value, (bool, np.bool_)):
            output.append(int(value))
        elif isinstance(value, (np.floating, float)):
            number = float(value)
            if not np.isfinite(number):
                raise ValueError("payload axes must be finite")
            output.append(number)
        elif isinstance(value, (str, bool)) or value is None:
            output.append(value)
        else:
            raise TypeError(f"unsupported payload axis value {value!r}")
    return output


def array_record(
    values: np.ndarray | Sequence[float],
    axes: Sequence[tuple[str, Iterable[Any]]],
    **metadata: Any,
) -> dict[str, Any]:
    """Encode one finite numerical array with ordered axes and row-major values."""

    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError("payload record values must be real numbers")
    if not np.all(np.isfinite(array)):
        raise ValueError("payload record values must be finite")
    if len(axes) != array.ndim:
        raise ValueError("payload record must name every array dimension")

    encoded_axes = []
    names: set[str] = set()
    for dimension, (name, coordinates) in enumerate(axes):
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("payload axis names must be unique non-empty strings")
        names.add(name)
        encoded = _json_axis_values(coordinates)
        if len(encoded) != array.shape[dimension]:
            raise ValueError(f"axis {name!r} does not match dimension {dimension}")
        encoded_axes.append({"name": name, "values": encoded})

    record = {
        "axes": encoded_axes,
        "shape": [int(value) for value in array.shape],
        "values": array.ravel(order="C").tolist(),
    }
    record.update(metadata)
    return record


def record_array(record: dict[str, Any]) -> np.ndarray:
    """Decode the row-major values of a validated payload record."""

    return np.asarray(record["values"], dtype=float).reshape(record["shape"])


def record_axis(record: dict[str, Any], name: str) -> list[Any]:
    """Return one named coordinate vector from a payload record."""

    for axis in record["axes"]:
        if axis["name"] == name:
            return list(axis["values"])
    raise KeyError(f"record has no axis named {name!r}")


def _validate_finite_json(value: Any, location: str = "payload") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{location} keys must be strings")
            _validate_finite_json(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{location}[{index}]")
    elif isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"{location} contains a non-finite number")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise TypeError(f"{location} contains unsupported value {value!r}")


def _validate_record(name: str, record: Any) -> None:
    if not isinstance(record, dict):
        raise TypeError(f"record {name!r} must be an object")
    for required in ("axes", "shape", "values"):
        if required not in record:
            raise ValueError(f"record {name!r} is missing {required!r}")
    shape = record["shape"]
    axes = record["axes"]
    values = record["values"]
    if not isinstance(shape, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in shape
    ):
        raise ValueError(f"record {name!r} has an invalid shape")
    if not isinstance(axes, list) or len(axes) != len(shape):
        raise ValueError(f"record {name!r} must provide one axis per dimension")
    names: set[str] = set()
    for dimension, axis in enumerate(axes):
        if not isinstance(axis, dict) or set(axis) != {"name", "values"}:
            raise ValueError(f"record {name!r} has an invalid axis descriptor")
        axis_name = axis["name"]
        if not isinstance(axis_name, str) or not axis_name or axis_name in names:
            raise ValueError(f"record {name!r} has duplicate or invalid axes")
        names.add(axis_name)
        if not isinstance(axis["values"], list) or len(axis["values"]) != shape[dimension]:
            raise ValueError(f"record {name!r} axis {axis_name!r} has the wrong length")
    count = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"record {name!r} values do not agree with its shape")
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"record {name!r} value {index} must be a finite JSON number")


def validate_numerical_payload(payload: Any) -> dict[str, Any]:
    """Validate the portable numerical-payload contract and return it."""

    if not isinstance(payload, dict):
        raise TypeError("numerical payload must be a JSON object")
    if payload.get("schema_version") != PAYLOAD_SCHEMA_VERSION:
        raise ValueError(f"unsupported numerical payload schema {payload.get('schema_version')!r}")
    as_profile(payload.get("profile"))
    for required in ("configuration", "provenance"):
        value = payload.get(required)
        if not isinstance(value, dict) or not value:
            raise ValueError(f"numerical payload {required} must be a non-empty object")
    records = payload.get("records")
    if not isinstance(records, dict) or not records:
        raise ValueError("numerical payload records must be a non-empty object")
    missing = EXPECTED_RECORD_NAMES.difference(records)
    if missing:
        raise ValueError(f"numerical payload is missing expected records: {sorted(missing)}")
    for name, record in records.items():
        if not isinstance(name, str) or not name:
            raise ValueError("numerical payload record names must be non-empty strings")
        _validate_record(name, record)
    _validate_finite_json(payload)
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def write_numerical_payload(payload: dict[str, Any], path: Path | str) -> Path:
    """Write a validated, deterministic JSON numerical payload."""

    validate_numerical_payload(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target.write_text(text, encoding="utf-8", newline="\n")
    return target


def load_numerical_payload(path: Path | str) -> dict[str, Any]:
    """Load and validate a numerical payload without invoking any analytics."""

    source = Path(path)

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} is forbidden")

    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream, parse_constant=reject_constant)
    return validate_numerical_payload(payload)


_FROZEN_ORACLE_MODULE: ModuleType | None = None


def _check_frozen_oracle_hash() -> str:
    if not FROZEN_ORACLE.is_file():
        raise FileNotFoundError(f"frozen physical Feynman--Kac oracle is missing: {FROZEN_ORACLE}")
    observed = _sha256(FROZEN_ORACLE)
    if observed != EXPECTED_FROZEN_ORACLE_SHA256:
        raise RuntimeError(
            "frozen physical Feynman--Kac oracle hash mismatch: "
            f"expected {EXPECTED_FROZEN_ORACLE_SHA256}, observed {observed}"
        )
    return observed


def _load_frozen_physical_fk_oracle() -> ModuleType:
    """Load the hash-pinned monolith solely for physical Feynman--Kac points."""

    global _FROZEN_ORACLE_MODULE
    _check_frozen_oracle_hash()
    if _FROZEN_ORACLE_MODULE is not None:
        return _FROZEN_ORACLE_MODULE
    module_name = "_stochvolmodels_frozen_regime_sv_physical_fk_oracle"
    specification = importlib.util.spec_from_file_location(module_name, FROZEN_ORACLE)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load frozen oracle from {FROZEN_ORACLE}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    _FROZEN_ORACLE_MODULE = module
    return module


def _physical_fk_values(config: ChapterConfig) -> np.ndarray:
    oracle = _load_frozen_physical_fk_oracle()
    output = np.empty((2, FK_HORIZONS.size, 2), dtype=float)
    bases = {Regime.GROWTH: 1_301, Regime.STRESS: 1_303}
    for regime in Regime:
        frozen_params = oracle.RegimeSwitchLogSvParams.equity_baseline(
            gamma=-0.5,
            initial_regime=int(regime),
            agent_horizon=3.0,
        )
        sigma = frozen_params.regimes[int(regime)].theta
        for horizon_index, horizon in enumerate(FK_HORIZONS):
            value, error = oracle.simulate_equilibrium_feynman_kac(
                frozen_params,
                float(horizon),
                sigma,
                int(regime),
                n_paths=config.monte_carlo_paths,
                steps_per_year=config.steps_per_year,
                seed=bases[regime] + int(10 * horizon),
            )
            output[int(regime), horizon_index] = (value, error)
    return output


def _build_validation_table(
    analytic_prices: np.ndarray,
    mc_prices: np.ndarray,
    mc_errors: np.ndarray,
    martingale: np.ndarray,
    equilibrium_coefficients: np.ndarray,
    physical_fk: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, str]]]:
    rows: list[list[float]] = []
    metadata: list[dict[str, str]] = []
    closure_short = ("quadratic Q", "full cubic Q")
    regime_labels = ("Growth", "Stress")
    atm = int(np.argmin(np.abs(np.exp(VALIDATION_LOG_MONEYNESS) - 1.0)))
    for closure_index, short_label in enumerate(closure_short):
        for regime in Regime:
            rows.append([1.0, *martingale[closure_index, int(regime)]])
            metadata.append(
                {
                    "check": rf"$E^{{\mathbb{{Q}}}}[S_T/F_0]$, {short_label}",
                    "regime": regime_labels[int(regime)],
                }
            )
            rows.append(
                [
                    analytic_prices[closure_index, int(regime), atm],
                    mc_prices[closure_index, int(regime), atm],
                    mc_errors[closure_index, int(regime), atm],
                ]
            )
            metadata.append(
                {
                    "check": f"3m near-ATM, {short_label}",
                    "regime": regime_labels[int(regime)],
                }
            )

    horizon_index = int(np.flatnonzero(np.isclose(FK_HORIZONS, 3.0))[0])
    curve_index = int(np.flatnonzero(np.isclose(EQUILIBRIUM_HORIZON_GRID, 3.0))[0])
    for regime in Regime:
        for closure_index, coefficient_label in enumerate(("log-linear", "log-quadratic")):
            rows.append(
                [
                    equilibrium_coefficients[closure_index, int(regime), curve_index],
                    physical_fk[int(regime), horizon_index, 0],
                    physical_fk[int(regime), horizon_index, 1],
                ]
            )
            metadata.append(
                {
                    "check": rf"$g(0,\theta)$, 3y, {coefficient_label}",
                    "regime": regime_labels[int(regime)],
                }
            )
    return np.asarray(rows), metadata


def build_numerical_payload(profile: ChapterProfile | str) -> dict[str, Any]:
    """Compute the complete package-owned chapter payload.

    Setting :data:`FORBID_RECOMPUTE_ENV` to ``"1"`` is a test-only safety
    guard.  It fails before profile parsing, file access, equilibrium solves, or
    any other numerical work.  Payload loading and rendering never call this
    function.
    """

    if os.environ.get(FORBID_RECOMPUTE_ENV) == "1":
        raise RuntimeError(f"numerical recomputation forbidden by {FORBID_RECOMPUTE_ENV}=1")
    config = profile_config(profile)
    if not ACCEPTANCE_MANIFEST.is_file():
        raise FileNotFoundError(f"acceptance manifest is missing: {ACCEPTANCE_MANIFEST}")
    frozen_hash = _check_frozen_oracle_hash()

    cache = _AnalyticsCache()
    closures = (EquilibriumClosure.LOG_LINEAR, EquilibriumClosure.LOG_QUADRATIC)
    closure_labels = ("log_linear", "log_quadratic")
    regime_labels = ("growth", "stress")
    baseline_params = tuple(equity_params(closure=closure) for closure in closures)
    equilibria = tuple(cache.equilibrium(params) for params in baseline_params)

    closure_prices = np.empty((2, 2, SMILE_MATURITIES.size, SMILE_LOG_MONEYNESS.size))
    closure_ivols = np.empty_like(closure_prices)
    for closure_index, params in enumerate(baseline_params):
        conditional = cache.chain(params, SMILE_MATURITIES, SMILE_LOG_MONEYNESS)
        closure_prices[closure_index] = _conditional_array(conditional, "prices")
        closure_ivols[closure_index] = _conditional_array(conditional, "ivols")

    maturity_3m = int(np.flatnonzero(np.isclose(SMILE_MATURITIES, 0.25))[0])
    risk_powers = np.array([-0.75, -0.5, -0.25, 0.0])
    risk_aversion_ivols = np.empty((risk_powers.size, 2, SMILE_LOG_MONEYNESS.size))
    for power_index, utility_power in enumerate(risk_powers):
        if utility_power == -0.5:
            risk_aversion_ivols[power_index] = closure_ivols[0, :, maturity_3m]
        else:
            params = equity_params(utility_power=float(utility_power))
            conditional = cache.chain(params, [0.25], SMILE_LOG_MONEYNESS)
            risk_aversion_ivols[power_index] = _conditional_array(conditional, "ivols")[:, 0]

    channel_names = (
        "all_zero",
        "diffusive_only",
        "timing_only",
        "tail_only",
        "full",
    )
    channel_scales = (
        RegimeRiskPremiaScales(0.0, 0.0, 0.0, 0.0),
        RegimeRiskPremiaScales(1.0, 1.0, 0.0, 0.0),
        RegimeRiskPremiaScales(0.0, 0.0, 1.0, 0.0),
        RegimeRiskPremiaScales(0.0, 0.0, 0.0, 1.0),
        RegimeRiskPremiaScales(),
    )
    channel_prices = np.empty((5, 2, SMILE_LOG_MONEYNESS.size))
    channel_ivols = np.empty_like(channel_prices)
    transform_phi = np.array([0.0, -1.0, -0.5 + 2.0j])
    channel_log_mgf = np.empty((5, 2, transform_phi.size, 2))
    linear_params = baseline_params[0]
    linear_equilibrium = equilibria[0]
    for scenario_index, scales in enumerate(channel_scales):
        if scales.is_full_equilibrium:
            channel_prices[scenario_index] = closure_prices[0, :, maturity_3m]
            channel_ivols[scenario_index] = closure_ivols[0, :, maturity_3m]
        else:
            conditional = cache.chain(
                linear_params,
                [0.25],
                SMILE_LOG_MONEYNESS,
                scales=scales,
            )
            channel_prices[scenario_index] = _conditional_array(conditional, "prices")[:, 0]
            channel_ivols[scenario_index] = _conditional_array(conditional, "ivols")[:, 0]
        log_mgf = compute_regime_switch_log_mgf_grid(
            linear_params,
            ttm=0.25,
            phi_grid=transform_phi,
            equilibrium=linear_equilibrium,
            scales=scales,
            expansion_order=ExpansionOrder.SECOND,
        )
        channel_log_mgf[scenario_index, :, :, 0] = np.real(log_mgf)
        channel_log_mgf[scenario_index, :, :, 1] = np.imag(log_mgf)

    refinement_prices = np.empty((2, 2, 2, SMILE_LOG_MONEYNESS.size))
    for closure_index, params in enumerate(baseline_params):
        refined = cache.chain(
            params,
            [0.25],
            SMILE_LOG_MONEYNESS,
            max_phi=FOURIER_REFINEMENT_POINTS,
        )
        refinement_prices[closure_index, 0] = closure_prices[closure_index, :, maturity_3m]
        refinement_prices[closure_index, 1] = _conditional_array(refined, "prices")[:, 0]

    transition_intensity = np.empty((2, SIGMA_GRID.size))
    risk_neutral_drift = np.empty((2, 2, SIGMA_GRID.size))
    physical_drift = np.empty((2, SIGMA_GRID.size))
    volatility_loading = np.empty((2, SIGMA_GRID.size))
    log_timing_ratio = np.empty((2, SIGMA_GRID.size))
    for regime in Regime:
        dynamics = linear_params.regimes[regime]
        physical_drift[int(regime)] = (dynamics.kappa1 + dynamics.kappa2 * SIGMA_GRID) * (
            dynamics.theta - SIGMA_GRID
        )
        for closure_index, (params, equilibrium) in enumerate(zip(baseline_params, equilibria)):
            state = evaluate_risk_neutral_state(
                params,
                equilibrium,
                horizon=3.0,
                sigma=SIGMA_GRID,
                regime=regime,
            )
            risk_neutral_drift[closure_index, int(regime)] = state.volatility_drift
            if closure_index == 0:
                transition_intensity[int(regime)] = state.transition_intensity
                volatility_loading[int(regime)] = state.volatility_loading
                log_timing_ratio[int(regime)] = state.log_timing_ratio

    jump_arithmetic_mean = np.empty((2, UTILITY_POWER_GRID.size))
    for power_index, utility_power in enumerate(UTILITY_POWER_GRID):
        trial = equity_params(utility_power=float(utility_power))
        tail_tilt = utility_power - 1.0
        for regime in Regime:
            ell = trial.jump_mgf(regime, tail_tilt)
            jump_arithmetic_mean[int(regime), power_index] = (
                trial.jump_mgf(regime, tail_tilt + 1.0) / ell - 1.0
            )

    closure_difference_bp = 10_000.0 * (closure_ivols[1] - closure_ivols[0])
    left = int(np.argmin(np.abs(SMILE_LOG_MONEYNESS + 0.20)))
    right = int(np.argmin(np.abs(SMILE_LOG_MONEYNESS - 0.20)))
    closure_table = np.empty((2, SMILE_MATURITIES.size, 3))
    for regime in Regime:
        linear_skew = (
            closure_ivols[0, int(regime), :, left] - closure_ivols[0, int(regime), :, right]
        )
        quadratic_skew = (
            closure_ivols[1, int(regime), :, left] - closure_ivols[1, int(regime), :, right]
        )
        closure_table[int(regime), :, 0] = 100.0 * linear_skew
        closure_table[int(regime), :, 1] = 100.0 * quadratic_skew
        closure_table[int(regime), :, 2] = 10_000.0 * (quadratic_skew - linear_skew)

    validation_prices = np.empty((2, 2, VALIDATION_LOG_MONEYNESS.size))
    validation_ivols = np.empty_like(validation_prices)
    for closure_index, params in enumerate(baseline_params):
        conditional = cache.chain(params, [0.25], VALIDATION_LOG_MONEYNESS)
        validation_prices[closure_index] = _conditional_array(conditional, "prices")[:, 0]
        validation_ivols[closure_index] = _conditional_array(conditional, "ivols")[:, 0]

    validation_mc_prices = np.empty_like(validation_prices)
    validation_mc_errors = np.empty_like(validation_prices)
    forward_martingale = np.empty((2, 2, 2))
    validation_strikes = np.exp(VALIDATION_LOG_MONEYNESS)
    validation_optiontypes = np.where(validation_strikes < 1.0, "P", "C")
    state_seeds = {Regime.GROWTH: 1_227, Regime.STRESS: 1_229}
    for closure_index, (params, equilibrium) in enumerate(zip(baseline_params, equilibria)):
        for regime in Regime:
            sample = simulate_regime_switch_logsv_terminal(
                params,
                ttm=0.25,
                equilibrium=equilibrium,
                initial_regime=regime,
                nb_path=config.monte_carlo_paths,
                nb_steps_per_year=config.steps_per_year,
                seed=state_seeds[regime],
            )
            prices, errors = compute_mc_vars_payoff(
                x0=sample.log_forward_return,
                sigma0=sample.sigma,
                qvar0=sample.qvar,
                ttm=0.25,
                forward=1.0,
                strikes_ttm=validation_strikes,
                optiontypes_ttm=validation_optiontypes,
                discfactor=1.0,
                variable_type=VariableType.LOG_RETURN,
            )
            validation_mc_prices[closure_index, int(regime)] = prices
            validation_mc_errors[closure_index, int(regime)] = errors
            forward_martingale[closure_index, int(regime)] = sample.forward_martingale

    equilibrium_coefficients = np.empty((2, 2, EQUILIBRIUM_HORIZON_GRID.size))
    for closure_index, equilibrium in enumerate(equilibria):
        for regime in Regime:
            sigma = baseline_params[closure_index].regimes[regime].theta
            equilibrium_coefficients[closure_index, int(regime)] = [
                np.exp(equilibrium.log_value_coefficient(float(horizon), sigma, regime))
                for horizon in EQUILIBRIUM_HORIZON_GRID
            ]

    physical_fk = _physical_fk_values(config)
    validation_table, validation_row_metadata = _build_validation_table(
        validation_prices,
        validation_mc_prices,
        validation_mc_errors,
        forward_martingale,
        equilibrium_coefficients,
        physical_fk,
    )

    maturity_labels = ("1 month", "3 months", "6 months", "1 year")
    records = {
        "closure.implied_volatility": array_record(
            closure_ivols,
            (
                ("closure", closure_labels),
                ("regime", regime_labels),
                ("maturity_years", SMILE_MATURITIES),
                ("log_moneyness", SMILE_LOG_MONEYNESS),
            ),
        ),
        "closure.implied_volatility_difference_bp": array_record(
            closure_difference_bp,
            (
                ("regime", regime_labels),
                ("maturity_years", SMILE_MATURITIES),
                ("log_moneyness", SMILE_LOG_MONEYNESS),
            ),
        ),
        "closure.table": array_record(
            closure_table,
            (
                ("regime", ("Growth", "Stress")),
                ("maturity", maturity_labels),
                ("metric", ("quadratic_skew_percent", "cubic_skew_percent", "difference_bp")),
            ),
        ),
        "premia.jump_arithmetic_mean": array_record(
            jump_arithmetic_mean,
            (("regime", regime_labels), ("utility_power", UTILITY_POWER_GRID)),
        ),
        "premia.log_timing_ratio": array_record(
            log_timing_ratio,
            (("regime", regime_labels), ("sigma", SIGMA_GRID)),
        ),
        "premia.physical_drift": array_record(
            physical_drift,
            (("regime", regime_labels), ("sigma", SIGMA_GRID)),
        ),
        "premia.physical_jump_arithmetic_mean": array_record(
            np.asarray(
                [transition.arithmetic_jump_mean for transition in linear_params.transitions]
            ),
            (("regime", regime_labels),),
        ),
        "premia.physical_transition_intensity": array_record(
            np.asarray([transition.intensity for transition in linear_params.transitions]),
            (("regime", regime_labels),),
        ),
        "premia.risk_neutral_drift": array_record(
            risk_neutral_drift,
            (
                ("closure", closure_labels),
                ("regime", regime_labels),
                ("sigma", SIGMA_GRID),
            ),
        ),
        "premia.transition_intensity": array_record(
            transition_intensity,
            (("regime", regime_labels), ("sigma", SIGMA_GRID)),
        ),
        "premia.volatility_loading": array_record(
            volatility_loading,
            (("regime", regime_labels), ("sigma", SIGMA_GRID)),
        ),
        "smiles.channel_implied_volatility": array_record(
            channel_ivols,
            (
                ("scenario", channel_names),
                ("regime", regime_labels),
                ("log_moneyness", SMILE_LOG_MONEYNESS),
            ),
        ),
        "smiles.channel_log_mgf": array_record(
            channel_log_mgf,
            (
                ("scenario", channel_names),
                ("regime", regime_labels),
                ("phi", ("0", "-1", "-0.5+2j")),
                ("component", ("real", "imag")),
            ),
        ),
        "smiles.channel_prices": array_record(
            channel_prices,
            (
                ("scenario", channel_names),
                ("regime", regime_labels),
                ("log_moneyness", SMILE_LOG_MONEYNESS),
            ),
        ),
        "smiles.initial_regime_implied_volatility": array_record(
            closure_ivols[0, :, maturity_3m],
            (("regime", regime_labels), ("log_moneyness", SMILE_LOG_MONEYNESS)),
        ),
        "smiles.risk_aversion_implied_volatility": array_record(
            risk_aversion_ivols,
            (
                ("utility_power", risk_powers),
                ("regime", regime_labels),
                ("log_moneyness", SMILE_LOG_MONEYNESS),
            ),
        ),
        "smiles.term_structure_implied_volatility": array_record(
            closure_ivols[0, int(Regime.GROWTH)],
            (("maturity_years", SMILE_MATURITIES), ("log_moneyness", SMILE_LOG_MONEYNESS)),
        ),
        "validation.analytic_implied_volatility": array_record(
            validation_ivols,
            (
                ("closure", closure_labels),
                ("regime", regime_labels),
                ("log_moneyness", VALIDATION_LOG_MONEYNESS),
            ),
        ),
        "validation.analytic_prices": array_record(
            validation_prices,
            (
                ("closure", closure_labels),
                ("regime", regime_labels),
                ("log_moneyness", VALIDATION_LOG_MONEYNESS),
            ),
        ),
        "validation.equilibrium_coefficients": array_record(
            equilibrium_coefficients,
            (
                ("closure", closure_labels),
                ("regime", regime_labels),
                ("horizon_years", EQUILIBRIUM_HORIZON_GRID),
            ),
        ),
        "validation.forward_martingale": array_record(
            forward_martingale,
            (
                ("closure", closure_labels),
                ("regime", regime_labels),
                ("metric", ("estimate", "standard_error")),
            ),
        ),
        "validation.fourier_refinement_prices": array_record(
            refinement_prices,
            (
                ("closure", closure_labels),
                ("fourier_points", (FOURIER_POINTS, FOURIER_REFINEMENT_POINTS)),
                ("regime", regime_labels),
                ("log_moneyness", SMILE_LOG_MONEYNESS),
            ),
        ),
        "validation.mc_prices": array_record(
            validation_mc_prices,
            (
                ("closure", closure_labels),
                ("regime", regime_labels),
                ("log_moneyness", VALIDATION_LOG_MONEYNESS),
            ),
        ),
        "validation.mc_standard_errors": array_record(
            validation_mc_errors,
            (
                ("closure", closure_labels),
                ("regime", regime_labels),
                ("log_moneyness", VALIDATION_LOG_MONEYNESS),
            ),
        ),
        "validation.physical_feynman_kac": array_record(
            physical_fk,
            (
                ("regime", regime_labels),
                ("horizon_years", FK_HORIZONS),
                ("metric", ("estimate", "standard_error")),
            ),
            source="hash-pinned frozen monolith; physical-measure validation only",
        ),
        "validation.table": array_record(
            validation_table,
            (
                ("row", range(validation_table.shape[0])),
                ("metric", ("analytic", "monte_carlo", "mc_standard_error")),
            ),
            row_metadata=validation_row_metadata,
        ),
    }
    payload = {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "profile": config.profile.value,
        "configuration": {
            "fourier_points": FOURIER_POINTS,
            "fourier_refinement_points": FOURIER_REFINEMENT_POINTS,
            "monte_carlo_paths": config.monte_carlo_paths,
            "q_monte_carlo_seeds": {"growth": 1_227, "stress": 1_229},
            "steps_per_year": config.steps_per_year,
        },
        "provenance": {
            "acceptance_manifest": {
                "path": _repository_relative(ACCEPTANCE_MANIFEST),
                "sha256": _sha256(ACCEPTANCE_MANIFEST),
            },
            "frozen_physical_feynman_kac_oracle": {
                "path": _repository_relative(FROZEN_ORACLE),
                "sha256": frozen_hash,
                "usage": "physical_feynman_kac_validation_only",
            },
            "production_analytics": "stochvolmodels package APIs",
        },
        "records": records,
    }
    return validate_numerical_payload(payload)


def write_artifact_manifest(
    output_directory: Path | str,
    *,
    profile: ChapterProfile | str,
    mode: str,
    payload_path: Path,
    artifacts: Sequence[Path],
) -> Path:
    """Write output-relative hashes for one computed or rerendered run."""

    if mode not in {"computed", "rerendered"}:
        raise ValueError("artifact-manifest mode must be 'computed' or 'rerendered'")
    output = validate_output_directory(output_directory)
    payload = Path(payload_path).resolve()
    try:
        payload_relative = payload.relative_to(output).as_posix()
    except ValueError as error:
        raise ValueError("payload_path must be inside output_directory") from error
    entries = []
    for artifact in sorted({Path(path).resolve() for path in artifacts}, key=str):
        if not artifact.is_file():
            raise FileNotFoundError(f"expected artifact is missing: {artifact}")
        try:
            relative = artifact.relative_to(output).as_posix()
        except ValueError as error:
            raise ValueError(f"artifact is outside output directory: {artifact}") from error
        entries.append({"path": relative, "sha256": _sha256(artifact)})
    manifest = {
        "schema_version": 1,
        "profile": as_profile(profile).value,
        "mode": mode,
        "payload": {"path": payload_relative, "sha256": _sha256(payload)},
        "artifacts": entries,
    }
    _validate_finite_json(manifest, "artifact_manifest")
    target = output / ARTIFACT_MANIFEST_FILENAME
    target.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target


__all__ = [
    "ACCEPTANCE_MANIFEST",
    "ARTIFACT_MANIFEST_FILENAME",
    "ChapterConfig",
    "ChapterProfile",
    "DEFAULT_OUTPUT_ROOT",
    "EXPECTED_FROZEN_ORACLE_SHA256",
    "FORBID_RECOMPUTE_ENV",
    "PAYLOAD_FILENAME",
    "REPOSITORY_ROOT",
    "SCRIPT_DIR",
    "array_record",
    "as_profile",
    "build_numerical_payload",
    "default_output_directory",
    "equity_params",
    "load_numerical_payload",
    "profile_config",
    "record_array",
    "record_axis",
    "validate_numerical_payload",
    "validate_output_directory",
    "write_artifact_manifest",
    "write_numerical_payload",
]
