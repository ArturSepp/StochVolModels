"""Package-routed analytics and portable payloads for the Student risk-premia chapter."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from enum import Enum
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import vanilla_option_pricers as bsm

from stochvolmodels.data.option_chain import OptionSlice
from stochvolmodels.fitters.tdist import imply_drift_tdist
from stochvolmodels.models.inverse_gamma_normal import (
    InverseGammaNormalParams,
    InverseGammaNormalTerminalModel,
)
from stochvolmodels.pricers.tdist_pricer import TdistParams, TdistTerminalModel

CHAPTER_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ACCEPTANCE_MANIFEST = CHAPTER_DIR / "acceptance_manifest.json"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs" / "volatility_book" / "ch_tdist_risk_premia"
PAYLOAD_FILENAME = "numerical_payload.json"
ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"
FORBID_RECOMPUTE_ENV = "STOCHVOLMODELS_TDIST_RISK_PREMIA_FORBID_RECOMPUTE"
PAYLOAD_SCHEMA_VERSION = 1
EXPECTED_CURVE_CAPTURE_SHA256 = "2f78c1e678774f692946fc37f97496c754a0f306604af136d5800a342fbefeae"
LOG_MONEYNESS = np.linspace(-0.35, 0.35, 29)
EXPECTED_SCENARIO_IDS = (
    "p_minus",
    "baseline_p",
    "p_plus",
    "eta_minus",
    "baseline_eta",
    "eta_plus",
    "q_minus",
    "baseline_q",
    "q_plus",
    "p_fixed_minus",
    "baseline_fixed",
    "p_fixed_plus",
)
VALIDATION_THRESHOLDS = {
    "max_put_call_parity_error": 1.0e-12,
    "max_black_round_trip_error": 1.0e-12,
    "max_128_vs_256_node_error": 3.0e-7,
    "max_symmetric_student_price_error": 5.0e-8,
    "max_symmetric_student_iv_error": 1.0e-6,
    "max_nonunit_discount_student_error": 4.0e-6,
    "max_call_monotonicity_violation": 1.0e-12,
    "max_call_convexity_violation": 1.0e-12,
    "max_variance_neutral_mean_v_error": 1.0e-13,
    "max_duplicate_baseline_curve_error": 1.0e-13,
}


class ChapterProfile(str, Enum):
    """Supported output profiles for the deterministic chapter pipeline."""

    CANONICAL = "canonical"


@dataclass(frozen=True, slots=True)
class ModelSetup:
    """Physical baseline and pricing conventions for the one-period example."""

    alpha_p: float = 4.0
    beta_p: float = 0.12
    c: float = 1.0
    ttm: float = 1.0
    forward: float = 1.0
    discfactor: float = 1.0
    quadrature_order: int = 256

    @property
    def physical_mean_v(self) -> float:
        """Return the physical inverse-gamma mean."""

        return self.beta_p / (self.alpha_p - 1.0)


@dataclass(frozen=True, slots=True)
class Scenario:
    """One chapter-owned pricing-kernel parameter combination."""

    identifier: str
    panel: str
    label: str
    p: float = 0.0
    eta: float = 0.0
    q: float = 0.0


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Package-computed smile and diagnostics for one scenario."""

    scenario: Scenario
    alpha_q: float
    beta_q: float
    mean_v: float
    shift: float
    default_probability: float
    log_moneyness: np.ndarray
    strikes: np.ndarray
    call_prices: np.ndarray
    implied_volatilities: np.ndarray
    atm_iv: float
    atm_skew: float
    rr_025: float
    bf_025: float


def as_profile(value: ChapterProfile | str) -> ChapterProfile:
    """Return a validated chapter profile."""

    if isinstance(value, ChapterProfile):
        return value
    try:
        return ChapterProfile(str(value))
    except ValueError as error:
        choices = ", ".join(profile.value for profile in ChapterProfile)
        raise ValueError(f"profile must be one of: {choices}") from error


def default_output_directory(profile: ChapterProfile | str) -> Path:
    """Return the ignored output directory for ``profile``."""

    return DEFAULT_OUTPUT_ROOT / as_profile(profile).value


def _assert_recompute_allowed() -> None:
    if os.environ.get(FORBID_RECOMPUTE_ENV) == "1":
        raise RuntimeError(f"numerical recomputation forbidden by {FORBID_RECOMPUTE_ENV}=1")


def validate_output_directory(path: Path | str) -> Path:
    """Resolve output and confine repository-local artifacts to the ignored root."""

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
            f"repository-local Student risk-premia artifacts must be below {DEFAULT_OUTPUT_ROOT}"
        ) from error
    return output


def raw_scenarios() -> tuple[Scenario, ...]:
    """Return the nine direct comparative-static scenarios in display order."""

    return (
        Scenario("p_minus", "p", "p = -0.75", p=-0.75),
        Scenario("baseline_p", "p", "p = 0"),
        Scenario("p_plus", "p", "p = +0.75", p=0.75),
        Scenario("eta_minus", "eta", "eta = -0.024", eta=-0.024),
        Scenario("baseline_eta", "eta", "eta = 0"),
        Scenario("eta_plus", "eta", "eta = +0.024", eta=0.024),
        Scenario("q_minus", "q", "q = -2", q=-2.0),
        Scenario("baseline_q", "q", "q = 0"),
        Scenario("q_plus", "q", "q = +2", q=2.0),
    )


def fixed_variance_scenarios(setup: ModelSetup) -> tuple[Scenario, ...]:
    """Return the three tail-premium scenarios with pricing mean variance fixed."""

    mean_v = setup.physical_mean_v
    return tuple(
        Scenario(
            ("p_fixed_minus" if p < 0.0 else "p_fixed_plus" if p > 0.0 else "baseline_fixed"),
            "p_fixed_variance",
            ("p = 0, eta = 0" if abs(p) < 1.0e-15 else f"p = {p:+.2f}, eta = {-mean_v * p:+.3f}"),
            p=p,
            eta=-mean_v * p,
        )
        for p in (-0.75, 0.0, 0.75)
    )


def pricing_law(setup: ModelSetup, scenario: Scenario) -> InverseGammaNormalTerminalModel:
    """Map chapter economics to the package-owned risk-neutral terminal law."""

    _assert_recompute_allowed()
    return InverseGammaNormalTerminalModel(
        InverseGammaNormalParams(
            alpha=setup.alpha_p - scenario.p,
            beta=setup.beta_p + scenario.eta,
            c=setup.c,
            q=scenario.q,
            ttm=setup.ttm,
        ),
        quadrature_order=setup.quadrature_order,
    )


def _option_slice(
    setup: ModelSetup,
    strikes: np.ndarray,
    optiontypes: np.ndarray,
    *,
    identifier: str,
) -> OptionSlice:
    return OptionSlice(
        ttm=setup.ttm,
        forward=setup.forward,
        strikes=np.asarray(strikes, dtype=float),
        optiontypes=np.asarray(optiontypes),
        discfactor=setup.discfactor,
        id=identifier,
    )


def _smile_statistics(
    log_moneyness: np.ndarray,
    implied_volatilities: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return ATM IV, local ATM slope, fixed-log-moneyness RR and BF."""

    atm_index = int(np.argmin(np.abs(log_moneyness)))
    step = float(log_moneyness[atm_index + 1] - log_moneyness[atm_index])
    left_index = int(np.argmin(np.abs(log_moneyness + 0.25)))
    right_index = int(np.argmin(np.abs(log_moneyness - 0.25)))
    atm_skew = float(
        (implied_volatilities[atm_index + 1] - implied_volatilities[atm_index - 1]) / (2.0 * step)
    )
    rr_025 = float(implied_volatilities[right_index] - implied_volatilities[left_index])
    bf_025 = float(
        0.5 * (implied_volatilities[left_index] + implied_volatilities[right_index])
        - implied_volatilities[atm_index]
    )
    return float(implied_volatilities[atm_index]), atm_skew, rr_025, bf_025


def compute_scenario(
    setup: ModelSetup,
    scenario: Scenario,
    log_moneyness: np.ndarray = LOG_MONEYNESS,
) -> ScenarioResult:
    """Compute one scenario entirely through the package terminal-law interface."""

    _assert_recompute_allowed()
    model = pricing_law(setup, scenario)
    log_strikes = np.asarray(log_moneyness, dtype=float)
    strikes = setup.forward * np.exp(log_strikes)
    option_slice = _option_slice(
        setup,
        strikes,
        np.full(strikes.size, "C"),
        identifier=f"tdist-risk-premia-{scenario.identifier}",
    )
    calls = model.price_european(option_slice)
    implied_volatilities = model.implied_vols(option_slice)
    atm_iv, atm_skew, rr_025, bf_025 = _smile_statistics(log_strikes, implied_volatilities)
    return ScenarioResult(
        scenario=scenario,
        alpha_q=model.params.alpha,
        beta_q=model.params.beta,
        mean_v=model.mean_mixing_variance,
        shift=model.martingale_shift(discfactor=setup.discfactor),
        default_probability=model.default_probability(discfactor=setup.discfactor),
        log_moneyness=log_strikes,
        strikes=strikes,
        call_prices=calls,
        implied_volatilities=implied_volatilities,
        atm_iv=atm_iv,
        atm_skew=atm_skew,
        rr_025=rr_025,
        bf_025=bf_025,
    )


def _student_reference_model(
    params: InverseGammaNormalParams,
    *,
    discfactor: float,
) -> TdistTerminalModel:
    rf_rate = -math.log(discfactor) / params.ttm
    vol = math.sqrt(params.c * params.beta / ((params.alpha - 1.0) * params.ttm))
    nu = 2.0 * params.alpha
    drift = imply_drift_tdist(rf_rate=rf_rate, vol=vol, nu=nu, ttm=params.ttm)
    return TdistTerminalModel(TdistParams(drift=drift, vol=vol, nu=nu, ttm=params.ttm))


def _nonunit_discount_student_error() -> float:
    setup = ModelSetup(ttm=0.75, forward=105.0, discfactor=0.96)
    scenario = Scenario("nonunit_discount", "validation", "non-unit discount")
    model = pricing_law(setup, scenario)
    strikes = setup.forward * np.exp(np.array([-0.20, 0.0, 0.20]))
    option_slice = _option_slice(
        setup,
        strikes,
        np.array(["P", "C", "C"]),
        identifier="tdist-risk-premia-nonunit-discount",
    )
    student = _student_reference_model(model.params, discfactor=setup.discfactor)
    return float(
        np.max(np.abs(model.price_european(option_slice) - student.price_european(option_slice)))
    )


def _validation_metrics(setup: ModelSetup, results: Sequence[ScenarioResult]) -> dict[str, float]:
    parity_errors: list[float] = []
    black_errors: list[float] = []
    refinement_errors: list[float] = []
    student_price_errors: list[float] = []
    student_iv_errors: list[float] = []
    monotonicity_violations: list[float] = []
    convexity_violations: list[float] = []

    unique_results: dict[tuple[float, float, float], ScenarioResult] = {}
    for result in results:
        unique_results.setdefault((result.alpha_q, result.beta_q, result.scenario.q), result)

    for result in unique_results.values():
        model = pricing_law(setup, result.scenario)
        put_slice = _option_slice(
            setup,
            result.strikes,
            np.full(result.strikes.size, "P"),
            identifier=f"tdist-risk-premia-{result.scenario.identifier}-puts",
        )
        puts = model.price_european(put_slice)
        parity_errors.extend(
            np.abs(result.call_prices - puts - setup.discfactor * (setup.forward - result.strikes))
        )
        repriced = np.array(
            [
                bsm.compute_bsm_vanilla_price(
                    forward=setup.forward,
                    strike=float(strike),
                    ttm=setup.ttm,
                    vol=float(volatility),
                    optiontype="C",
                    discfactor=setup.discfactor,
                )
                for strike, volatility in zip(
                    result.strikes,
                    result.implied_volatilities,
                )
            ],
            dtype=float,
        )
        black_errors.extend(np.abs(np.asarray(repriced, dtype=float) - result.call_prices))
        monotonicity_violations.append(float(max(0.0, np.max(np.diff(result.call_prices)))))
        slopes = np.diff(result.call_prices) / np.diff(result.strikes)
        convexity_violations.append(float(max(0.0, -np.min(np.diff(slopes)))))

        lower_setup = ModelSetup(**{**asdict(setup), "quadrature_order": 128})
        lower_model = pricing_law(lower_setup, result.scenario)
        selected = np.array([math.exp(-0.25), 1.0, math.exp(0.25)]) * setup.forward
        selected_slice = _option_slice(
            setup,
            selected,
            np.full(selected.size, "C"),
            identifier=f"tdist-risk-premia-{result.scenario.identifier}-refinement",
        )
        refinement_errors.extend(
            np.abs(
                model.price_european(selected_slice) - lower_model.price_european(selected_slice)
            )
        )

        if abs(result.scenario.q) < 1.0e-15:
            call_slice = _option_slice(
                setup,
                result.strikes,
                np.full(result.strikes.size, "C"),
                identifier=f"tdist-risk-premia-{result.scenario.identifier}-student",
            )
            student = _student_reference_model(model.params, discfactor=setup.discfactor)
            student_price_errors.extend(
                np.abs(student.price_european(call_slice) - result.call_prices)
            )
            student_iv_errors.extend(
                np.abs(student.implied_vols(call_slice) - result.implied_volatilities)
            )

    fixed = results[9:]
    baseline = [
        result
        for result in results
        if abs(result.scenario.p) < 1.0e-15
        and abs(result.scenario.eta) < 1.0e-15
        and abs(result.scenario.q) < 1.0e-15
    ]
    return {
        "max_put_call_parity_error": float(max(parity_errors)),
        "max_black_round_trip_error": float(max(black_errors)),
        "max_128_vs_256_node_error": float(max(refinement_errors)),
        "max_symmetric_student_price_error": float(max(student_price_errors)),
        "max_symmetric_student_iv_error": float(max(student_iv_errors)),
        "max_nonunit_discount_student_error": _nonunit_discount_student_error(),
        "max_call_monotonicity_violation": max(monotonicity_violations),
        "max_call_convexity_violation": max(convexity_violations),
        "max_variance_neutral_mean_v_error": max(
            abs(result.mean_v - setup.physical_mean_v) for result in fixed
        ),
        "max_duplicate_baseline_curve_error": max(
            float(np.max(np.abs(result.implied_volatilities - baseline[0].implied_volatilities)))
            for result in baseline[1:]
        ),
    }


def enforce_validation(validation: dict[str, float]) -> None:
    """Raise when a package or book-production acceptance limit is exceeded."""

    failures = {
        name: (validation[name], threshold)
        for name, threshold in VALIDATION_THRESHOLDS.items()
        if not np.isfinite(validation[name])
        or validation[name] < 0.0
        or validation[name] > threshold
    }
    if failures:
        details = ", ".join(
            f"{name}={value:.3e} outside [0,{threshold:.3e}]"
            for name, (value, threshold) in failures.items()
        )
        raise RuntimeError(f"Student risk-premia validation failed: {details}")


def compute_examples(
    setup: ModelSetup | None = None,
) -> tuple[list[ScenarioResult], dict[str, float]]:
    """Compute all 12 chapter scenarios and package-owned validation metrics."""

    _assert_recompute_allowed()
    selected_setup = ModelSetup() if setup is None else setup
    scenarios = raw_scenarios() + fixed_variance_scenarios(selected_setup)
    cached_results: dict[tuple[float, float, float], ScenarioResult] = {}
    results = []
    for scenario in scenarios:
        key = (
            selected_setup.alpha_p - scenario.p,
            selected_setup.beta_p + scenario.eta,
            scenario.q,
        )
        if key not in cached_results:
            cached_results[key] = compute_scenario(selected_setup, scenario)
        results.append(replace(cached_results[key], scenario=scenario))
    validation = _validation_metrics(selected_setup, results)
    enforce_validation(validation)
    return results, validation


def curve_capture(results: Sequence[ScenarioResult]) -> dict[str, Any]:
    """Return the portable rounded capture used by the frozen acceptance hash."""

    return {
        "log_moneyness": np.round(results[0].log_moneyness, 8).tolist(),
        "records": [
            {
                "alpha": round(result.alpha_q, 12),
                "beta": round(result.beta_q, 12),
                "q": round(result.scenario.q, 12),
                "shift": round(result.shift, 12),
                "default_probability": round(result.default_probability, 12),
                "call_prices": np.round(result.call_prices, 10).tolist(),
                "implied_volatilities": np.round(result.implied_volatilities, 10).tolist(),
            }
            for result in results
        ],
    }


def _compact_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _result_payload(result: ScenarioResult) -> dict[str, Any]:
    return {
        "scenario": asdict(result.scenario),
        "alpha_q": result.alpha_q,
        "beta_q": result.beta_q,
        "mean_v": result.mean_v,
        "shift": result.shift,
        "default_probability": result.default_probability,
        "log_moneyness": result.log_moneyness.tolist(),
        "strikes": result.strikes.tolist(),
        "call_prices": result.call_prices.tolist(),
        "implied_volatilities": result.implied_volatilities.tolist(),
        "atm_iv": result.atm_iv,
        "atm_skew": result.atm_skew,
        "rr_025": result.rr_025,
        "bf_025": result.bf_025,
    }


def _validate_finite_json(value: Any, location: str = "payload") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_json(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location} is non-finite")


def _require_exact_keys(value: Any, expected: set[str], location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{location} must be a dictionary")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{location} keys differ: missing={missing}, extra={extra}")
    return value


def _finite_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{location} must be a JSON number, not {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{location} is non-finite")
    return number


def _finite_vector(value: Any, size: int, location: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{location} must contain exactly {size} values")
    return np.asarray(
        [_finite_number(item, f"{location}[{index}]") for index, item in enumerate(value)],
        dtype=float,
    )


def validate_numerical_payload(payload: Any) -> dict[str, Any]:
    """Validate the portable payload shape, values and internal curve hash."""

    required = {
        "schema_version",
        "chapter",
        "profile",
        "setup",
        "scenarios",
        "validation",
        "curve_capture_sha256",
        "provenance",
    }
    payload = _require_exact_keys(payload, required, "numerical payload")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != PAYLOAD_SCHEMA_VERSION
    ):
        raise ValueError("unsupported numerical payload schema version")
    if payload["chapter"] != "volatility_book/ch_tdist_risk_premia":
        raise ValueError("numerical payload chapter identifier is wrong")
    if not isinstance(payload["profile"], str):
        raise TypeError("numerical payload profile must be a string")
    as_profile(payload["profile"])

    setup = _require_exact_keys(
        payload["setup"], set(asdict(ModelSetup())), "numerical payload.setup"
    )
    expected_setup = asdict(ModelSetup())
    for name, expected in expected_setup.items():
        value = setup[name]
        if name == "quadrature_order":
            if type(value) is not int or value != expected:
                raise ValueError(f"numerical payload.setup.{name} differs from {expected}")
        elif _finite_number(value, f"numerical payload.setup.{name}") != float(expected):
            raise ValueError(f"numerical payload.setup.{name} differs from {expected}")

    scenarios = payload["scenarios"]
    if not isinstance(scenarios, list) or len(scenarios) != 12:
        raise ValueError("numerical payload must contain exactly 12 scenarios")
    expected_scenarios = raw_scenarios() + fixed_variance_scenarios(ModelSetup())
    result_keys = {
        "scenario",
        "alpha_q",
        "beta_q",
        "mean_v",
        "shift",
        "default_probability",
        "log_moneyness",
        "strikes",
        "call_prices",
        "implied_volatilities",
        "atm_iv",
        "atm_skew",
        "rr_025",
        "bf_025",
    }
    scenario_keys = {"identifier", "panel", "label", "p", "eta", "q"}
    expected_strikes = ModelSetup().forward * np.exp(LOG_MONEYNESS)
    for index, (record, expected_scenario) in enumerate(
        zip(scenarios, expected_scenarios, strict=True)
    ):
        record = _require_exact_keys(record, result_keys, f"scenario {index}")
        scenario = _require_exact_keys(
            record["scenario"], scenario_keys, f"scenario {index}.scenario"
        )
        expected_metadata = asdict(expected_scenario)
        for name in ("identifier", "panel", "label"):
            if not isinstance(scenario[name], str) or scenario[name] != expected_metadata[name]:
                raise ValueError(f"scenario {index}.scenario.{name} is wrong")
        for name in ("p", "eta", "q"):
            value = _finite_number(scenario[name], f"scenario {index}.scenario.{name}")
            if value != float(expected_metadata[name]):
                raise ValueError(f"scenario {index}.scenario.{name} is wrong")

        log_moneyness = _finite_vector(
            record["log_moneyness"], 29, f"scenario {index}.log_moneyness"
        )
        strikes = _finite_vector(record["strikes"], 29, f"scenario {index}.strikes")
        call_prices = _finite_vector(record["call_prices"], 29, f"scenario {index}.call_prices")
        implied_volatilities = _finite_vector(
            record["implied_volatilities"], 29, f"scenario {index}.implied_volatilities"
        )
        if not np.array_equal(log_moneyness, LOG_MONEYNESS):
            raise ValueError(f"scenario {index}.log_moneyness grid is wrong")
        if not np.allclose(strikes, expected_strikes, rtol=0.0, atol=1.0e-15):
            raise ValueError(f"scenario {index}.strikes are inconsistent with the setup")
        if np.any(strikes <= 0.0):
            raise ValueError(f"scenario {index}.strikes must be positive")
        if np.any(call_prices < 0.0):
            raise ValueError(f"scenario {index}.call_prices must be non-negative")
        if np.any(implied_volatilities <= 0.0):
            raise ValueError(f"scenario {index}.implied_volatilities must be positive")

        scalars = {
            name: _finite_number(record[name], f"scenario {index}.{name}")
            for name in (
                "alpha_q",
                "beta_q",
                "mean_v",
                "shift",
                "default_probability",
                "atm_iv",
                "atm_skew",
                "rr_025",
                "bf_025",
            )
        }
        expected_alpha = ModelSetup().alpha_p - expected_scenario.p
        expected_beta = ModelSetup().beta_p + expected_scenario.eta
        expected_mean_v = expected_beta / (expected_alpha - 1.0)
        for name, expected in (
            ("alpha_q", expected_alpha),
            ("beta_q", expected_beta),
            ("mean_v", expected_mean_v),
        ):
            if not math.isclose(scalars[name], expected, rel_tol=0.0, abs_tol=1.0e-15):
                raise ValueError(f"scenario {index}.{name} is inconsistent with its law")
        if not 0.0 <= scalars["default_probability"] <= 1.0:
            raise ValueError(f"scenario {index}.default_probability is outside [0, 1]")
        summaries = _smile_statistics(log_moneyness, implied_volatilities)
        for name, expected in zip(
            ("atm_iv", "atm_skew", "rr_025", "bf_025"), summaries, strict=True
        ):
            if not math.isclose(scalars[name], expected, rel_tol=0.0, abs_tol=1.0e-14):
                raise ValueError(f"scenario {index}.{name} is inconsistent with its smile")

    scenario_ids = tuple(record["scenario"]["identifier"] for record in scenarios)
    if scenario_ids != EXPECTED_SCENARIO_IDS:
        raise ValueError("numerical payload scenario identifiers or order are wrong")

    validation = _require_exact_keys(
        payload["validation"], set(VALIDATION_THRESHOLDS), "numerical payload.validation"
    )
    validated_metrics = {
        name: _finite_number(validation[name], f"numerical payload.validation.{name}")
        for name in VALIDATION_THRESHOLDS
    }
    enforce_validation(validated_metrics)

    provenance = _require_exact_keys(
        payload["provenance"],
        {"acceptance_manifest", "production_model", "frozen_oracle_used_in_production"},
        "numerical payload.provenance",
    )
    manifest_provenance = _require_exact_keys(
        provenance["acceptance_manifest"],
        {"path", "sha256"},
        "numerical payload.provenance.acceptance_manifest",
    )
    expected_manifest_path = "volatility_book/ch_tdist_risk_premia/acceptance_manifest.json"
    if manifest_provenance["path"] != expected_manifest_path:
        raise ValueError("numerical payload acceptance-manifest path is wrong")
    if manifest_provenance["sha256"] != _sha256(ACCEPTANCE_MANIFEST):
        raise ValueError("numerical payload acceptance-manifest hash is wrong")
    expected_model = "stochvolmodels.models.inverse_gamma_normal.InverseGammaNormalTerminalModel"
    if provenance["production_model"] != expected_model:
        raise ValueError("numerical payload production model is wrong")
    if provenance["frozen_oracle_used_in_production"] is not False:
        raise ValueError("numerical payload must record that production excludes the oracle")

    _validate_finite_json(payload)
    reconstructed = {
        "log_moneyness": np.round(scenarios[0]["log_moneyness"], 8).tolist(),
        "records": [
            {
                "alpha": round(float(record["alpha_q"]), 12),
                "beta": round(float(record["beta_q"]), 12),
                "q": round(float(record["scenario"]["q"]), 12),
                "shift": round(float(record["shift"]), 12),
                "default_probability": round(float(record["default_probability"]), 12),
                "call_prices": np.round(record["call_prices"], 10).tolist(),
                "implied_volatilities": np.round(record["implied_volatilities"], 10).tolist(),
            }
            for record in scenarios
        ],
    }
    actual_hash = _compact_sha256(reconstructed)
    if not isinstance(payload["curve_capture_sha256"], str):
        raise TypeError("numerical payload curve capture hash must be a string")
    if payload["curve_capture_sha256"] != actual_hash:
        raise ValueError("numerical payload curve capture hash is inconsistent")
    if actual_hash != EXPECTED_CURVE_CAPTURE_SHA256:
        raise ValueError("numerical payload differs from the frozen accepted curves")
    return payload


def build_numerical_payload(
    profile: ChapterProfile | str = ChapterProfile.CANONICAL,
) -> dict[str, Any]:
    """Compute and validate the package-routed numerical payload."""

    _assert_recompute_allowed()
    if not ACCEPTANCE_MANIFEST.is_file():
        raise FileNotFoundError(f"acceptance manifest is missing: {ACCEPTANCE_MANIFEST}")
    selected_profile = as_profile(profile)
    setup = ModelSetup()
    results, validation = compute_examples(setup)
    capture_hash = _compact_sha256(curve_capture(results))
    if capture_hash != EXPECTED_CURVE_CAPTURE_SHA256:
        raise RuntimeError("package-routed curves differ from the frozen acceptance hash")
    payload = {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "chapter": "volatility_book/ch_tdist_risk_premia",
        "profile": selected_profile.value,
        "setup": asdict(setup),
        "scenarios": [_result_payload(result) for result in results],
        "validation": validation,
        "curve_capture_sha256": capture_hash,
        "provenance": {
            "acceptance_manifest": {
                "path": "volatility_book/ch_tdist_risk_premia/acceptance_manifest.json",
                "sha256": _sha256(ACCEPTANCE_MANIFEST),
            },
            "production_model": (
                "stochvolmodels.models.inverse_gamma_normal.InverseGammaNormalTerminalModel"
            ),
            "frozen_oracle_used_in_production": False,
        },
    }
    return validate_numerical_payload(payload)


def write_numerical_payload(payload: dict[str, Any], path: Path | str) -> Path:
    """Write a validated payload with stable JSON formatting."""

    validated = validate_numerical_payload(payload)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(validated, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target


def load_numerical_payload(path: Path | str) -> dict[str, Any]:
    """Load and validate one payload, rejecting JSON non-finite constants."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as stream:
        payload = json.load(stream, parse_constant=reject_constant)
    return validate_numerical_payload(payload)


def write_artifact_manifest(
    output_directory: Path | str,
    *,
    profile: ChapterProfile | str,
    mode: str,
    payload_path: Path,
    artifacts: Iterable[Path],
) -> Path:
    """Write output-relative hashes for a computed or payload-only render."""

    if mode not in {"computed", "rerendered"}:
        raise ValueError("artifact manifest mode must be 'computed' or 'rerendered'")
    output = validate_output_directory(output_directory)
    payload = Path(payload_path).resolve()
    try:
        payload_relative = payload.relative_to(output).as_posix()
    except ValueError as error:
        raise ValueError("payload_path must be inside output_directory") from error
    entries = []
    for artifact in sorted({Path(item).resolve() for item in artifacts}, key=str):
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
    "ChapterProfile",
    "DEFAULT_OUTPUT_ROOT",
    "EXPECTED_CURVE_CAPTURE_SHA256",
    "EXPECTED_SCENARIO_IDS",
    "FORBID_RECOMPUTE_ENV",
    "LOG_MONEYNESS",
    "ModelSetup",
    "PAYLOAD_FILENAME",
    "Scenario",
    "ScenarioResult",
    "as_profile",
    "build_numerical_payload",
    "compute_examples",
    "compute_scenario",
    "curve_capture",
    "default_output_directory",
    "enforce_validation",
    "fixed_variance_scenarios",
    "load_numerical_payload",
    "pricing_law",
    "raw_scenarios",
    "validate_numerical_payload",
    "validate_output_directory",
    "write_artifact_manifest",
    "write_numerical_payload",
]
