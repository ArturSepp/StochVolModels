"""Contracts for pathwise products and discounted path valuation."""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

import stochvolmodels
from stochvolmodels.data.model_paths import ModelPaths
from stochvolmodels.products import Payoff
from stochvolmodels.products.payoffs import (
    EuropeanOptionPayoff,
    IntegratedVarianceOptionPayoff,
    VarianceQuote,
)
from stochvolmodels.utils.config import OptionType
from stochvolmodels.utils.mc_payoffs import compute_mc_vars_payoff
from stochvolmodels.valuation import (
    LegacyAdditiveForwardRecenter,
    PathEstimator,
    RecenterMean,
    value_paths,
    value_paths_self_normalized,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class _FixedPayoff:
    """Return caller-supplied values through the structural payoff protocol."""

    required_asset_ids: tuple[str, ...] = ()
    expiry = 1.0

    def __init__(
        self,
        values: object,
        *,
        payoff_unit: str = "USD",
        settlement_unit: str = "USD",
    ) -> None:
        self.values = values
        self.payoff_unit = payoff_unit
        self.settlement_unit = settlement_unit

    def __call__(self, paths: ModelPaths) -> object:
        del paths
        return self.values


def _paths(
    terminal_values: np.ndarray | list[float],
    *,
    observation_times: np.ndarray | None = None,
    sampling_measure: str = "Q",
    target_measure: str | None = None,
    numeraire: str = "money_market_account",
    log_likelihood_ratios: np.ndarray | None = None,
    states: dict[str, np.ndarray] | None = None,
    state_units: dict[str, str] | None = None,
) -> ModelPaths:
    """Build a minimal, fully declared path payload for deterministic tests."""
    terminal = np.asarray(terminal_values, dtype=float)
    times = np.array([0.0, 1.0]) if observation_times is None else observation_times
    assets = np.ones((terminal.size, times.size, 1))
    assets[:, -1, 0] = terminal
    return ModelPaths(
        observation_times=times,
        assets=assets,
        asset_ids=("asset",),
        sampling_measure=sampling_measure,
        target_measure=sampling_measure if target_measure is None else target_measure,
        numeraire=numeraire,
        scheme="deterministic_test_fixture",
        states={} if states is None else states,
        state_units={} if state_units is None else state_units,
        log_likelihood_ratios=log_likelihood_ratios,
    )


def _call(*, expiry: float = 1.0, strike: float = 0.0) -> EuropeanOptionPayoff:
    """Return the standard asset call used by estimator tests."""
    return EuropeanOptionPayoff(
        asset_id="asset",
        expiry=expiry,
        strike=strike,
        option_type=OptionType.CALL,
        unit="USD",
    )


def test_european_payoffs_are_explicit_structural_read_only_products() -> None:
    """Select one asset and expiry while preserving C/P payoff identities and units."""
    paths = _paths([0.5, 1.5, 3.0])
    call = _call(strike=1.0)
    put = EuropeanOptionPayoff(
        asset_id="asset",
        expiry=1.0,
        strike=1.0,
        option_type="P",
        unit="USD",
    )

    call_values = call(paths)
    put_values = put(paths)
    assert isinstance(call, Payoff)
    assert isinstance(put, Payoff)
    assert call.required_asset_ids == put.required_asset_ids == ("asset",)
    assert call.payoff_unit == call.settlement_unit == "USD"
    assert call.option_type is OptionType.CALL
    assert put.option_type is OptionType.PUT
    np.testing.assert_array_equal(call_values, [0.0, 0.5, 2.0])
    np.testing.assert_array_equal(put_values, [0.5, 0.0, 0.0])
    np.testing.assert_array_equal(call_values - put_values, paths.assets[:, -1, 0] - 1.0)
    assert call_values.dtype == np.float64
    assert not call_values.flags.writeable
    with pytest.raises(ValueError):
        call_values[0] = 99.0
    with pytest.raises(FrozenInstanceError):
        call.strike = 2.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"asset_id": " asset"},
        {"unit": ""},
        {"expiry": 0.0},
        {"expiry": np.inf},
        {"expiry": True},
        {"expiry": "1.0"},
        {"strike": -0.1},
        {"strike": np.nan},
        {"strike": False},
        {"option_type": "IC"},
        {"option_type": OptionType.INVERSE_PUT},
        {"option_type": "X"},
    ],
)
def test_european_constructor_rejects_ambiguous_or_unsupported_inputs(
    changes: dict[str, object],
) -> None:
    """Only finite, standard C/P products with explicit trimmed labels are admitted."""
    request: dict[str, object] = {
        "asset_id": "asset",
        "expiry": 1.0,
        "strike": 1.0,
        "option_type": "C",
        "unit": "USD",
    }
    request.update(changes)
    with pytest.raises((TypeError, ValueError)):
        EuropeanOptionPayoff(**request)  # type: ignore[arg-type]


def test_european_evaluation_rejects_missing_expiry_asset_and_failed_paths() -> None:
    """Path selection is exact and never silently omits an invalid terminal row."""
    with pytest.raises(ValueError, match="expiry"):
        _call(expiry=0.5)(_paths([1.0, 2.0]))
    missing_asset = _paths([1.0, 2.0])
    missing_asset.asset_ids = ("other",)
    with pytest.raises(ValueError, match="asset"):
        _call()(missing_asset)
    with pytest.raises(ValueError, match="finite"):
        _call()(_paths([1.0, np.nan]))


def test_variance_payoffs_subtract_initial_state_and_annualize_exactly_once() -> None:
    """Integrated and annualized strikes consume the cumulative-state increment."""
    times = np.array([0.0, 0.5])
    qvar = np.array([[0.1, 0.3], [0.2, 0.7], [1.0, 1.2]])
    paths = _paths(
        [1.0, 1.0, 1.0],
        observation_times=times,
        states={"qvar": qvar},
        state_units={"qvar": "integrated variance"},
    )
    integrated = IntegratedVarianceOptionPayoff(
        state_name="qvar",
        expiry=0.5,
        strike=0.15,
        option_type="C",
        quote=VarianceQuote.INTEGRATED,
        unit="variance-years",
    )
    annualized = IntegratedVarianceOptionPayoff(
        state_name="qvar",
        expiry=0.5,
        strike=0.3,
        option_type=OptionType.CALL,
        quote="annualized",
        unit="variance",
    )
    integrated_put = IntegratedVarianceOptionPayoff(
        state_name="qvar",
        expiry=0.5,
        strike=0.15,
        option_type="P",
        quote=VarianceQuote.INTEGRATED,
        unit="variance-years",
    )

    assert isinstance(integrated, Payoff)
    assert integrated.required_asset_ids == annualized.required_asset_ids == ()
    assert integrated.quote is VarianceQuote.INTEGRATED
    assert annualized.quote is VarianceQuote.ANNUALIZED
    np.testing.assert_allclose(integrated(paths), [0.05, 0.35, 0.05], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(annualized(paths), [0.1, 0.7, 0.1], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        integrated(paths) - integrated_put(paths),
        qvar[:, -1] - qvar[:, 0] - 0.15,
        rtol=0.0,
        atol=1e-15,
    )
    assert not annualized(paths).flags.writeable


def test_variance_payoff_enforces_state_shape_unit_increment_and_quote() -> None:
    """A variance option cannot infer a state convention from its name or shape."""
    request = {
        "state_name": "qvar",
        "expiry": 1.0,
        "strike": 0.0,
        "option_type": "C",
        "quote": VarianceQuote.INTEGRATED,
        "unit": "variance-years",
    }
    with pytest.raises((TypeError, ValueError), match="quote"):
        IntegratedVarianceOptionPayoff(**{**request, "quote": "per_day"})
    for changes in (
        {"state_name": " qvar"},
        {"expiry": 0.0},
        {"strike": -1.0},
        {"option_type": "IC"},
        {"unit": ""},
    ):
        with pytest.raises((TypeError, ValueError)):
            IntegratedVarianceOptionPayoff(**{**request, **changes})

    missing = _paths([1.0, 1.0])
    with pytest.raises(ValueError, match="state"):
        IntegratedVarianceOptionPayoff(**request)(missing)

    wrong_unit = _paths(
        [1.0, 1.0],
        states={"qvar": np.array([[0.0, 0.2], [0.0, 0.3]])},
        state_units={"qvar": "annualized variance"},
    )
    with pytest.raises(ValueError, match="integrated variance"):
        IntegratedVarianceOptionPayoff(**request)(wrong_unit)

    vector_state = _paths(
        [1.0, 1.0],
        states={"qvar": np.zeros((2, 2, 1))},
        state_units={"qvar": "integrated variance"},
    )
    with pytest.raises(ValueError, match="scalar"):
        IntegratedVarianceOptionPayoff(**request)(vector_state)

    decreasing = _paths(
        [1.0, 1.0],
        states={"qvar": np.array([[0.2, 0.1], [0.0, 0.3]])},
        state_units={"qvar": "integrated variance"},
    )
    with pytest.raises(ValueError, match="non-decreasing"):
        IntegratedVarianceOptionPayoff(**request)(decreasing)

    times = np.array([0.0, 0.5, 1.0])
    for invalid_state in (
        np.array([[0.0, np.nan, 0.3], [0.0, 0.1, 0.2]]),
        np.array([[0.0, 0.3, 0.2], [0.0, 0.1, 0.2]]),
        np.array([[-0.1, 0.0, 0.2], [0.0, 0.1, 0.2]]),
    ):
        invalid_paths = _paths(
            [1.0, 1.0],
            observation_times=times,
            states={"qvar": invalid_state},
            state_units={"qvar": "integrated variance"},
        )
        with pytest.raises(ValueError, match="integrated variance"):
            IntegratedVarianceOptionPayoff(**request)(invalid_paths)


def test_payoff_arithmetic_overflow_is_rejected() -> None:
    """Finite inputs cannot escape a public payoff as infinite float64 values."""
    maximum = np.finfo(np.float64).max
    put = EuropeanOptionPayoff(
        asset_id="asset",
        expiry=1.0,
        strike=maximum,
        option_type="P",
        unit="USD",
    )
    with pytest.raises(FloatingPointError, match="finite"):
        put(_paths([-maximum, 1.0]))

    expiry = float(np.nextafter(0.0, 1.0))
    variance_paths = _paths(
        [1.0, 1.0],
        observation_times=np.array([0.0, expiry]),
        states={"qvar": np.array([[0.0, maximum], [0.0, 1.0]])},
        state_units={"qvar": "integrated variance"},
    )
    annualized = IntegratedVarianceOptionPayoff(
        state_name="qvar",
        expiry=expiry,
        strike=0.0,
        option_type="C",
        quote=VarianceQuote.ANNUALIZED,
        unit="variance",
    )
    with pytest.raises(ValueError, match="finite"):
        annualized(variance_paths)


def test_hand_computed_direct_raw_and_self_normalized_estimators() -> None:
    """Fix point estimates, modern standard errors and ESS without model numerics."""
    values = np.array([1.0, 2.0, 4.0])
    direct = value_paths(paths=_paths(values), payoff=_call())
    log_weights = np.log(np.array([0.5, 1.0, 2.0]))
    weighted_paths = _paths(
        values,
        sampling_measure="P",
        target_measure="Q",
        log_likelihood_ratios=log_weights,
    )
    raw = value_paths(paths=weighted_paths, payoff=_call())
    normalized = value_paths_self_normalized(paths=weighted_paths, payoff=_call())

    assert direct.estimator is PathEstimator.MONTE_CARLO
    assert direct.value == pytest.approx(7.0 / 3.0)
    assert direct.standard_error == pytest.approx(math.sqrt(7.0) / 3.0)
    assert direct.standard_error_basis == "iid_paths"
    assert direct.n_paths == direct.n_independent_groups == 3
    assert direct.group_size == 1
    assert direct.path_effective_sample_size == 3.0
    assert direct.group_effective_sample_size == 3.0
    assert direct.path_ess_fraction == direct.group_ess_fraction == 1.0
    assert direct.mean_likelihood_ratio is None
    assert direct.log_mean_likelihood_ratio is None
    assert direct.recenter_shift is None
    assert direct.settlement_unit == "USD"

    assert raw.estimator is PathEstimator.RAW_LIKELIHOOD_RATIO
    assert raw.value == pytest.approx(3.5)
    assert raw.standard_error == pytest.approx(math.sqrt(21.0) / 2.0)
    assert normalized.estimator is PathEstimator.SELF_NORMALIZED_LIKELIHOOD_RATIO
    assert normalized.value == pytest.approx(3.0)
    assert normalized.standard_error == pytest.approx(6.0 / 7.0)
    for result in (raw, normalized):
        assert result.path_effective_sample_size == pytest.approx(7.0 / 3.0)
        assert result.path_ess_fraction == pytest.approx(7.0 / 9.0)
        assert result.group_effective_sample_size == pytest.approx(7.0 / 3.0)
        assert result.group_ess_fraction == pytest.approx(7.0 / 9.0)
        assert result.mean_likelihood_ratio == pytest.approx(7.0 / 6.0)
        assert result.log_mean_likelihood_ratio == pytest.approx(math.log(7.0 / 6.0))
    with pytest.raises(FrozenInstanceError):
        direct.value = 0.0  # type: ignore[misc]


def test_raw_scale_is_preserved_but_ratio_estimator_and_ess_are_scale_free() -> None:
    """Adding a log-weight constant cannot silently normalize the raw estimator."""
    values = np.array([1.0, 2.0, 4.0])
    logs = np.log(np.array([0.5, 1.0, 2.0]))
    base_paths = _paths(
        values,
        sampling_measure="P",
        target_measure="Q",
        log_likelihood_ratios=logs,
    )
    scaled_paths = _paths(
        values,
        sampling_measure="P",
        target_measure="Q",
        log_likelihood_ratios=logs + math.log(10.0),
    )
    raw = value_paths(paths=base_paths, payoff=_call())
    raw_scaled = value_paths(paths=scaled_paths, payoff=_call())
    normalized = value_paths_self_normalized(paths=base_paths, payoff=_call())
    normalized_scaled = value_paths_self_normalized(paths=scaled_paths, payoff=_call())

    assert raw_scaled.value == pytest.approx(10.0 * raw.value)
    assert raw_scaled.standard_error == pytest.approx(10.0 * raw.standard_error)
    assert raw_scaled.mean_likelihood_ratio == pytest.approx(10.0 * raw.mean_likelihood_ratio)
    assert raw_scaled.log_mean_likelihood_ratio == pytest.approx(
        raw.log_mean_likelihood_ratio + math.log(10.0)
    )
    assert normalized_scaled.value == pytest.approx(normalized.value)
    assert normalized_scaled.standard_error == pytest.approx(normalized.standard_error)
    assert normalized_scaled.path_effective_sample_size == pytest.approx(
        normalized.path_effective_sample_size
    )
    assert normalized_scaled.group_effective_sample_size == pytest.approx(
        normalized.group_effective_sample_size
    )


def test_weighted_payoff_scaling_preserves_finite_extreme_contributions() -> None:
    """A tiny shifted weight cannot erase a large but finite payoff contribution."""
    paths = _paths(
        [0.0, 1.0e300],
        sampling_measure="P",
        target_measure="Q",
        log_likelihood_ratios=np.array([1000.0, 0.0]),
    )
    raw = value_paths(paths=paths, payoff=_call())
    normalized = value_paths_self_normalized(paths=paths, payoff=_call())
    expected_normalized = math.exp(math.log(1.0e300) - 1000.0)

    assert raw.value == pytest.approx(5.0e299)
    assert raw.standard_error == pytest.approx(5.0e299)
    assert normalized.value == pytest.approx(expected_normalized, rel=2e-13)
    assert normalized.standard_error == pytest.approx(2.0 * expected_normalized, rel=2e-13)


def test_zero_weights_are_allowed_but_self_normalized_inference_needs_two_groups() -> None:
    """Zero-mass paths remain aligned while a one-group ratio SE is rejected."""
    paths = _paths(
        [1.0, 2.0, 4.0],
        sampling_measure="P",
        target_measure="Q",
        log_likelihood_ratios=np.array([0.0, -np.inf, 0.0]),
    )
    raw = value_paths(paths=paths, payoff=_call())
    normalized = value_paths_self_normalized(paths=paths, payoff=_call())
    assert raw.value == pytest.approx(5.0 / 3.0)
    assert normalized.value == pytest.approx(2.5)
    assert normalized.path_effective_sample_size == pytest.approx(2.0)

    degenerate = _paths(
        [1.0, 2.0, 4.0],
        sampling_measure="P",
        target_measure="Q",
        log_likelihood_ratios=np.array([0.0, -np.inf, -np.inf]),
    )
    with pytest.raises(ValueError, match="positive.*group"):
        value_paths_self_normalized(paths=degenerate, payoff=_call())


def test_explicit_independent_groups_determine_the_standard_error() -> None:
    """Antithetic structure comes only from caller IDs, never opaque provenance."""
    paths = _paths([1.0, 3.0, 3.0, 1.0])
    ungrouped = value_paths(paths=paths, payoff=_call())
    grouped = value_paths(
        paths=paths,
        payoff=_call(),
        independent_group_ids=np.array([0, 1, 0, 1]),
    )

    assert ungrouped.standard_error > 0.0
    assert grouped.value == 2.0
    assert grouped.standard_error == 0.0
    assert grouped.standard_error_basis == "independent_groups"
    assert grouped.n_paths == 4
    assert grouped.n_independent_groups == 2
    assert grouped.group_size == 2
    assert grouped.path_effective_sample_size == 4.0
    assert grouped.group_effective_sample_size == 2.0


def test_grouped_raw_and_self_normalized_inference_use_group_weight_residuals() -> None:
    """Fix grouped LR point estimates, ratio influence SE, and both ESS definitions."""
    paths = _paths(
        [1.0, 3.0, 2.0, 4.0],
        sampling_measure="P",
        target_measure="Q",
        log_likelihood_ratios=np.log(np.array([1.0, 2.0, 3.0, 4.0])),
    )
    group_ids = np.array([0, 1, 0, 1])
    raw = value_paths(paths=paths, payoff=_call(), independent_group_ids=group_ids)
    normalized = value_paths_self_normalized(
        paths=paths,
        payoff=_call(),
        independent_group_ids=group_ids,
    )

    assert raw.value == pytest.approx(7.25)
    assert raw.standard_error == pytest.approx(3.75)
    assert normalized.value == pytest.approx(2.9)
    assert normalized.standard_error == pytest.approx(0.92)
    assert normalized.path_effective_sample_size == pytest.approx(10.0 / 3.0)
    assert normalized.group_effective_sample_size == pytest.approx(25.0 / 13.0)
    assert normalized.n_independent_groups == 2
    assert normalized.group_size == 2


@pytest.mark.parametrize(
    "group_ids",
    [
        [0, 1, 0, 1],
        np.array([0.0, 1.0, 0.0, 1.0]),
        np.array([False, True, False, True]),
        np.array([0, 1]),
        np.array([[0, 1], [0, 1]]),
        np.array([0, 0, 0, 0]),
        np.array([0, 0, 0, 1]),
    ],
)
def test_malformed_independent_groups_are_rejected(group_ids: object) -> None:
    """Groups must be an equal-size integer partition with at least two members."""
    with pytest.raises((TypeError, ValueError)):
        value_paths(
            paths=_paths([1.0, 3.0, 3.0, 1.0]),
            payoff=_call(),
            independent_group_ids=group_ids,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_explicit_unweighted_recenter_matches_legacy_additive_values(
    option_type: OptionType,
) -> None:
    """Legacy compatibility shifts the asset before C/P payoff and keeps ddof=1 inference."""
    forward = 2.0
    terminal_spots = np.array([1.0, 2.0, 4.0])
    strike = 1.5
    discount = 0.9
    paths = _paths(terminal_spots)
    payoff = EuropeanOptionPayoff(
        asset_id="asset",
        expiry=1.0,
        strike=strike,
        option_type=option_type,
        unit="USD",
    )
    recenter = LegacyAdditiveForwardRecenter(
        target_forward=forward,
        mean=RecenterMean.UNWEIGHTED,
    )
    result = value_paths(
        paths=paths,
        payoff=payoff,
        discount_factor=discount,
        recenter=recenter,
    )
    raw = value_paths(paths=paths, payoff=payoff, discount_factor=discount)
    legacy_price, legacy_se = compute_mc_vars_payoff.py_func(
        x0=np.log(terminal_spots / forward),
        sigma0=np.ones(terminal_spots.size),
        qvar0=np.zeros(terminal_spots.size),
        ttm=1.0,
        forward=forward,
        strikes_ttm=np.array([strike]),
        optiontypes_ttm=np.array([option_type.value]),
        discfactor=discount,
    )
    shifted = terminal_spots + forward - np.mean(terminal_spots)
    if option_type is OptionType.CALL:
        adjusted_payoff = np.maximum(shifted - strike, 0.0)
        raw_payoff = np.maximum(terminal_spots - strike, 0.0)
    else:
        adjusted_payoff = np.maximum(strike - shifted, 0.0)
        raw_payoff = np.maximum(strike - terminal_spots, 0.0)

    assert result.value == pytest.approx(legacy_price[0], rel=0.0, abs=1e-15)
    assert result.value == pytest.approx(discount * np.mean(adjusted_payoff))
    assert raw.value == pytest.approx(discount * np.mean(raw_payoff))
    assert result.recenter_shift == pytest.approx(-1.0 / 3.0)
    assert result.standard_error == pytest.approx(
        discount * np.std(adjusted_payoff, ddof=1) / math.sqrt(adjusted_payoff.size)
    )
    assert result.standard_error != pytest.approx(legacy_se[0])
    assert result.standard_error_basis == "legacy_conditional_iid_paths"


def test_normalized_weighted_recenter_is_explicit_and_self_normalized_only() -> None:
    """The weighted shift pins the SN asset mean without claiming a raw-LR target."""
    terminal_spots = np.array([1.0, 2.0, 10.0])
    logs = np.log(np.array([8.0, 1.0, 1.0]))
    paths = _paths(
        terminal_spots,
        sampling_measure="P",
        target_measure="Q",
        log_likelihood_ratios=logs,
    )
    recenter = LegacyAdditiveForwardRecenter(
        target_forward=3.0,
        mean=RecenterMean.NORMALIZED_LIKELIHOOD_WEIGHTED,
    )
    result = value_paths_self_normalized(paths=paths, payoff=_call(), recenter=recenter)
    scaled_paths = _paths(
        terminal_spots,
        sampling_measure="P",
        target_measure="Q",
        log_likelihood_ratios=logs + math.log(10.0),
    )
    scaled = value_paths_self_normalized(
        paths=scaled_paths,
        payoff=_call(),
        recenter=recenter,
    )

    assert result.recenter_shift == pytest.approx(1.0)
    assert result.value == pytest.approx(3.0)
    assert result.standard_error_basis == "legacy_conditional_iid_paths"
    assert scaled.recenter_shift == pytest.approx(result.recenter_shift)
    assert scaled.value == pytest.approx(result.value)
    assert scaled.standard_error == pytest.approx(result.standard_error)
    with pytest.raises(ValueError, match="self-normalized"):
        value_paths(paths=paths, payoff=_call(), recenter=recenter)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"target_forward": 0.0}, "target_forward"),
        ({"target_forward": True}, "target_forward"),
        ({"target_forward": np.inf}, "target_forward"),
        ({"mean": "weighted"}, "mean"),
    ],
)
def test_legacy_recenter_policy_rejects_invalid_conventions(
    changes: dict[str, object],
    message: str,
) -> None:
    """Legacy compatibility requires a positive forward and named mean basis."""
    request: dict[str, object] = {
        "target_forward": 1.0,
        "mean": RecenterMean.UNWEIGHTED,
    }
    request.update(changes)
    with pytest.raises(ValueError, match=message):
        LegacyAdditiveForwardRecenter(**request)  # type: ignore[arg-type]


def test_measure_numeraire_weight_payoff_and_recenter_failures_are_explicit() -> None:
    """Invalid economic declarations fail instead of changing estimators or dropping rows."""
    payoff = _call()
    invalid_requests = [
        (_paths([1.0, 2.0], sampling_measure="P"), payoff, {}),
        (
            _paths([1.0, 2.0], sampling_measure="P", target_measure="Q"),
            payoff,
            {},
        ),
        (
            _paths(
                [1.0, 2.0],
                log_likelihood_ratios=np.zeros(2),
            ),
            payoff,
            {},
        ),
        (_paths([1.0, 2.0], numeraire="spot"), payoff, {}),
        (
            _paths(
                [1.0, 2.0],
                sampling_measure="P",
                target_measure="Q",
                log_likelihood_ratios=np.full(2, -np.inf),
            ),
            payoff,
            {},
        ),
        (_paths([1.0, 2.0]), _FixedPayoff(np.array([1.0, np.nan])), {}),
        (
            _paths([1.0, 2.0]),
            _FixedPayoff(np.array([1.0, 2.0]), settlement_unit="EUR"),
            {},
        ),
        (_paths([1.0, 2.0]), _FixedPayoff([1.0, 2.0]), {}),
        (_paths([1.0, 2.0]), payoff, {"discount_factor": 0.0}),
    ]
    for paths, candidate_payoff, keywords in invalid_requests:
        with pytest.raises((TypeError, ValueError)):
            value_paths(paths=paths, payoff=candidate_payoff, **keywords)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="self-normalized"):
        value_paths_self_normalized(paths=_paths([1.0, 2.0]), payoff=payoff)
    with pytest.raises(ValueError, match="likelihood"):
        value_paths(
            paths=_paths([1.0, 2.0]),
            payoff=payoff,
            recenter=LegacyAdditiveForwardRecenter(
                target_forward=1.0,
                mean=RecenterMean.NORMALIZED_LIKELIHOOD_WEIGHTED,
            ),
        )


def test_direct_modules_do_not_expand_the_package_root() -> None:
    """Keep the provisional D2 API on direct public submodules."""
    for name in (
        "EuropeanOptionPayoff",
        "IntegratedVarianceOptionPayoff",
        "VarianceQuote",
        "PathValuationResult",
        "value_paths",
    ):
        assert name not in stochvolmodels.__all__
        assert not hasattr(stochvolmodels, name)


def test_direct_module_imports_stay_outside_heavy_pricing_layers() -> None:
    """The bounded product/valuation import loads no pricer or optional analytic stack."""
    source_root = str(PACKAGE_ROOT.parent)
    code = f"""
import sys
sys.path.insert(0, {source_root!r})
import stochvolmodels.products.payoffs
import stochvolmodels.valuation
forbidden = (
    "numba",
    "pandas",
    "scipy",
    "stochvolmodels.pricers",
    "vanilla_option_pricers",
)
loaded = sorted(
    root
    for root in forbidden
    if any(name == root or name.startswith(root + ".") for name in sys.modules)
)
assert not loaded, loaded
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.paper_replication
def test_logsv_raw_european_and_variance_values_match_independent_analytics() -> None:
    """Raw MMA paths reproduce transform prices and the analytic variance moment within four SE."""
    from stochvolmodels.data.option_chain import OptionChain
    from stochvolmodels.models.logsv import LogSvMeasure, LogSvModel
    from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
    from stochvolmodels.pricers.logsv.vol_moments_ode import compute_analytic_qvar
    from stochvolmodels.pricers.logsv_pricer import LogSVPricer

    maturity = 0.25
    discount = 0.98
    params = LogSvParams(
        sigma0=0.2,
        theta=0.22,
        kappa1=3.0,
        kappa2=12.0,
        beta=-0.3,
        volvol=0.4,
        H=0.5,
    )
    paths = LogSvModel(params).simulate_paths(
        measure=LogSvMeasure.MMA,
        observation_times=np.array([0.0, maturity]),
        spot0=1.0,
        n_paths=40_000,
        steps_per_year=360,
        seed=20260829,
    )
    strikes = np.array([0.9, 1.0, 1.1])
    option_types = np.array(["P", "C", "C"])
    chain = OptionChain.slice_to_chain(
        ttm=maturity,
        forward=1.0,
        strikes=strikes,
        optiontypes=option_types,
        discfactor=discount,
        id="3m",
    )
    analytic_prices = np.asarray(LogSVPricer().price_chain(chain, params)[0])

    for strike, option_type, analytic_price in zip(strikes, option_types, analytic_prices):
        payoff = EuropeanOptionPayoff(
            asset_id="zero_drift_price",
            expiry=maturity,
            strike=float(strike),
            option_type=str(option_type),
            unit="forward units",
        )
        result = value_paths(paths=paths, payoff=payoff, discount_factor=discount)
        assert result.estimator is PathEstimator.MONTE_CARLO
        assert result.recenter_shift is None
        assert abs(result.value - analytic_price) <= 4.0 * result.standard_error

    variance_payoff = IntegratedVarianceOptionPayoff(
        state_name="quadratic_variance",
        expiry=maturity,
        strike=0.0,
        option_type="C",
        quote=VarianceQuote.ANNUALIZED,
        unit="annualized variance",
    )
    variance_result = value_paths(paths=paths, payoff=variance_payoff)
    expected_variance = compute_analytic_qvar(params, ttm=maturity, n_terms=8)
    assert abs(variance_result.value - expected_variance) <= 4.0 * variance_result.standard_error


@pytest.mark.slow
def test_tgarch_direct_q_and_raw_weighted_p_agree_with_antithetic_groups() -> None:
    """The new raw LR boundary preserves the discrete-model pricing-measure comparison."""
    from stochvolmodels.models.tgarch import TgarchMeasure, TgarchModel, TgarchParams

    maturity = 0.25
    n_paths = 65_536
    params = TgarchParams(
        theta=0.6,
        kappa1=3.0,
        kappa2=3.0,
        beta=1.0,
        eps=1.5,
        sigma0=0.6,
        r=0.0,
        gamma0=0.0,
        gamma1=0.5,
        eta0=0.0,
        eta1=-0.38,
        spot0=1.0,
    )
    model = TgarchModel(params)
    request = {
        "maturity": maturity,
        "max_dt": 1.0 / 52.0,
        "n_paths": n_paths,
        "seed": 20260825,
        "chunk_steps": 8,
    }
    p_paths = model.simulate_paths(
        measure=TgarchMeasure.P,
        track_log_weights=True,
        **request,
    )
    q_paths = model.simulate_paths(
        measure=TgarchMeasure.Q_EXACT,
        **{**request, "seed": 20260826},
    )
    pair_count = n_paths // 2
    pair_ids = np.concatenate([np.arange(pair_count), np.arange(pair_count)])
    payoff = EuropeanOptionPayoff(
        asset_id="spot",
        expiry=maturity,
        strike=params.spot0,
        option_type="C",
        unit="spot units",
    )
    weighted_p = value_paths(
        paths=p_paths,
        payoff=payoff,
        discount_factor=math.exp(-params.r * maturity),
        independent_group_ids=pair_ids,
    )
    direct_q = value_paths(
        paths=q_paths,
        payoff=payoff,
        discount_factor=math.exp(-params.r * maturity),
        independent_group_ids=pair_ids,
    )
    combined_error = math.hypot(weighted_p.standard_error, direct_q.standard_error)

    assert weighted_p.estimator is PathEstimator.RAW_LIKELIHOOD_RATIO
    assert direct_q.estimator is PathEstimator.MONTE_CARLO
    assert weighted_p.group_size == direct_q.group_size == 2
    assert abs(weighted_p.value - direct_q.value) <= 4.0 * combined_error
