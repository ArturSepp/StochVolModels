"""Contracts for the continuous LogSV path adapter."""

from __future__ import annotations

import math

import numpy as np
import pytest

import stochvolmodels
import stochvolmodels.models as model_capabilities
from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models import (
    PathModel,
    TerminalDistributionModel,
    TerminalSmileModel,
    TransformModel,
)
from stochvolmodels.models.logsv import LogSvMeasure, LogSvModel
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv.vol_moments_ode import (
    compute_analytic_qvar,
    compute_expected_vol_t,
)
from stochvolmodels.pricers.logsv_pricer import (
    LogSVPricer,
    get_randoms_for_chain_valuation,
    simulate_logsv_x_vol_terminal,
)


LEGACY_SOURCE_SHA256 = "50f8b31efae2c3c714e981ced0b3b30d92cb33cb8290702e323a1433a611b40f"
FROZEN_MMA_TERMINAL = {
    "log_zero_drift_return": np.array(
        [
            -0.09448133673790876,
            0.00039262963071787,
            -0.08346602986851795,
            0.10172383272398701,
            0.21066873623061724,
            -0.22088927985083490,
        ]
    ),
    "sigma": np.array(
        [
            0.25775861842505790,
            0.24595365743917758,
            0.25163995708791365,
            0.18084538211707090,
            0.21673776243277626,
            0.28810890120076976,
        ]
    ),
    "quadratic_variance": np.array(
        [
            0.01273183140647660,
            0.01174100124681368,
            0.01054472743454223,
            0.00650508057190608,
            0.00826029937998708,
            0.01364856292167067,
        ]
    ),
}


def _params(**overrides: object) -> LogSvParams:
    """Return the bounded continuous-LogSV fixture used by the adapter audit."""
    values: dict[str, object] = {
        "sigma0": 0.2,
        "theta": 0.22,
        "kappa1": 3.0,
        "kappa2": 12.0,
        "beta": -0.3,
        "volvol": 0.4,
        "H": 0.5,
    }
    values.update(overrides)
    return LogSvParams(**values)


def _simulate(
    measure: LogSvMeasure | str,
    *,
    observation_times: np.ndarray | None = None,
    spot0: float = 1.0,
    n_paths: int = 6,
    steps_per_year: int = 12,
    seed: int = 20260828,
    params: LogSvParams | None = None,
):
    """Simulate the standard audited request through the public submodule boundary."""
    if observation_times is None:
        observation_times = np.array([0.0, 0.25])
    return LogSvModel(params or _params()).simulate_paths(
        measure=measure,
        observation_times=observation_times,
        spot0=spot0,
        n_paths=n_paths,
        steps_per_year=steps_per_year,
        seed=seed,
    )


def _legacy_full_grid(
    measure: LogSvMeasure,
    *,
    params: LogSvParams,
    observation_times: np.ndarray,
    spot0: float,
    n_paths: int,
    steps_per_year: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run the existing terminal kernel sequentially without the new adapter."""
    w0s, w1s, dts = get_randoms_for_chain_valuation(
        ttms=observation_times[1:],
        nb_path=n_paths,
        nb_steps_per_year=steps_per_year,
        seed=seed,
    )
    x = np.zeros(n_paths)
    sigma = np.full(n_paths, params.sigma0)
    qvar = np.zeros(n_paths)
    x_grid = np.zeros((n_paths, observation_times.size))
    sigma_grid = np.full((n_paths, observation_times.size), params.sigma0)
    qvar_grid = np.zeros((n_paths, observation_times.size))

    for index, (target, w0, w1, dt) in enumerate(
        zip(observation_times[1:], w0s, w1s, dts),
        start=1,
    ):
        interval = float(target - observation_times[index - 1])
        x, sigma, qvar = simulate_logsv_x_vol_terminal(
            ttm=interval,
            x0=x,
            sigma0=sigma,
            qvar0=qvar,
            theta=params.theta,
            kappa1=params.kappa1,
            kappa2=params.kappa2,
            beta=params.beta,
            volvol=params.volvol,
            vol_backbone_eta=1.0,
            is_spot_measure=measure is LogSvMeasure.MMA,
            nb_path=n_paths,
            W0=w0,
            W1=w1,
            dt=dt,
        )
        x_grid[:, index] = x
        sigma_grid[:, index] = sigma
        qvar_grid[:, index] = qvar

    states = {
        "log_zero_drift_return": x_grid,
        "sigma": sigma_grid,
        "quadratic_variance": qvar_grid,
    }
    return spot0 * np.exp(x_grid)[:, :, None], states


def _independent_numpy_recurrence(
    measure: LogSvMeasure,
    *,
    params: LogSvParams,
    observation_times: np.ndarray,
    spot0: float,
    n_paths: int,
    steps_per_year: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Reproduce the explicit scheme directly from its scalar recurrence."""
    rng = np.random.RandomState(seed)
    x = np.zeros(n_paths)
    sigma = np.full(n_paths, params.sigma0)
    qvar = np.zeros(n_paths)
    x_grid = np.zeros((n_paths, observation_times.size))
    sigma_grid = np.full((n_paths, observation_times.size), params.sigma0)
    qvar_grid = np.zeros((n_paths, observation_times.size))
    alpha = -1.0 if measure is LogSvMeasure.MMA else 1.0
    adjustment = 0.0 if measure is LogSvMeasure.MMA else params.beta
    vartheta_squared = params.beta * params.beta + params.volvol * params.volvol

    for interval_index, interval in enumerate(np.diff(observation_times), start=1):
        step_count = int(interval * steps_per_year) + 1
        dt = float(interval / step_count)
        sqrt_dt = math.sqrt(dt)
        w0_matrix = rng.normal(0.0, 1.0, size=(step_count, n_paths))
        w1_matrix = rng.normal(0.0, 1.0, size=(step_count, n_paths))
        log_sigma = np.log(sigma)

        for raw_w0, raw_w1 in zip(w0_matrix, w1_matrix):
            w0 = sqrt_dt * raw_w0
            w1 = sqrt_dt * raw_w1
            sigma_squared_dt = sigma * sigma * dt
            x = x + alpha * 0.5 * sigma_squared_dt + sigma * w0
            log_sigma = (
                log_sigma
                + (
                    (params.kappa1 * params.theta / sigma - params.kappa1)
                    + params.kappa2 * (params.theta - sigma)
                    + adjustment * sigma
                    - 0.5 * vartheta_squared
                )
                * dt
                + params.beta * w0
                + params.volvol * w1
            )
            sigma = np.exp(log_sigma)
            qvar = qvar + 0.5 * (sigma_squared_dt + sigma * sigma * dt)

        x_grid[:, interval_index] = x
        sigma_grid[:, interval_index] = sigma
        qvar_grid[:, interval_index] = qvar

    states = {
        "log_zero_drift_return": x_grid,
        "sigma": sigma_grid,
        "quadratic_variance": qvar_grid,
    }
    return spot0 * np.exp(x_grid)[:, :, None], states


def _assert_global_random_states_equal(left: tuple[object, ...], right: tuple[object, ...]) -> None:
    """Compare every field of NumPy's legacy process-global RNG state."""
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def _standard_error(values: np.ndarray) -> float:
    """Return the ordinary raw-sample standard error of a mean."""
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def test_logsv_path_adapter_public_submodule_imports() -> None:
    """Expose the adapter through its direct public submodule only."""
    assert LogSvMeasure.MMA.value == "Q_MMA"
    assert LogSvMeasure.INVERSE.value == "Q_INVERSE"
    assert LogSvModel.__module__ == "stochvolmodels.models.logsv"


def test_logsv_types_are_path_capability_only_and_do_not_expand_root_exports() -> None:
    """Keep pricing, transforms and terminal laws outside this thin dynamic adapter."""
    assert "LogSvModel" not in stochvolmodels.__all__
    assert not hasattr(stochvolmodels, "LogSvModel")
    assert not hasattr(model_capabilities, "LogSvModel")
    model = LogSvModel(_params())
    assert isinstance(model, PathModel)
    assert not isinstance(model, TransformModel)
    assert not isinstance(model, TerminalDistributionModel)
    assert not isinstance(model, TerminalSmileModel)


def test_model_snapshots_mutable_legacy_parameters_and_returns_detached_copies() -> None:
    """Caller and property mutations cannot change the model's six bound dynamics."""
    caller_params = _params()
    model = LogSvModel(caller_params)
    caller_params.sigma0 = 0.91
    caller_params.beta = 0.17

    first = model.params
    second = model.params
    assert first is not caller_params
    assert second is not first
    assert first.sigma0 == second.sigma0 == 0.2
    assert first.beta == second.beta == -0.3

    first.theta = 0.91
    assert model.params.theta == 0.22


@pytest.mark.parametrize(
    ("measure", "numeraire", "condition"),
    [
        (LogSvMeasure.MMA, "money_market_account", "kappa2 >= beta"),
        (LogSvMeasure.INVERSE, "spot", "kappa2 >= 2 * beta"),
    ],
)
def test_full_grid_payload_is_read_only_and_has_explicit_zero_drift_semantics(
    measure: LogSvMeasure,
    numeraire: str,
    condition: str,
) -> None:
    """Return all observations with explicit state, measure, scheme and diagnostics."""
    requested_times = np.array([0.0, 0.1, 0.25])
    paths = _simulate(
        measure,
        observation_times=requested_times,
        spot0=1.25,
        n_paths=7,
        steps_per_year=8,
        seed=17,
    )
    requested_times[-1] = 0.3

    np.testing.assert_array_equal(paths.observation_times, [0.0, 0.1, 0.25])
    assert paths.assets.shape == (7, 3, 1)
    assert paths.asset_ids == ("zero_drift_price",)
    assert set(paths.states) == {
        "log_zero_drift_return",
        "sigma",
        "quadratic_variance",
    }
    assert all(state.shape == (7, 3) for state in paths.states.values())
    assert paths.state_units == {
        "log_zero_drift_return": "log return",
        "sigma": "annualized volatility",
        "quadratic_variance": "integrated variance",
    }
    np.testing.assert_array_equal(paths.assets[:, 0, 0], 1.25)
    np.testing.assert_array_equal(paths.states["log_zero_drift_return"][:, 0], 0.0)
    np.testing.assert_array_equal(paths.states["sigma"][:, 0], 0.2)
    np.testing.assert_array_equal(paths.states["quadratic_variance"][:, 0], 0.0)
    np.testing.assert_allclose(
        paths.assets[:, :, 0],
        1.25 * np.exp(paths.states["log_zero_drift_return"]),
        rtol=0.0,
        atol=0.0,
    )
    assert np.all(np.diff(paths.states["quadratic_variance"], axis=1) >= 0.0)
    assert paths.sampling_measure == paths.target_measure == measure.value
    assert paths.numeraire == numeraire
    assert paths.scheme == "logsv_explicit_log_euler_trapezoidal_qvar_v1"
    assert paths.log_likelihood_ratios is None

    provenance = paths.provenance
    assert provenance["steps_per_year"] == 8
    assert provenance["interval_step_counts"] == (1, 2)
    assert provenance["interval_dts"] == pytest.approx((0.1, 0.075), rel=0.0, abs=1e-16)
    assert provenance["seed"] == 17
    assert provenance["generator"] == "numpy.random.RandomState"
    assert provenance["bit_generator"] == "MT19937"
    assert provenance["normal_convention"] == "unscaled_standard_normals_kernel_scales_sqrt_dt"
    assert provenance["draw_ordering"] == "per_interval_W0_matrix_then_W1_matrix_row_major"
    assert provenance["kernel"] == "simulate_logsv_x_vol_terminal"
    assert provenance["vol_backbone_eta"] == 1.0
    assert provenance["legacy_source_sha256"] == LEGACY_SOURCE_SHA256
    assert provenance["numpy_version"] == np.__version__
    assert provenance["observation_partition_dependent"] is True

    diagnostics = paths.diagnostics
    assert diagnostics["nonfinite_asset_count"] == 0
    assert diagnostics["nonfinite_log_return_count"] == 0
    assert diagnostics["nonfinite_sigma_count"] == 0
    assert diagnostics["nonfinite_qvar_count"] == 0
    assert diagnostics["asset_positive_infinity_count"] == 0
    assert diagnostics["asset_underflow_count"] == 0
    assert diagnostics["nonpositive_sigma_count"] == 0
    assert diagnostics["negative_qvar_count"] == 0
    assert diagnostics["qvar_decrease_count"] == 0
    assert diagnostics["failed_path_count"] == 0
    assert diagnostics["measure_martingale_condition"] == condition

    arrays = [paths.observation_times, paths.assets, *paths.states.values()]
    assert requested_times.flags.writeable
    for array in arrays:
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 99.0


def test_local_randomstate_replays_without_mutating_process_global_rng() -> None:
    """The adapter owns an MT19937 instance and never consumes NumPy's global stream."""
    np.random.seed(91827)
    global_before = np.random.get_state()
    first = _simulate(LogSvMeasure.MMA)
    global_after = np.random.get_state()
    second = _simulate("Q_MMA")
    global_final = np.random.get_state()

    _assert_global_random_states_equal(global_before, global_after)
    _assert_global_random_states_equal(global_before, global_final)
    np.testing.assert_array_equal(first.assets, second.assets)
    for name in first.states:
        np.testing.assert_array_equal(first.states[name], second.states[name])


@pytest.mark.parametrize("measure", list(LogSvMeasure))
def test_adapter_matches_direct_sequential_legacy_kernel_exactly(measure: LogSvMeasure) -> None:
    """Both measures carry each terminal state through the historical interval loop."""
    params = _params()
    times = np.array([0.0, 0.1, 0.25])
    request = {
        "params": params,
        "observation_times": times,
        "spot0": 1.3,
        "n_paths": 9,
        "steps_per_year": 8,
        "seed": 17,
    }
    paths = _simulate(measure, **request)
    expected_assets, expected_states = _legacy_full_grid(measure, **request)

    np.testing.assert_array_equal(paths.assets, expected_assets)
    for name, expected in expected_states.items():
        np.testing.assert_array_equal(paths.states[name], expected)


def test_frozen_mma_terminal_capture_preserves_seeded_legacy_numerics() -> None:
    """The audited six-path fixture fixes drift, scaling and trapezoidal-QV semantics."""
    paths = _simulate(LogSvMeasure.MMA)

    assert paths.provenance["interval_step_counts"] == (4,)
    assert paths.provenance["interval_dts"] == pytest.approx((0.0625,), rel=0.0, abs=0.0)
    for name, expected in FROZEN_MMA_TERMINAL.items():
        np.testing.assert_allclose(
            paths.states[name][:, -1],
            expected,
            rtol=0.0,
            atol=5.0e-15,
        )


@pytest.mark.parametrize("measure", list(LogSvMeasure))
def test_paths_match_independent_pure_numpy_recurrence(measure: LogSvMeasure) -> None:
    """A second implementation verifies Brownian scaling, drift, carry and QV update."""
    params = _params()
    times = np.array([0.0, 0.07, 0.25])
    request = {
        "params": params,
        "observation_times": times,
        "spot0": 0.87,
        "n_paths": 11,
        "steps_per_year": 13,
        "seed": 219,
    }
    paths = _simulate(measure, **request)
    expected_assets, expected_states = _independent_numpy_recurrence(measure, **request)

    np.testing.assert_allclose(paths.assets, expected_assets, rtol=0.0, atol=5.0e-15)
    for name, expected in expected_states.items():
        np.testing.assert_allclose(paths.states[name], expected, rtol=0.0, atol=5.0e-15)


def test_observation_partition_is_recorded_and_changes_seeded_terminal_paths() -> None:
    """The historical per-interval +1 step and draw ordering make partitions numerical."""
    unpartitioned = _simulate(
        LogSvMeasure.MMA,
        observation_times=np.array([0.0, 0.25]),
        n_paths=12,
        steps_per_year=8,
        seed=17,
    )
    partitioned = _simulate(
        LogSvMeasure.MMA,
        observation_times=np.array([0.0, 0.1, 0.25]),
        n_paths=12,
        steps_per_year=8,
        seed=17,
    )

    assert unpartitioned.provenance["interval_step_counts"] == (3,)
    assert partitioned.provenance["interval_step_counts"] == (1, 2)
    assert unpartitioned.provenance["interval_dts"] == pytest.approx((1.0 / 12.0,))
    assert partitioned.provenance["interval_dts"] == pytest.approx((0.1, 0.075))
    assert not np.array_equal(
        unpartitioned.assets[:, -1, 0],
        partitioned.assets[:, -1, 0],
    )


def test_constructor_rejects_non_logsv_parameter_payload() -> None:
    """The adapter does not infer dynamics from an arbitrary object."""
    with pytest.raises(ValueError, match="LogSvParams"):
        LogSvModel(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["sigma0", "theta", "kappa1", "kappa2", "beta", "volvol"])
@pytest.mark.parametrize("bad_value", [True, np.inf, 1.0 + 0.0j])
def test_constructor_rejects_boolean_nonfinite_and_nonreal_dynamics(
    field: str,
    bad_value: object,
) -> None:
    """All six scalar dynamics must be unambiguous finite real numbers."""
    params = _params()
    setattr(params, field, bad_value)
    with pytest.raises(ValueError, match=field):
        LogSvModel(params)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("sigma0", 0.0),
        ("theta", -0.1),
        ("kappa1", -0.1),
        ("kappa2", -0.1),
        ("volvol", -0.1),
    ],
)
def test_constructor_rejects_inadmissible_parameter_signs(field: str, bad_value: float) -> None:
    """Volatility levels are positive and rates or diffusion magnitudes non-negative."""
    params = _params()
    setattr(params, field, bad_value)
    with pytest.raises(ValueError, match=field):
        LogSvModel(params)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"H": 0.49}, "H"),
        ({"vol_backbone": object()}, "backbone"),
        ({"nodes": np.array([0.1])}, "nodes"),
        ({"weights": np.array([1.0])}, "weights"),
    ],
)
def test_constructor_rejects_rough_and_backbone_extensions(
    overrides: dict[str, object],
    message: str,
) -> None:
    """This adapter cannot silently route to rough or term-structured dynamics."""
    with pytest.raises(ValueError, match=message):
        LogSvModel(_params(**overrides))


def test_simulation_rejects_unknown_measure() -> None:
    """Only the two implemented pricing measures are accepted."""
    with pytest.raises(ValueError, match="measure must be one of"):
        _simulate("Q")


@pytest.mark.parametrize(
    "bad_times",
    [
        [0.0, 0.25],
        np.array([0.0, 0.25], dtype=complex),
        np.array([[0.0, 0.25]]),
        np.array([0.0]),
        np.array([0.01, 0.25]),
        np.array([0.0, 0.25, 0.25]),
        np.array([0.0, np.nan]),
    ],
)
def test_simulation_rejects_invalid_observation_grids(bad_times: object) -> None:
    """The numerical partition is an explicit finite increasing NumPy grid from zero."""
    with pytest.raises(ValueError, match="observation_times"):
        _simulate(LogSvMeasure.MMA, observation_times=bad_times)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("change", "bad_value"),
    [
        ("spot0", True),
        ("spot0", 0.0),
        ("spot0", np.inf),
        ("spot0", 1.0 + 0.0j),
        ("n_paths", True),
        ("n_paths", 0),
        ("n_paths", 2.5),
        ("steps_per_year", True),
        ("steps_per_year", 0),
        ("steps_per_year", 2.5),
        ("seed", True),
        ("seed", -1),
        ("seed", 2**32),
        ("seed", 1.5),
    ],
)
def test_simulation_rejects_invalid_scalar_requests(change: str, bad_value: object) -> None:
    """Spot, counts and MT19937 seeds reject ambiguous or inadmissible values."""
    request: dict[str, object] = {
        "measure": LogSvMeasure.MMA,
        "observation_times": np.array([0.0, 0.25]),
        "spot0": 1.0,
        "n_paths": 6,
        "steps_per_year": 12,
        "seed": 20260828,
    }
    request[change] = bad_value
    with pytest.raises(ValueError, match=change):
        LogSvModel(_params()).simulate_paths(**request)


def test_measure_specific_true_martingale_conditions_are_enforced() -> None:
    """Book-valuation paths reject local martingales outside the published conditions."""
    with pytest.raises(ValueError, match="kappa2 >= beta"):
        _simulate(LogSvMeasure.MMA, params=_params(kappa2=0.1, beta=0.2))

    inverse_only_failure = _params(kappa2=0.3, beta=0.2)
    _simulate(LogSvMeasure.MMA, params=inverse_only_failure)
    with pytest.raises(ValueError, match=r"kappa2 >= 2 \* beta"):
        _simulate(LogSvMeasure.INVERSE, params=inverse_only_failure)


@pytest.mark.paper_replication
def test_raw_paths_match_analytic_prices_martingales_and_moments_within_four_se() -> None:
    """Independent raw expectations validate both measures without forward recentering."""
    maturity = 0.25
    params = _params()
    chain = OptionChain.slice_to_chain(
        ttm=maturity,
        forward=1.0,
        strikes=np.array([0.9, 1.0, 1.1]),
        optiontypes=np.array(["P", "C", "C"]),
        discfactor=0.98,
        id="3m",
    )
    n_paths = 40_000
    mma_paths = _simulate(
        LogSvMeasure.MMA,
        observation_times=np.array([0.0, maturity]),
        n_paths=n_paths,
        steps_per_year=360,
        seed=20260829,
        params=params,
    )
    terminal_asset = mma_paths.assets[:, -1, 0]
    analytic_prices = np.asarray(LogSVPricer().price_chain(chain, params)[0])
    raw_payoffs = np.column_stack(
        [
            0.98 * np.maximum(0.9 - terminal_asset, 0.0),
            0.98 * np.maximum(terminal_asset - 1.0, 0.0),
            0.98 * np.maximum(terminal_asset - 1.1, 0.0),
        ]
    )
    mc_prices = np.mean(raw_payoffs, axis=0)
    mc_errors = np.std(raw_payoffs, axis=0, ddof=1) / math.sqrt(n_paths)
    assert np.all(np.abs(analytic_prices - mc_prices) <= 4.0 * mc_errors)

    assert abs(float(np.mean(terminal_asset)) - 1.0) <= 4.0 * _standard_error(terminal_asset)
    terminal_sigma = mma_paths.states["sigma"][:, -1]
    expected_sigma = compute_expected_vol_t(params, np.array([maturity]), n_terms=8)[0]
    assert abs(float(np.mean(terminal_sigma)) - expected_sigma) <= 4.0 * _standard_error(
        terminal_sigma
    )
    qvar_rate = mma_paths.states["quadratic_variance"][:, -1] / maturity
    expected_qvar_rate = compute_analytic_qvar(params, ttm=maturity, n_terms=8)
    assert abs(float(np.mean(qvar_rate)) - expected_qvar_rate) <= 4.0 * _standard_error(
        qvar_rate
    )

    inverse_paths = _simulate(
        LogSvMeasure.INVERSE,
        observation_times=np.array([0.0, maturity]),
        n_paths=n_paths,
        steps_per_year=360,
        seed=20260829,
        params=params,
    )
    inverse_zero_drift_price = 1.0 / inverse_paths.assets[:, -1, 0]
    assert abs(float(np.mean(inverse_zero_drift_price)) - 1.0) <= 4.0 * _standard_error(
        inverse_zero_drift_price
    )
