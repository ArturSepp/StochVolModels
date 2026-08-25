"""Characterization of the two-state equilibrium regime-switching LogSV pipeline."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from numba.typed import List
from scipy.linalg import expm

import stochvolmodels.models.regime_logsv as regime_model
from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.pricers.logsv import affine_expansion as scalar_afe
from stochvolmodels.models.regime_logsv import (
    CrraRiskPremia,
    EquilibriumClosure,
    Regime,
    RegimeLogSvDynamics,
    RegimeSwitchLogSvParams,
    RegimeTransition,
    _equilibrium_rhs,
    evaluate_risk_neutral_state,
    solve_regime_switch_equilibrium,
)
from stochvolmodels.models.regime_logsv_simulation import (
    simulate_regime_switch_logsv_terminal,
)
from stochvolmodels.pricers.regime_switch_logsv_pricer import (
    RegimeSwitchLogSVPricer,
    _option_rhs,
    compute_regime_switch_log_mgf_grid,
)
from stochvolmodels.utils.mc_payoffs import compute_mc_vars_payoff


def _state_chain() -> OptionChain:
    ttms = np.array([0.10, 0.25])
    strikes = np.array([0.90, 1.00, 1.10])
    return OptionChain(
        ttms=ttms,
        forwards=np.ones_like(ttms),
        discfactors=np.exp(-0.01 * ttms),
        strikes_ttms=List(strikes.copy() for _ in ttms),
        optiontypes_ttms=List(np.array(["P", "C", "C"]) for _ in ttms),
        ids=np.array(["1m", "3m"]),
    )


def _paired_chain() -> OptionChain:
    strikes = np.array([0.90, 1.00, 1.10])
    return OptionChain.slice_to_chain(
        ttm=0.25,
        forward=1.0,
        strikes=np.concatenate((strikes, strikes)),
        optiontypes=np.array(["C", "C", "C", "P", "P", "P"]),
        discfactor=0.98,
        id="3m",
    )


def _equity_params(
    closure: EquilibriumClosure = EquilibriumClosure.LOG_LINEAR,
    *,
    utility_power: float = -0.5,
    agent_horizon: float = 3.0,
    initial_regime: Regime = Regime.GROWTH,
) -> RegimeSwitchLogSvParams:
    common = dict(kappa1=2.6949, kappa2=10.0107, beta=-1.5082, volvol=0.8503)
    return RegimeSwitchLogSvParams(
        sigma0=0.15,
        regimes=(
            RegimeLogSvDynamics(theta=0.15, **common),
            RegimeLogSvDynamics(theta=0.225, **common),
        ),
        transitions=(
            RegimeTransition(intensity=0.10, mean_log_jump=-0.25 / 0.75),
            RegimeTransition(intensity=1.00, mean_log_jump=0.15 / 1.15),
        ),
        risk_premia=CrraRiskPremia(
            utility_power=utility_power,
            agent_horizon=agent_horizon,
            closure=closure,
        ),
        initial_regime=initial_regime,
    )


def test_regime_switch_params_encode_paper_jumps_and_copy_nested_specs() -> None:
    params = _equity_params()
    copied = RegimeSwitchLogSvParams.copy(params)

    assert copied is not params
    assert copied.regimes == params.regimes
    assert isinstance(copied.regimes[0], RegimeLogSvDynamics)
    assert isinstance(copied.risk_premia, CrraRiskPremia)
    assert copied.risk_premia.relative_risk_aversion == pytest.approx(1.5)
    observed = [transition.arithmetic_jump_mean for transition in params.transitions]
    np.testing.assert_allclose(observed, [-0.25, 0.15], rtol=0.0, atol=1.0e-15)
    assert [transition.intensity for transition in params.transitions] == [0.1, 1.0]
    with pytest.raises(FrozenInstanceError):
        params.risk_premia = CrraRiskPremia(0.0, 3.0)

    with pytest.raises(ValueError, match="initial_regime"):
        replace(params, initial_regime=True)
    with pytest.raises(ValueError, match="initial_regime"):
        params.with_initial_regime(1.0)

    with pytest.raises(ValueError, match="growth-to-stress"):
        replace(
            params,
            transitions=(
                RegimeTransition(0.1, 0.01),
                params.transitions[Regime.STRESS],
            ),
        )


def test_regime_risk_premia_scales_are_strict_frozen_diagnostics() -> None:
    scales = regime_model.RegimeRiskPremiaScales()

    assert scales.equity_brownian == 1.0
    assert scales.orthogonal_brownian == 1.0
    assert scales.timing == 1.0
    assert scales.tail == 1.0
    assert scales.is_full_equilibrium
    with pytest.raises(FrozenInstanceError):
        scales.tail = 0.0

    for name in ("equity_brownian", "orthogonal_brownian", "timing", "tail"):
        for invalid in (True, np.inf, np.nan, "1"):
            with pytest.raises(ValueError, match="finite"):
                replace(scales, **{name: invalid})

    unrestricted = regime_model.RegimeRiskPremiaScales(-1.0, 2.0, -3.0, 4.0)
    assert not unrestricted.is_full_equilibrium


def test_full_regime_risk_premia_scales_preserve_default_transform_exactly() -> None:
    params = _equity_params()
    equilibrium = solve_regime_switch_equilibrium(params)
    phi_grid = np.array([0.0, -1.0, 0.2 + 0.7j, -0.5 + 1.3j])

    default = compute_regime_switch_log_mgf_grid(
        params,
        ttm=0.25,
        phi_grid=phi_grid,
        equilibrium=equilibrium,
    )
    explicit = compute_regime_switch_log_mgf_grid(
        params,
        ttm=0.25,
        phi_grid=phi_grid,
        equilibrium=equilibrium,
        scales=regime_model.RegimeRiskPremiaScales(),
    )

    np.testing.assert_array_equal(explicit, default)
    for regime in Regime:
        default_state = evaluate_risk_neutral_state(
            params,
            equilibrium,
            horizon=1.2,
            sigma=0.19,
            regime=regime,
        )
        explicit_state = evaluate_risk_neutral_state(
            params,
            equilibrium,
            horizon=1.2,
            sigma=0.19,
            regime=regime,
            scales=regime_model.RegimeRiskPremiaScales(),
        )
        assert explicit_state == default_state
    with pytest.raises(TypeError, match="RegimeRiskPremiaScales"):
        compute_regime_switch_log_mgf_grid(
            params,
            ttm=0.25,
            phi_grid=phi_grid,
            equilibrium=equilibrium,
            scales=None,
        )


def test_regime_risk_premia_scales_match_hand_derived_q_state_decomposition() -> None:
    params = _equity_params()
    equilibrium = solve_regime_switch_equilibrium(params)
    scales = regime_model.RegimeRiskPremiaScales(0.3, 0.7, 0.4, 0.2)
    horizon = 1.2
    sigma = 0.19
    regime = Regime.GROWTH
    dynamics = params.regimes[regime]
    loading = equilibrium.volatility_loading(horizon, sigma, regime)
    log_ratio = equilibrium.log_timing_ratio(horizon, sigma, regime)
    tail_tilt = scales.tail * (params.risk_premia.utility_power - 1.0)
    ell_tilt = params.jump_mgf(regime, tail_tilt)
    expected_drift = (dynamics.kappa1 + dynamics.kappa2 * sigma) * (dynamics.theta - sigma)
    expected_drift -= (
        scales.equity_brownian
        * dynamics.beta
        * params.risk_premia.relative_risk_aversion
        * sigma**2
    )
    expected_drift += (
        (
            scales.equity_brownian * dynamics.beta**2
            + scales.orthogonal_brownian * dynamics.volvol**2
        )
        * loading
        * sigma**2
    )
    expected_intensity = params.transitions[regime].intensity
    expected_intensity *= np.exp(scales.timing * log_ratio) * ell_tilt
    expected_jump_mean = params.transitions[regime].mean_log_jump
    expected_jump_mean /= 1.0 - expected_jump_mean * tail_tilt
    expected_arithmetic_mean = params.jump_mgf(regime, tail_tilt + 1.0) / ell_tilt - 1.0

    state = evaluate_risk_neutral_state(
        params,
        equilibrium,
        horizon,
        sigma,
        regime,
        scales=scales,
    )

    np.testing.assert_allclose(
        (
            state.volatility_drift,
            state.transition_intensity,
            state.mean_log_jump,
            state.arithmetic_jump_mean,
            state.volatility_loading,
            state.log_timing_ratio,
        ),
        (
            expected_drift,
            expected_intensity,
            expected_jump_mean,
            expected_arithmetic_mean,
            loading,
            log_ratio,
        ),
        rtol=0.0,
        atol=2.0e-15,
    )


def test_analytic_regime_risk_premia_channels_match_frozen_chapter_oracle() -> None:
    log_moneyness = np.linspace(-0.30, 0.20, 31)
    strikes = np.exp(log_moneyness)
    optiontypes = np.where(strikes < 1.0, "P", "C")
    chain = OptionChain.slice_to_chain(
        ttm=0.25,
        forward=1.0,
        strikes=strikes,
        optiontypes=optiontypes,
        id="3m",
    )
    params = _equity_params()
    equilibrium = solve_regime_switch_equilibrium(params)
    pricer = RegimeSwitchLogSVPricer()
    selected = np.array([0, 6, 12, 18, 24, 30])
    scenarios = (
        (
            regime_model.RegimeRiskPremiaScales(0.0, 0.0, 0.0, 0.0),
            [
                34.799145449632,
                28.762816043292,
                22.534731212228,
                16.242068479341,
                10.582902544890,
                13.592731914187,
            ],
            [
                -0.019344416231497248 - 0.004650502361071163j,
                -0.03751125270880886 + 0.0020582735401356088j,
            ],
        ),
        (
            regime_model.RegimeRiskPremiaScales(1.0, 1.0, 0.0, 0.0),
            [
                35.191913150100,
                29.271264409580,
                23.089002999124,
                16.712030832712,
                10.913737351373,
                13.605481975684,
            ],
            [
                -0.02038517727598558 - 0.0048845276238280505j,
                -0.03909450899480648 + 0.001638838458468522j,
            ],
        ),
        (
            regime_model.RegimeRiskPremiaScales(0.0, 0.0, 1.0, 0.0),
            [
                34.799446638392,
                28.763167873213,
                22.535123687118,
                16.242473551049,
                10.583168808267,
                13.592785371068,
            ],
            [
                -0.01934522939884087 - 0.004650684817072648j,
                -0.03751212788443771 + 0.002057660901379376j,
            ],
        ),
        (
            regime_model.RegimeRiskPremiaScales(0.0, 0.0, 0.0, 1.0),
            [
                47.176385634239,
                38.563317155283,
                29.472817366671,
                20.385313375699,
                12.446937125298,
                13.021095772689,
            ],
            [
                -0.03486054914022379 - 0.022361192892172848j,
                -0.031762131135938995 - 0.0015712614346930682j,
            ],
        ),
        (
            regime_model.RegimeRiskPremiaScales(),
            [
                47.359678403215,
                38.863951473492,
                29.906520899408,
                20.859939639867,
                12.828313462224,
                13.043665846078,
            ],
            [
                -0.035904056514211774 - 0.02259830761030641j,
                -0.033356228072906245 - 0.001990430276935223j,
            ],
        ),
    )

    for scales, expected_percent, expected_log_mgf in scenarios:
        conditional = pricer.compute_state_conditional_prices_with_vols(
            chain,
            params,
            equilibrium=equilibrium,
            scales=scales,
            max_phi=1_601,
        )
        _, implied_vols = conditional.for_regime(Regime.GROWTH)
        np.testing.assert_allclose(
            100.0 * np.asarray(implied_vols[0])[selected],
            expected_percent,
            rtol=1.0e-7,
            atol=5.0e-10,
        )
        roots = compute_regime_switch_log_mgf_grid(
            params,
            ttm=0.25,
            phi_grid=np.array([0.0, -1.0, -0.5 + 2.0j]),
            equilibrium=equilibrium,
            scales=scales,
        )
        np.testing.assert_allclose(roots[:, :2], 0.0, rtol=0.0, atol=2.0e-11)
        np.testing.assert_allclose(
            roots[:, 2],
            expected_log_mgf,
            rtol=1.0e-7,
            atol=5.0e-12,
        )


def test_equity_risk_premia_match_hand_computed_jump_and_timing_tilts() -> None:
    params = _equity_params()
    equilibrium = solve_regime_switch_equilibrium(params)

    np.testing.assert_allclose(
        [params.transition_factor(regime) for regime in Regime],
        [4.0 / 5.0, 2668.0 / 2695.0],
        rtol=0.0,
        atol=2.0e-15,
    )
    expected = (
        (0.2, -2.0 / 3.0, -2.0 / 5.0, 0.05090175),
        (46.0 / 55.0, 6.0 / 55.0, 6.0 / 49.0, 0.365639625),
    )
    for regime in Regime:
        state = evaluate_risk_neutral_state(
            params,
            equilibrium,
            horizon=0.0,
            sigma=params.sigma0,
            regime=regime,
        )
        np.testing.assert_allclose(
            (
                state.transition_intensity,
                state.mean_log_jump,
                state.arithmetic_jump_mean,
                state.volatility_drift,
            ),
            expected[regime],
            rtol=0.0,
            atol=2.0e-14,
        )


def test_log_linear_equilibrium_collector_matches_explicit_four_equations() -> None:
    params = _equity_params()
    coefficients = np.array([[0.021, -0.037], [-0.014, 0.029]])
    generic = _equilibrium_rhs(0.7, coefficients.ravel(), params, 1).reshape(2, 2)
    explicit = np.zeros_like(generic)
    utility_power = params.risk_premia.utility_power
    qvar_loading = 0.5 * utility_power * (1.0 - utility_power)

    for regime in Regime:
        other = Regime(1 - int(regime))
        dynamics = params.regimes[regime]
        delta = dynamics.theta - params.regimes[other].theta
        difference0 = coefficients[other, 0] + coefficients[other, 1] * delta
        difference0 -= coefficients[regime, 0]
        difference1 = coefficients[other, 1] - coefficients[regime, 1]
        loading = coefficients[regime, 1]
        transition = params.transitions[regime]
        coupling = transition.intensity * params.transition_factor(regime) * np.exp(difference0)
        explicit[regime, 0] = (
            0.5 * dynamics.vartheta2 * dynamics.theta**2 * loading**2
            + qvar_loading * dynamics.theta**2
            + coupling
            - transition.intensity
        )
        explicit[regime, 1] = (
            -dynamics.kappa_bar * loading
            + dynamics.vartheta2 * dynamics.theta * loading**2
            + 2.0 * qvar_loading * dynamics.theta
            + coupling * difference1
        )
    np.testing.assert_allclose(generic, explicit, rtol=0.0, atol=2.0e-14)


def test_log_quadratic_equilibrium_collector_matches_explicit_six_equations() -> None:
    params = _equity_params(EquilibriumClosure.LOG_QUADRATIC)
    coefficients = np.array([[0.021, -0.037, 0.016], [-0.014, 0.029, -0.011]])
    generic = _equilibrium_rhs(0.7, coefficients.ravel(), params, 2).reshape(2, 3)
    explicit = np.zeros_like(generic)
    utility_power = params.risk_premia.utility_power
    qvar_loading = 0.5 * utility_power * (1.0 - utility_power)

    for regime in Regime:
        other = Regime(1 - int(regime))
        dynamics = params.regimes[regime]
        delta = dynamics.theta - params.regimes[other].theta
        own0, own1, own2 = coefficients[regime]
        target0, target1, target2 = coefficients[other]
        difference0 = target0 + target1 * delta + target2 * delta**2 - own0
        difference1 = target1 + 2.0 * target2 * delta - own1
        difference2 = target2 - own2
        square_plus_second = own1**2 + 2.0 * own2
        coupling = (
            params.transitions[regime].intensity
            * params.transition_factor(regime)
            * np.exp(difference0)
        )
        explicit[regime, 0] = (
            0.5 * dynamics.vartheta2 * dynamics.theta**2 * square_plus_second
            + qvar_loading * dynamics.theta**2
            + coupling
            - params.transitions[regime].intensity
        )
        explicit[regime, 1] = (
            -dynamics.kappa_bar * own1
            + dynamics.vartheta2 * dynamics.theta * square_plus_second
            + 2.0 * dynamics.vartheta2 * dynamics.theta**2 * own1 * own2
            + 2.0 * qvar_loading * dynamics.theta
            + coupling * difference1
        )
        explicit[regime, 2] = (
            -2.0 * dynamics.kappa_bar * own2
            - dynamics.kappa2 * own1
            + 0.5 * dynamics.vartheta2 * square_plus_second
            + 4.0 * dynamics.vartheta2 * dynamics.theta * own1 * own2
            + 2.0 * dynamics.vartheta2 * dynamics.theta**2 * own2**2
            + qvar_loading
            + coupling * (difference2 + 0.5 * difference1**2)
        )
    np.testing.assert_allclose(generic, explicit, rtol=0.0, atol=3.0e-14)


def test_equilibrium_solver_rejects_outward_quadratic_and_cubic_q_drifts() -> None:
    outward_quadratic = RegimeLogSvDynamics(
        theta=0.2,
        kappa1=1.0,
        kappa2=0.0,
        beta=-1.0,
        volvol=0.0,
    )
    log_linear = RegimeSwitchLogSvParams(
        sigma0=0.2,
        regimes=(outward_quadratic, outward_quadratic),
        transitions=(RegimeTransition(0.0, 0.0), RegimeTransition(0.0, 0.0)),
        risk_premia=CrraRiskPremia(0.0, 1.0),
    )
    with pytest.raises(ValueError, match="continuous-boundary"):
        solve_regime_switch_equilibrium(log_linear)

    log_quadratic = _equity_params(
        EquilibriumClosure.LOG_QUADRATIC,
        utility_power=0.5,
    )
    with pytest.raises(ValueError, match="outward cubic"):
        solve_regime_switch_equilibrium(log_quadratic)


@pytest.mark.parametrize("closure", list(EquilibriumClosure))
def test_frozen_volatility_equilibrium_matches_matrix_exponential(
    closure: EquilibriumClosure,
) -> None:
    frozen = RegimeLogSvDynamics(
        theta=0.2,
        kappa1=2.0,
        kappa2=2.0,
        beta=0.0,
        volvol=0.0,
    )
    params = RegimeSwitchLogSvParams(
        sigma0=0.2,
        regimes=(frozen, frozen),
        transitions=(
            RegimeTransition(0.1, -1.0 / 3.0),
            RegimeTransition(1.0, 3.0 / 23.0),
        ),
        risk_premia=CrraRiskPremia(-0.5, 3.0, closure),
    )
    solution = solve_regime_switch_equilibrium(params)
    utility_power = params.risk_premia.utility_power
    potential = 0.5 * utility_power * (1.0 - utility_power) * params.sigma0**2
    matrix = np.array(
        [
            [
                potential - params.transitions[Regime.GROWTH].intensity,
                params.transitions[Regime.GROWTH].intensity
                * params.transition_factor(Regime.GROWTH),
            ],
            [
                params.transitions[Regime.STRESS].intensity
                * params.transition_factor(Regime.STRESS),
                potential - params.transitions[Regime.STRESS].intensity,
            ],
        ]
    )
    for horizon in (0.25, 1.0, 3.0):
        expected = expm(matrix * horizon) @ np.ones(2)
        observed = np.array(
            [
                np.exp(solution.log_value_coefficient(horizon, params.sigma0, regime))
                for regime in Regime
            ]
        )
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2.0e-10)


@pytest.mark.parametrize("closure", list(EquilibriumClosure))
def test_mgf_martingale_roots_hold_conditionally_on_each_state(
    closure: EquilibriumClosure,
) -> None:
    params = _equity_params(closure)
    log_mgf = compute_regime_switch_log_mgf_grid(
        params=params,
        equilibrium=solve_regime_switch_equilibrium(params),
        ttm=0.25,
        phi_grid=np.array([0.0, -1.0]),
    )

    assert log_mgf.shape == (2, 2)
    np.testing.assert_allclose(log_mgf, 0.0, rtol=0.0, atol=2.0e-11)


def test_option_rhs_uses_fixed_agent_horizon_time_index() -> None:
    class RecordingEquilibrium:
        def __init__(self) -> None:
            self.horizons: list[float] = []

        def coefficients(self, horizon: float) -> np.ndarray:
            self.horizons.append(horizon)
            return np.zeros((2, 2))

    params = _equity_params()
    recording = RecordingEquilibrium()
    _option_rhs(
        0.07,
        np.zeros(6, dtype=np.complex128),
        params=params,
        equilibrium=recording,
        ttm=0.25,
        phi_grid=np.array([-0.5 + 0.3j]),
        degree=2,
    )
    assert recording.horizons == pytest.approx([2.82])


@pytest.mark.parametrize("closure", list(EquilibriumClosure))
def test_induced_volatility_drift_has_the_derived_polynomial_degree(
    closure: EquilibriumClosure,
) -> None:
    params = _equity_params(closure)
    equilibrium = solve_regime_switch_equilibrium(params)
    sigma = np.linspace(0.08, 0.36, 31)

    for regime in Regime:
        dynamics = params.regimes[regime]
        state = evaluate_risk_neutral_state(
            params,
            equilibrium,
            params.risk_premia.agent_horizon,
            sigma,
            regime,
        )
        degree = closure.degree + 1
        fitted = np.polynomial.polynomial.polyfit(
            sigma - dynamics.theta, state.volatility_drift, degree
        )
        reconstructed = np.polynomial.polynomial.polyval(sigma - dynamics.theta, fitted)
        np.testing.assert_allclose(state.volatility_drift, reconstructed, rtol=0.0, atol=2.0e-10)
        if closure == EquilibriumClosure.LOG_LINEAR:
            loading = equilibrium.coefficients(params.risk_premia.agent_horizon)[regime, 1]
            expected = (dynamics.kappa1 + dynamics.kappa2 * sigma) * (dynamics.theta - sigma)
            expected += (
                -dynamics.beta * params.risk_premia.relative_risk_aversion
                + dynamics.vartheta2 * loading
            ) * sigma**2
            np.testing.assert_allclose(state.volatility_drift, expected, rtol=0.0, atol=2.0e-14)
        else:
            quadratic = equilibrium.coefficients(params.risk_premia.agent_horizon)[regime, 2]
            np.testing.assert_allclose(
                fitted[3], 2.0 * dynamics.vartheta2 * quadratic, rtol=0.0, atol=2.0e-10
            )


def test_induced_q_simulation_matches_frozen_canonical_replay() -> None:
    """Freeze RNG ordering, shared shocks, atomic jumps, and the fixed Q clock."""

    sample = simulate_regime_switch_logsv_terminal(
        _equity_params(),
        0.25,
        initial_regime=Regime.STRESS,
        nb_path=12,
        nb_steps_per_year=16,
        seed=2405,
    )
    expected_log_return = np.array(
        [
            0.25206743279672245,
            0.02975041356977068,
            -0.02977693732986204,
            0.02984860526583684,
            -0.06949354199644273,
            0.03742854742393706,
            -0.17070685323021778,
            0.05303943376062875,
            0.00938031920743862,
            -0.07875865721749825,
            -0.00215093092262483,
            0.00521628028106579,
        ]
    )
    expected_sigma = np.array(
        [
            0.2673122023101864,
            0.08664019794326777,
            0.29540743080334736,
            0.21389304776175921,
            0.2900213974297296,
            0.1977909609263617,
            0.35332538206498165,
            0.10702367516598514,
            0.08596977370091605,
            0.33486649625160375,
            0.11691849400901172,
            0.1500424677570816,
        ]
    )
    expected_qvar = np.array(
        [
            0.00797458173341974,
            0.0031873555151374,
            0.01341211747127344,
            0.0072446492782667,
            0.00600456478851687,
            0.01128016965994743,
            0.01415554379013085,
            0.00475670500507588,
            0.00639454520276119,
            0.00651010418451535,
            0.00677205839131207,
            0.00525881590248569,
        ]
    )

    np.testing.assert_allclose(
        sample.log_forward_return, expected_log_return, rtol=0.0, atol=5.0e-13
    )
    np.testing.assert_allclose(sample.sigma, expected_sigma, rtol=0.0, atol=5.0e-13)
    np.testing.assert_allclose(sample.qvar, expected_qvar, rtol=0.0, atol=5.0e-14)
    np.testing.assert_array_equal(
        sample.regime,
        np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=np.int8),
    )


def test_state_conditional_chain_outputs_follow_model_pricer_contract() -> None:
    chain = _state_chain()
    params = _equity_params()
    pricer = RegimeSwitchLogSVPricer()
    conditional = pricer.compute_state_conditional_prices_with_vols(chain, params, max_phi=401)

    for regime in Regime:
        prices, ivols = conditional.for_regime(regime)
        recomputed_ivols = chain.compute_model_ivols_from_chain_data(model_prices=prices)
        assert len(prices) == len(chain.ttms)
        assert len(ivols) == len(chain.ttms)
        for price_slice, ivol_slice, recomputed_slice, strikes in zip(
            prices, ivols, recomputed_ivols, chain.strikes_ttms
        ):
            assert price_slice.shape == strikes.shape
            assert ivol_slice.shape == strikes.shape
            assert np.all(np.isfinite(price_slice))
            assert np.all(np.isfinite(ivol_slice))
            assert np.all(price_slice >= 0.0)
            np.testing.assert_allclose(ivol_slice, recomputed_slice, rtol=0.0, atol=0.0)

    growth_prices, _ = conditional.for_regime(Regime.GROWTH)
    stress_prices, _ = conditional.for_regime(Regime.STRESS)
    assert (
        max(np.max(np.abs(growth - stress)) for growth, stress in zip(growth_prices, stress_prices))
        > 1.0e-4
    )

    standard_growth = pricer.price_chain(chain, params, max_phi=401)
    standard_stress = pricer.price_chain(chain, params, initial_regime=Regime.STRESS, max_phi=401)
    for standard, expected in zip(standard_growth, growth_prices):
        np.testing.assert_allclose(standard, expected, rtol=0.0, atol=0.0)
    for standard, expected in zip(standard_stress, stress_prices):
        np.testing.assert_allclose(standard, expected, rtol=0.0, atol=0.0)


def test_state_conditional_prices_are_forward_homogeneous() -> None:
    base = _state_chain()
    scale = 3.7
    scaled = OptionChain(
        ttms=base.ttms.copy(),
        forwards=scale * base.forwards,
        discfactors=base.discfactors.copy(),
        strikes_ttms=List(scale * np.asarray(strikes) for strikes in base.strikes_ttms),
        optiontypes_ttms=List(
            np.asarray(optiontypes).copy() for optiontypes in base.optiontypes_ttms
        ),
        ids=base.ids.copy(),
    )
    pricer = RegimeSwitchLogSVPricer()
    base_conditional = pricer.compute_state_conditional_prices_with_vols(
        base, _equity_params(), max_phi=401
    )
    scaled_conditional = pricer.compute_state_conditional_prices_with_vols(
        scaled, _equity_params(), max_phi=401
    )

    for regime in Regime:
        base_prices, base_ivols = base_conditional.for_regime(regime)
        scaled_prices, scaled_ivols = scaled_conditional.for_regime(regime)
        for base_price, scaled_price, base_ivol, scaled_ivol in zip(
            base_prices, scaled_prices, base_ivols, scaled_ivols
        ):
            np.testing.assert_allclose(scaled_price, scale * base_price, rtol=2.0e-13, atol=2.0e-14)
            np.testing.assert_allclose(scaled_ivol, base_ivol, rtol=0.0, atol=2.0e-13)


def test_state_conditional_prices_satisfy_put_call_parity() -> None:
    chain = _paired_chain()
    conditional = RegimeSwitchLogSVPricer().compute_state_conditional_prices_with_vols(
        chain, _equity_params(), max_phi=801
    )
    strikes = np.array([0.90, 1.00, 1.10])

    for regime in Regime:
        prices, _ = conditional.for_regime(regime)
        calls, puts = prices[0][:3], prices[0][3:]
        np.testing.assert_allclose(
            calls - puts,
            chain.discfactors[0] * (chain.forwards[0] - strikes),
            rtol=0.0,
            atol=2.0e-10,
        )


def test_invisible_switching_reduces_to_scalar_logsv_first_expansion() -> None:
    dynamics = RegimeLogSvDynamics(
        theta=0.2,
        kappa1=2.0,
        kappa2=5.0,
        beta=0.0,
        volvol=0.4,
    )
    params = RegimeSwitchLogSvParams(
        sigma0=0.21,
        regimes=(dynamics, dynamics),
        transitions=(RegimeTransition(0.4, 0.0), RegimeTransition(0.7, 0.0)),
        risk_premia=CrraRiskPremia(0.0, 1.0),
    )
    phi = np.array([-0.5 + value * 1j for value in (0.0, 0.5, 1.0, 2.0)])
    regime_log_mgf = compute_regime_switch_log_mgf_grid(
        params,
        ttm=0.25,
        phi_grid=phi,
        expansion_order=scalar_afe.ExpansionOrder.FIRST,
    )
    zeros = np.zeros_like(phi)
    _, scalar_log_mgf = scalar_afe.compute_logsv_a_mgf_grid(
        ttm=0.25,
        phi_grid=phi,
        psi_grid=zeros,
        theta_grid=zeros,
        sigma0=params.sigma0,
        theta=dynamics.theta,
        kappa1=dynamics.kappa1,
        kappa2=dynamics.kappa2,
        beta=dynamics.beta,
        volvol=dynamics.volvol,
        expansion_order=scalar_afe.ExpansionOrder.FIRST,
    )

    np.testing.assert_allclose(
        regime_log_mgf[Regime.GROWTH], scalar_log_mgf, rtol=2.0e-6, atol=5.0e-10
    )
    np.testing.assert_allclose(
        regime_log_mgf[Regime.STRESS], scalar_log_mgf, rtol=2.0e-6, atol=5.0e-10
    )


def test_maturity_and_payoff_scope_fail_closed() -> None:
    params = _equity_params()
    pricer = RegimeSwitchLogSVPricer()
    with pytest.raises(ValueError, match="odd integer"):
        pricer.price_chain(_state_chain(), params, max_phi=401.0)

    too_long = OptionChain.slice_to_chain(
        ttm=params.risk_premia.agent_horizon + 0.01,
        forward=1.0,
        strikes=np.array([1.0]),
        optiontypes=np.array(["C"]),
        id="too-long",
    )
    with pytest.raises(ValueError, match="agent_horizon"):
        pricer.price_chain(too_long, params, max_phi=101)

    inverse = OptionChain.slice_to_chain(
        ttm=0.1,
        forward=1.0,
        strikes=np.array([1.0]),
        optiontypes=np.array(["IC"]),
        id="inverse",
    )
    with pytest.raises(NotImplementedError, match="ordinary European"):
        pricer.price_chain(inverse, params, max_phi=101)


@pytest.mark.slow
@pytest.mark.parametrize("closure", list(EquilibriumClosure))
def test_fourier_grid_refinement_is_small_for_both_initial_states(
    closure: EquilibriumClosure,
) -> None:
    chain = OptionChain.slice_to_chain(
        ttm=0.25,
        forward=1.0,
        strikes=np.array([0.9, 1.0, 1.1]),
        optiontypes=np.array(["P", "C", "C"]),
        id="3m",
    )
    params = _equity_params(closure)
    pricer = RegimeSwitchLogSVPricer()
    coarse = pricer.compute_state_conditional_prices_with_vols(chain, params, max_phi=1_601)
    refined = pricer.compute_state_conditional_prices_with_vols(chain, params, max_phi=3_201)

    for regime in Regime:
        coarse_prices, _ = coarse.for_regime(regime)
        refined_prices, _ = refined.for_regime(regime)
        np.testing.assert_allclose(coarse_prices[0], refined_prices[0], rtol=0.0, atol=5.0e-6)


@pytest.mark.slow
@pytest.mark.parametrize("initial_regime", list(Regime))
@pytest.mark.parametrize("closure", list(EquilibriumClosure))
def test_state_conditional_analytic_prices_match_induced_q_monte_carlo(
    closure: EquilibriumClosure,
    initial_regime: Regime,
) -> None:
    chain = OptionChain.slice_to_chain(
        ttm=0.25,
        forward=1.0,
        strikes=np.array([0.90, 1.00, 1.10]),
        optiontypes=np.array(["P", "C", "C"]),
        id="3m",
    )
    params = _equity_params(closure)
    pricer = RegimeSwitchLogSVPricer()
    analytic = np.asarray(
        pricer.price_chain(
            chain,
            params,
            initial_regime=initial_regime,
            max_phi=1_601,
        )[0]
    )
    sample = simulate_regime_switch_logsv_terminal(
        params,
        ttm=chain.ttms[0],
        initial_regime=initial_regime,
        nb_path=40_000,
        nb_steps_per_year=720,
        seed=911 + 10 * closure.degree + int(initial_regime),
    )
    monte_carlo, standard_errors = compute_mc_vars_payoff(
        x0=sample.log_forward_return,
        sigma0=sample.sigma,
        qvar0=sample.qvar,
        ttm=chain.ttms[0],
        forward=chain.forwards[0],
        strikes_ttm=np.asarray(chain.strikes_ttms[0]),
        optiontypes_ttm=np.asarray(chain.optiontypes_ttms[0]),
        discfactor=chain.discfactors[0],
    )
    tolerance = 5.0 * standard_errors + 1.5e-4
    assert np.all(np.abs(analytic - monte_carlo) <= tolerance)

    forward_mean, forward_error = sample.forward_martingale
    assert abs(forward_mean - 1.0) <= 5.0 * forward_error + 5.0e-4


@pytest.mark.slow
def test_public_mc_chain_prices_two_maturities_and_honours_state_override() -> None:
    chain = _state_chain()
    params = _equity_params()
    pricer = RegimeSwitchLogSVPricer()
    analytic = pricer.price_chain(
        chain,
        params,
        initial_regime=Regime.STRESS,
        max_phi=1_601,
    )
    monte_carlo, standard_errors = pricer.model_mc_price_chain(
        chain,
        params,
        initial_regime=Regime.STRESS,
        nb_path=25_000,
        nb_steps_per_year=720,
        seed=2405,
    )

    assert len(monte_carlo) == len(chain.ttms)
    assert len(standard_errors) == len(chain.ttms)
    for analytic_slice, mc_slice, error_slice in zip(analytic, monte_carlo, standard_errors):
        assert np.all(np.abs(np.asarray(analytic_slice) - mc_slice) <= 6.0 * error_slice + 2.5e-4)
