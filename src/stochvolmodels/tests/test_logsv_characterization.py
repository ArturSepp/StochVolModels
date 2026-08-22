import numpy as np
import pandas as pd
import pytest
from numba.typed import List
from scipy.stats import norm
import stochvolmodels.fitters.logsv_smile as smile
from stochvolmodels.pricers.logsv import affine_expansion as afe
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv.vol_moments_ode import (
    compute_analytic_qvar,
    compute_analytic_vol_moments,
    compute_expected_vol_t,
    compute_sqrt_qvar_t,
    compute_vol_moments_t,
    fit_model_vol_backbone_to_varswaps,
)

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.pricers.logsv_pricer import (
    LogSVPricer,
    get_randoms_for_chain_valuation,
    get_randoms_for_rough_vol_chain_valuation,
    logsv_mc_chain_pricer,
    logsv_mc_chain_pricer_fixed_randoms,
    set_vol_scaler,
    simulate_vol_paths,
    simulate_logsv_x_vol_terminal,
    v0_implied,
)
from stochvolmodels.utils.config import VariableType


def _paired_chain() -> OptionChain:
    strikes = np.array([0.9, 1.0, 1.1])
    return OptionChain.slice_to_chain(
        ttm=0.25,
        forward=1.0,
        strikes=np.concatenate((strikes, strikes)),
        optiontypes=np.array(["C", "C", "C", "P", "P", "P"]),
        discfactor=0.98,
        id="3m",
    )


def _representative_params() -> LogSvParams:
    return LogSvParams(
        sigma0=0.2,
        theta=0.22,
        kappa1=3.0,
        kappa2=12.0,
        beta=-0.3,
        volvol=0.4,
    )


def _reference_bsm_prices(chain: OptionChain, vol: float) -> np.ndarray:
    strikes = np.asarray(chain.strikes_ttms[0])
    optiontypes = np.asarray(chain.optiontypes_ttms[0])
    forward = chain.forwards[0]
    total_vol = vol * np.sqrt(chain.ttms[0])
    d1 = np.log(forward / strikes) / total_vol + 0.5 * total_vol
    d2 = d1 - total_vol
    undiscounted = np.where(
        optiontypes == "C",
        forward * norm.cdf(d1) - strikes * norm.cdf(d2),
        strikes * norm.cdf(-d2) - forward * norm.cdf(-d1),
    )
    return chain.discfactors[0] * undiscounted


@pytest.mark.paper_replication
def test_logsv_constant_volatility_limit_matches_independent_bsm_reference() -> None:
    chain = _paired_chain()
    params = LogSvParams(
        sigma0=0.2,
        theta=0.2,
        kappa1=4.0,
        kappa2=20.0,
        beta=0.0,
        volvol=1.0e-3,
    )

    prices = np.asarray(LogSVPricer().price_chain(chain, params)[0])
    expected = _reference_bsm_prices(chain, vol=0.2)
    calls, puts = prices[:3], prices[3:]
    strikes = np.array([0.9, 1.0, 1.1])

    np.testing.assert_allclose(prices, expected, rtol=0.0, atol=1.2e-5)
    assert np.all(np.isfinite(prices))
    assert np.all(prices >= 0.0)
    assert np.all(np.diff(calls) <= 0.0)
    assert np.all(np.diff(puts) >= 0.0)
    np.testing.assert_allclose(
        calls - puts,
        chain.discfactors[0] * (chain.forwards[0] - strikes),
        rtol=0.0,
        atol=2.0e-8,
    )


def test_logsv_chain_slice_and_vanilla_interfaces_are_consistent() -> None:
    """All public pricing entry points must delegate to the same log-SV result."""
    chain = OptionChain.slice_to_chain(
        ttm=0.25,
        forward=1.0,
        strikes=np.array([0.9, 1.0, 1.1]),
        optiontypes=np.array(["P", "C", "C"]),
        discfactor=0.98,
        id="3m",
    )
    params = _representative_params()
    pricer = LogSVPricer()

    chain_prices = np.asarray(pricer.price_chain(chain, params)[0])
    slice_prices, slice_ivols = pricer.price_slice(
        params=params,
        ttm=chain.ttms[0],
        forward=chain.forwards[0],
        strikes=np.asarray(chain.strikes_ttms[0]),
        optiontypes=np.asarray(chain.optiontypes_ttms[0]),
        discfactor=chain.discfactors[0],
    )

    np.testing.assert_allclose(slice_prices, chain_prices, rtol=0.0, atol=1.0e-14)
    assert np.all(np.isfinite(slice_ivols))
    for index, (strike, optiontype) in enumerate(
        zip(chain.strikes_ttms[0], chain.optiontypes_ttms[0])
    ):
        vanilla_price, vanilla_ivol = pricer.price_vanilla(
            params=params,
            ttm=chain.ttms[0],
            forward=chain.forwards[0],
            strike=strike,
            optiontype=optiontype,
            discfactor=chain.discfactors[0],
        )
        np.testing.assert_allclose(vanilla_price, chain_prices[index], rtol=0.0, atol=1.0e-14)
        np.testing.assert_allclose(vanilla_ivol, slice_ivols[index], rtol=0.0, atol=1.0e-12)


@pytest.mark.parametrize(
    ("is_spot_measure", "roots"),
    [(True, [0.0, -1.0]), (False, [0.0, 1.0])],
)
@pytest.mark.paper_replication
def test_logsv_mgf_normalization_roots(
    is_spot_measure: bool, roots: list[float]
) -> None:
    params = _representative_params()
    phi = np.asarray(roots, dtype=np.complex128)
    zeros = np.zeros_like(phi)

    coefficients, log_mgf = afe.compute_logsv_a_mgf_grid(
        ttm=0.25,
        phi_grid=phi,
        psi_grid=zeros,
        theta_grid=zeros,
        is_spot_measure=is_spot_measure,
        **params.to_dict(),
    )

    assert np.all(np.isfinite(coefficients))
    np.testing.assert_allclose(log_mgf, 0.0, rtol=0.0, atol=1.0e-14)


@pytest.mark.parametrize("expansion_order", [afe.ExpansionOrder.FIRST, afe.ExpansionOrder.SECOND])
def test_logsv_affine_terms_preserve_measure_and_dimension_contracts(
    expansion_order: afe.ExpansionOrder,
) -> None:
    """The published quadratic terms are measure invariant and correctly sized."""
    params = _representative_params()
    common = dict(
        theta=params.theta,
        kappa1=params.kappa1,
        kappa2=params.kappa2,
        beta=params.beta,
        volvol=params.volvol,
        phi=np.complex128(-0.5 + 0.7j),
        psi=np.complex128(0.0),
        expansion_order=expansion_order,
    )
    spot = afe.func_a_ode_quadratic_terms.py_func(**common, is_spot_measure=True)
    inverse = afe.func_a_ode_quadratic_terms.py_func(**common, is_spot_measure=False)
    compiled = afe.func_a_ode_quadratic_terms(**common, is_spot_measure=True)
    n_terms = 3 if expansion_order == afe.ExpansionOrder.FIRST else 5

    assert afe.get_expansion_n.py_func(expansion_order) == n_terms
    assert spot[0].shape == (n_terms, n_terms, n_terms)
    assert spot[1].shape == (n_terms, n_terms)
    assert spot[2].shape == (n_terms,)
    np.testing.assert_allclose(spot[0], inverse[0], rtol=0.0, atol=0.0)
    for python_value, compiled_value in zip(spot, compiled):
        np.testing.assert_allclose(python_value, compiled_value, rtol=0.0, atol=5.0e-15)


def test_logsv_affine_rhs_jacobian_matches_finite_difference() -> None:
    """The analytic ODE Jacobian agrees with independent central differences."""
    params = _representative_params()
    matrices = afe.func_a_ode_quadratic_terms.py_func(
        theta=params.theta,
        kappa1=params.kappa1,
        kappa2=params.kappa2,
        beta=params.beta,
        volvol=params.volvol,
        phi=np.complex128(-0.5 + 0.4j),
        psi=np.complex128(0.0),
        expansion_order=afe.ExpansionOrder.FIRST,
    )
    coefficients = np.array([0.01 + 0.02j, -0.03 + 0.01j, 0.02 - 0.01j])
    jacobian = afe.func_rhs_jac.py_func(0.0, coefficients, *matrices)
    epsilon = 1.0e-7
    finite_difference = np.column_stack(
        [
            (
                afe.func_rhs.py_func(
                    0.0,
                    coefficients + epsilon * np.eye(3)[index],
                    *matrices,
                )
                - afe.func_rhs.py_func(
                    0.0,
                    coefficients - epsilon * np.eye(3)[index],
                    *matrices,
                )
            )
            / (2.0 * epsilon)
            for index in range(3)
        ]
    )

    np.testing.assert_allclose(jacobian, finite_difference, rtol=2.0e-8, atol=2.0e-10)
    np.testing.assert_allclose(
        afe.func_rhs(0.0, coefficients, *matrices),
        afe.func_rhs.py_func(0.0, coefficients, *matrices),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_logsv_affine_semi_analytic_solver_matches_scipy_integrator() -> None:
    """The fast daily-step solver agrees with SciPy's independent adaptive solver."""
    params = _representative_params()
    common = dict(
        ttm=0.05,
        theta=params.theta,
        kappa1=params.kappa1,
        kappa2=params.kappa2,
        beta=params.beta,
        volvol=params.volvol,
        phi=np.complex128(-0.5 + 0.35j),
        psi=np.complex128(0.0),
        is_spot_measure=True,
        expansion_order=afe.ExpansionOrder.FIRST,
    )
    scipy_solution = afe.solve_ode_for_a(**common).y[:, -1]
    stiff_solution = afe.solve_ode_for_a(**common, is_stiff_solver=True).y[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        semi_analytic = afe.solve_analytic_ode_for_a.py_func(**common, year_days=2_000)
    compiled = afe.solve_analytic_ode_for_a(**common, year_days=2_000)

    np.testing.assert_allclose(semi_analytic, scipy_solution, rtol=2.0e-5, atol=2.0e-7)
    np.testing.assert_allclose(stiff_solution, scipy_solution, rtol=3.0e-3, atol=2.0e-7)
    np.testing.assert_allclose(compiled, semi_analytic, rtol=0.0, atol=1.0e-13)


def test_logsv_affine_grid_solvers_preserve_transform_roots() -> None:
    """Both grid solvers keep the exact MGF normalization roots at zero."""
    params = _representative_params()
    phi = np.array([0.0 + 0.0j, -1.0 + 0.0j])
    psi = np.zeros_like(phi)
    common = dict(
        phi_grid=phi,
        psi_grid=psi,
        ttm=0.05,
        theta=params.theta,
        kappa1=params.kappa1,
        kappa2=params.kappa2,
        beta=params.beta,
        volvol=params.volvol,
        is_spot_measure=True,
        expansion_order=afe.ExpansionOrder.FIRST,
    )
    scipy_grid = afe.solve_a_ode_grid(**common)
    with np.errstate(divide="ignore", invalid="ignore"):
        semi_analytic_grid = afe.solve_analytic_ode_grid_phi.py_func(**common)
        legacy_root = afe.solve_analytic_ode_for_a0.py_func(
            t_span=(0.0, 0.05),
            theta=params.theta,
            kappa1=params.kappa1,
            kappa2=params.kappa2,
            beta=params.beta,
            volvol=params.volvol,
            phi=np.complex128(0.0),
            psi=np.complex128(0.0),
            expansion_order=afe.ExpansionOrder.FIRST,
        )

    np.testing.assert_allclose(scipy_grid, 0.0, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(semi_analytic_grid, 0.0, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(legacy_root, 0.0, rtol=0.0, atol=1.0e-14)


def test_logsv_affine_mgf_analytic_and_adaptive_routes_agree() -> None:
    params = _representative_params()
    phi = np.array([-0.5 + 0.25j])
    zeros = np.zeros_like(phi)
    common = dict(
        ttm=0.05,
        phi_grid=phi,
        psi_grid=zeros,
        theta_grid=zeros,
        variable_type=VariableType.LOG_RETURN,
        expansion_order=afe.ExpansionOrder.FIRST,
        is_spot_measure=True,
        **params.to_dict(),
    )
    _, adaptive = afe.compute_logsv_a_mgf_grid(**common, is_analytic=False)
    _, semi_analytic = afe.compute_logsv_a_mgf_grid(**common, is_analytic=True)

    np.testing.assert_allclose(semi_analytic, adaptive, rtol=2.0e-4, atol=2.0e-6)
    with pytest.raises(NotImplementedError):
        afe.compute_logsv_a_mgf_grid(
            **{**common, "expansion_order": afe.ExpansionOrder.ZERO},
            is_analytic=False,
        )


def test_logsv_affine_initial_conditions_cover_all_state_variables() -> None:
    phi = np.array([-0.5 + 0.0j, -0.5 + 0.5j])
    psi = np.array([0.0 + 0.0j, 0.0 + 0.5j])
    theta = np.array([0.0 + 0.0j, 0.0 + 0.5j])

    log_return = afe.get_init_conditions_a.py_func(
        phi, psi, theta, 3, VariableType.LOG_RETURN
    )
    qvar = afe.get_init_conditions_a.py_func(phi, psi, theta, 3, VariableType.Q_VAR)
    sigma = afe.get_init_conditions_a.py_func(phi, psi, theta, 3, VariableType.SIGMA)

    np.testing.assert_array_equal(log_return, np.zeros((2, 3)))
    np.testing.assert_array_equal(qvar, np.zeros((2, 3)))
    np.testing.assert_allclose(sigma[:, 1], -theta, rtol=0.0, atol=0.0)
    with pytest.raises(NotImplementedError):
        afe.get_init_conditions_a.py_func(phi, psi, theta, 3, object())


@pytest.mark.paper_replication
def test_logsv_analytic_prices_and_martingale_match_fixed_random_mc() -> None:
    chain = OptionChain.slice_to_chain(
        ttm=0.25,
        forward=1.0,
        strikes=np.array([0.9, 1.0, 1.1]),
        optiontypes=np.array(["P", "C", "C"]),
        discfactor=0.98,
        id="3m",
    )
    params = _representative_params()
    nb_path = 40_000
    nb_steps = 91
    dt = chain.ttms[0] / nb_steps
    rng = np.random.default_rng(123)
    w0s = List([rng.standard_normal((nb_steps, nb_path))])
    w1s = List([rng.standard_normal((nb_steps, nb_path))])
    dts = List([dt])

    analytic = np.asarray(LogSVPricer().price_chain(chain, params)[0])
    mc_prices, mc_errors = logsv_mc_chain_pricer_fixed_randoms(
        ttms=chain.ttms,
        forwards=chain.forwards,
        discfactors=chain.discfactors,
        strikes_ttms=chain.strikes_ttms,
        optiontypes_ttms=chain.optiontypes_ttms,
        W0s=w0s,
        W1s=w1s,
        dts=dts,
        v0=params.sigma0,
        theta=params.theta,
        kappa1=params.kappa1,
        kappa2=params.kappa2,
        beta=params.beta,
        volvol=params.volvol,
        vol_backbone_etas=params.get_vol_backbone_etas(chain.ttms),
    )
    python_mc_prices, python_mc_errors = logsv_mc_chain_pricer_fixed_randoms.py_func(
        ttms=chain.ttms,
        forwards=chain.forwards,
        discfactors=chain.discfactors,
        strikes_ttms=chain.strikes_ttms,
        optiontypes_ttms=chain.optiontypes_ttms,
        W0s=w0s,
        W1s=w1s,
        dts=dts,
        v0=params.sigma0,
        theta=params.theta,
        kappa1=params.kappa1,
        kappa2=params.kappa2,
        beta=params.beta,
        volvol=params.volvol,
        vol_backbone_etas=params.get_vol_backbone_etas(chain.ttms),
    )
    mc_prices = np.asarray(mc_prices[0])
    mc_errors = np.asarray(mc_errors[0])

    np.testing.assert_allclose(python_mc_prices[0], mc_prices, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(python_mc_errors[0], mc_errors, rtol=0.0, atol=1.0e-14)
    assert np.all(np.isfinite(mc_prices))
    assert np.all(mc_errors > 0.0)
    assert np.all(np.abs(analytic - mc_prices) <= 4.0 * mc_errors)

    x, sigma, qvar = simulate_logsv_x_vol_terminal(
        ttm=chain.ttms[0],
        x0=np.zeros(nb_path),
        sigma0=np.full(nb_path, params.sigma0),
        qvar0=np.zeros(nb_path),
        theta=params.theta,
        kappa1=params.kappa1,
        kappa2=params.kappa2,
        beta=params.beta,
        volvol=params.volvol,
        nb_path=nb_path,
        W0=w0s[0],
        W1=w1s[0],
        dt=dt,
    )
    python_x, python_sigma, python_qvar = simulate_logsv_x_vol_terminal.py_func(
        ttm=chain.ttms[0],
        x0=np.zeros(nb_path),
        sigma0=np.full(nb_path, params.sigma0),
        qvar0=np.zeros(nb_path),
        theta=params.theta,
        kappa1=params.kappa1,
        kappa2=params.kappa2,
        beta=params.beta,
        volvol=params.volvol,
        nb_path=nb_path,
        W0=w0s[0],
        W1=w1s[0],
        dt=dt,
    )
    np.testing.assert_allclose(python_x, x, rtol=0.0, atol=3.0e-14)
    np.testing.assert_allclose(python_sigma, sigma, rtol=0.0, atol=3.0e-14)
    np.testing.assert_allclose(python_qvar, qvar, rtol=0.0, atol=3.0e-14)
    terminal_spot = np.exp(x)
    spot_error = np.std(terminal_spot, ddof=1) / np.sqrt(nb_path)

    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(sigma))
    assert np.all(np.isfinite(qvar))
    assert np.all(sigma > 0.0)
    assert np.all(qvar >= 0.0)
    assert abs(np.mean(terminal_spot) - 1.0) <= 4.0 * spot_error

    expected_sigma = compute_expected_vol_t(params, np.array([chain.ttms[0]]), n_terms=8)[0]
    expected_qvar = compute_analytic_qvar(params, ttm=chain.ttms[0], n_terms=8)
    sigma_error = np.std(sigma, ddof=1) / np.sqrt(nb_path)
    qvar_rate = qvar / chain.ttms[0]
    qvar_error = np.std(qvar_rate, ddof=1) / np.sqrt(nb_path)
    assert abs(np.mean(sigma) - expected_sigma) <= 4.0 * sigma_error
    assert abs(np.mean(qvar_rate) - expected_qvar) <= 4.0 * qvar_error


@pytest.mark.paper_replication
def test_logsv_martingale_and_fourth_moment_constraints_hold() -> None:
    params = _representative_params()
    eigenvalues = np.linalg.eigvals(params.get_vol_moments_lambda(n_terms=4))

    assert params.kappa2 - params.beta > 0.0
    assert params.kappa2 - 2.0 * params.beta > 0.0
    assert params.kappa - 1.5 * params.vartheta2 > 0.0
    assert np.all(np.isfinite(eigenvalues))
    assert np.all(np.real(eigenvalues) < 0.0)


def test_logsv_moment_helpers_preserve_zero_time_and_vector_contracts() -> None:
    params = _representative_params()
    maturities = np.array([0.0, 0.05, 0.25])
    displacement = params.sigma0 - params.theta
    initial = compute_analytic_vol_moments(params, t=0.0, n_terms=4)
    moments = compute_vol_moments_t(params, maturities, n_terms=4)
    expected_vol = compute_expected_vol_t(params, maturities, n_terms=4)
    sqrt_qvar = compute_sqrt_qvar_t(params, maturities, n_terms=4)

    np.testing.assert_allclose(
        initial,
        np.array([displacement**power for power in range(1, 5)]),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(moments[0], initial, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(expected_vol, moments[:, 0] + params.theta, atol=1.0e-15)
    np.testing.assert_allclose(sqrt_qvar[0], params.sigma0, atol=1.0e-15)
    for index, maturity in enumerate(maturities):
        np.testing.assert_allclose(
            sqrt_qvar[index] ** 2,
            compute_analytic_qvar(params, ttm=maturity, n_terms=4),
            rtol=0.0,
            atol=1.0e-15,
        )


def test_logsv_parameter_container_grids_backbone_and_diagnostics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    params = LogSvParams(
        sigma0=0.2,
        theta=0.25,
        kappa1=2.0,
        kappa2=None,
        beta=-0.3,
        volvol=0.4,
        H=0.5,
    )
    np.testing.assert_allclose(params.kappa2, params.kappa1 / params.theta)
    params.approximate_kernel(T=1.0)
    np.testing.assert_array_equal(params.nodes, np.array([1.0e-3]))
    np.testing.assert_array_equal(params.weights, np.array([1.0]))
    assert params.to_dict()["sigma0"] == params.sigma0
    assert "sigma0=0.20" in params.to_str()
    np.testing.assert_allclose(params.kappa, params.kappa1 + params.kappa2 * params.theta)
    np.testing.assert_allclose(params.theta2, params.theta**2)
    np.testing.assert_allclose(params.vartheta2, params.beta**2 + params.volvol**2)
    np.testing.assert_allclose(params.gamma, params.kappa1 / params.theta)
    assert np.isfinite(params.eta)

    backbone = pd.Series([0.9, 1.1, 1.2], index=[0.1, 0.5, 1.0])
    params.set_vol_backbone(backbone)
    np.testing.assert_allclose(params.get_vol_backbone_eta(0.3), 1.1)
    np.testing.assert_allclose(
        params.get_vol_backbone_etas(np.array([0.05, 0.3, 0.8])),
        np.array([0.9, 1.1, 1.2]),
    )

    for variable_type, direct in (
        (VariableType.LOG_RETURN, params.get_x_grid),
        (VariableType.SIGMA, params.get_sigma_grid),
        (VariableType.Q_VAR, params.get_qvar_grid),
    ):
        dispatched = params.get_variable_space_grid(variable_type, ttm=0.25, n_stdevs=2, n=17)
        np.testing.assert_allclose(
            dispatched,
            direct(ttm=0.25, n_stdevs=2, n=17),
            rtol=0.0,
            atol=0.0,
        )
        assert dispatched.shape == (17,)
        assert np.all(np.diff(dispatched) > 0.0)
    with pytest.raises(NotImplementedError):
        params.get_variable_space_grid(object())

    params.assert_vol_moments_stability(n_terms=4)
    params.print_vol_moments_stability(n_terms=4)
    output = capsys.readouterr().out
    assert "vol moments stable" in output
    assert "eigenvalues" in output


@pytest.mark.parametrize(("hurst", "node_count"), [(0.45, 2), (0.3, 3)])
def test_logsv_rough_kernel_approximation_selects_documented_node_count(
    hurst: float,
    node_count: int,
) -> None:
    params = LogSvParams(H=hurst)
    params.approximate_kernel(T=1.0)

    assert params.nodes.shape == params.weights.shape == (node_count,)
    assert np.all(np.isfinite(params.nodes))
    assert np.all(np.isfinite(params.weights))


def test_logsv_varswap_backbone_is_one_for_model_consistent_quotes() -> None:
    params = _representative_params()
    maturities = np.array([0.05, 0.25, 0.75])
    model_strikes = pd.Series(
        compute_sqrt_qvar_t(params, maturities, n_terms=4),
        index=maturities,
    )

    fitted = fit_model_vol_backbone_to_varswaps(params, model_strikes, n_terms=4)

    np.testing.assert_allclose(fitted.to_numpy(), 1.0, rtol=0.0, atol=1.0e-13)
    np.testing.assert_array_equal(fitted.index.to_numpy(), maturities)


def test_logsv_fixed_random_generator_replays_the_same_seed() -> None:
    kwargs = dict(ttms=np.array([0.25]), nb_path=8, nb_steps_per_year=12, seed=7)
    first_w0, first_w1, first_dt = get_randoms_for_chain_valuation(**kwargs)
    second_w0, second_w1, second_dt = get_randoms_for_chain_valuation(**kwargs)

    np.testing.assert_array_equal(first_w0[0], second_w0[0])
    np.testing.assert_array_equal(first_w1[0], second_w1[0])
    np.testing.assert_array_equal(first_dt, second_dt)


def test_logsv_fixed_random_generator_does_not_mutate_global_rng() -> None:
    np.random.seed(91)
    expected = np.random.random()
    np.random.seed(91)

    get_randoms_for_chain_valuation(
        ttms=np.array([0.25]), nb_path=8, nb_steps_per_year=12, seed=7
    )

    np.testing.assert_allclose(np.random.random(), expected, rtol=0.0, atol=0.0)


def test_logsv_unfixed_mc_python_dispatch_produces_valid_prices() -> None:
    chain = OptionChain.slice_to_chain(
        ttm=0.02,
        forward=1.0,
        strikes=np.array([0.9, 1.0, 1.1]),
        optiontypes=np.array(["P", "C", "C"]),
        discfactor=0.99,
        id="short",
    )
    params = _representative_params()
    prices, errors = logsv_mc_chain_pricer.py_func(
        ttms=chain.ttms,
        forwards=chain.forwards,
        discfactors=chain.discfactors,
        strikes_ttms=chain.strikes_ttms,
        optiontypes_ttms=chain.optiontypes_ttms,
        v0=params.sigma0,
        theta=params.theta,
        kappa1=params.kappa1,
        kappa2=params.kappa2,
        beta=params.beta,
        volvol=params.volvol,
        vol_backbone_etas=params.get_vol_backbone_etas(chain.ttms),
        nb_path=512,
        nb_steps_per_year=360,
    )

    assert np.all(np.isfinite(prices[0]))
    assert np.all(np.asarray(prices[0]) >= 0.0)
    assert np.all(np.isfinite(errors[0]))
    assert np.all(np.asarray(errors[0]) >= 0.0)


def test_logsv_vol_path_simulator_and_short_term_scalers() -> None:
    params = _representative_params()
    brownians = np.zeros((8, 4))
    spot_paths, grid = simulate_vol_paths(
        ttm=0.02,
        v0=params.sigma0,
        theta=params.theta,
        kappa1=params.kappa1,
        kappa2=params.kappa2,
        beta=params.beta,
        volvol=params.volvol,
        nb_path=4,
        nb_steps_per_year=360,
        brownians=brownians,
        is_spot_measure=True,
    )
    inverse_paths, inverse_grid = simulate_vol_paths(
        ttm=0.02,
        v0=params.sigma0,
        theta=params.theta,
        kappa1=params.kappa1,
        kappa2=params.kappa2,
        beta=params.beta,
        volvol=params.volvol,
        nb_path=4,
        nb_steps_per_year=360,
        brownians=brownians,
        is_spot_measure=False,
    )

    assert spot_paths.shape == inverse_paths.shape == (9, 4)
    np.testing.assert_array_equal(grid, inverse_grid)
    np.testing.assert_allclose(spot_paths[0], params.sigma0, rtol=0.0, atol=0.0)
    assert np.all(spot_paths > 0.0)
    assert np.all(inverse_paths > 0.0)
    assert not np.array_equal(spot_paths[-1], inverse_paths[-1])
    np.testing.assert_allclose(
        set_vol_scaler(params.sigma0, 0.25),
        params.sigma0 * np.sqrt(0.5 / 12.0),
        rtol=0.0,
        atol=0.0,
    )
    simple = params.sigma0 - (params.beta**2 + params.volvol**2) * 0.02 / 4.0
    np.testing.assert_allclose(
        v0_implied.py_func(params.sigma0, 1.1, params.volvol, params.theta, params.kappa1, 0.02),
        params.sigma0 - (1.1**2 + params.volvol**2) * 0.02 / 4.0,
    )
    np.testing.assert_allclose(
        v0_implied.py_func(params.sigma0, 0.0, params.volvol, params.theta, params.kappa1, 0.02),
        params.sigma0 - params.volvol**2 * 0.02 / 4.0,
    )
    assert np.isfinite(
        v0_implied.py_func(
            params.sigma0,
            params.beta,
            params.volvol,
            params.theta,
            params.kappa1,
            0.02,
        )
    )
    assert simple < params.sigma0


def test_logsv_rough_random_generator_replays_without_global_rng_mutation() -> None:
    kwargs = dict(ttms=np.array([0.02, 0.04]), nb_path=4, nb_steps_per_year=360, seed=13)
    np.random.seed(29)
    expected = np.random.random()
    np.random.seed(29)
    first_z0, first_z1, first_grids = get_randoms_for_rough_vol_chain_valuation(**kwargs)
    second_z0, second_z1, second_grids = get_randoms_for_rough_vol_chain_valuation(**kwargs)

    np.testing.assert_array_equal(first_z0, second_z0)
    np.testing.assert_array_equal(first_z1, second_z1)
    for first, second in zip(first_grids, second_grids):
        np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.random.random(), expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("is_vega_weights", [False, True])
def test_approximate_logsv_smile_fit_recovers_synthetic_parameters(
    is_vega_weights: bool,
) -> None:
    log_strikes = np.linspace(-0.25, 0.25, 11)
    expected = {smile.ATM_VOL: 0.25, smile.BETA: -0.4, smile.VOLVOL: 0.6}
    mid_vols = smile.calc_logsv_ivols(log_strikes, **expected)

    fitted = smile.fit_logsv_ivols(
        log_strikes,
        mid_vols,
        ttm=0.5,
        is_vega_weights=is_vega_weights,
    )

    for name, value in expected.items():
        np.testing.assert_allclose(fitted[name], value, rtol=0.0, atol=2.0e-6)
    np.testing.assert_allclose(
        smile.calc_logsv_ivols(log_strikes, **fitted),
        mid_vols,
        rtol=0.0,
        atol=2.0e-8,
    )


@pytest.mark.parametrize(
    ("log_strikes", "mid_vols", "ttm", "message"),
    [
        (np.array([[0.0]]), np.array([0.2]), 0.5, "aligned"),
        (np.array([-0.1, 0.0]), np.array([0.2, 0.2]), 0.5, "three"),
        (np.array([-0.1, 0.0, 0.1]), np.array([0.2, np.nan, 0.2]), 0.5, "three"),
        (np.array([-0.1, 0.0, 0.1]), np.array([0.2, 0.2, 0.2]), 0.0, "ttm"),
    ],
)
def test_approximate_logsv_smile_fit_rejects_invalid_inputs(
    log_strikes: np.ndarray,
    mid_vols: np.ndarray,
    ttm: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        smile.fit_logsv_ivols(log_strikes, mid_vols, ttm)


def test_approximate_logsv_smile_partials_and_density_are_consistent() -> None:
    params = dict(sigma0=0.25, beta=-0.2, volvol=0.4)
    log_strikes = np.linspace(-0.2, 0.2, 81)
    level, first, second = smile.calc_logsv_ivols_partials(
        log_strikes,
        **params,
        eps=1.0e-4,
        is_analytic=False,
    )
    direct = smile.calc_logsv_ivols(log_strikes, **params)
    np.testing.assert_allclose(level, direct, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        first[1:-1],
        np.gradient(direct, log_strikes)[1:-1],
        rtol=0.0,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        second[2:-2],
        np.gradient(np.gradient(direct, log_strikes), log_strikes)[2:-2],
        rtol=0.0,
        atol=2.0e-8,
    )

    away_from_atm = np.array([-0.2, -0.1, 0.1, 0.2])
    analytic = smile.calc_logsv_ivols_partials(
        away_from_atm,
        **params,
        is_analytic=True,
    )
    assert all(np.all(np.isfinite(values)) for values in analytic)
    with np.errstate(divide="ignore", invalid="ignore"):
        exact = smile.calc_logsv_ivols(away_from_atm, **params, is_quadratic=False)
    assert np.all(np.isfinite(exact))

    density_grid = np.linspace(-1.0, 1.0, 2_001)
    density = smile.calc_logsv_pdf(
        ttm=0.5,
        log_strikes=density_grid,
        is_norm=False,
        **params,
    )
    normalized = smile.calc_logsv_pdf(
        ttm=0.5,
        log_strikes=density_grid,
        is_norm=True,
        **params,
    )
    expected_mass = density.to_numpy() * (density_grid[1] - density_grid[0])
    expected_mass /= np.sum(expected_mass)
    np.testing.assert_allclose(
        normalized.to_numpy(),
        expected_mass,
        rtol=0.0,
        atol=2.0e-18,
    )
    np.testing.assert_allclose(normalized.sum(), 1.0, rtol=0.0, atol=2.0e-16)


def test_approximate_logsv_density_default_grid_and_normalization_validation() -> None:
    params = dict(ttm=0.5, sigma0=0.25, beta=-0.2, volvol=0.4)
    density = smile.calc_logsv_pdf_core(**params)
    probability_mass = smile.calc_logsv_pdf(is_norm=True, **params)

    assert len(density) == len(probability_mass) == 100
    assert np.all(np.isfinite(density))
    np.testing.assert_allclose(probability_mass.sum(), 1.0, rtol=0.0, atol=2.0e-16)

    with pytest.raises(ValueError, match="at least two"):
        smile.calc_logsv_pdf(log_strikes=np.array([0.0]), is_norm=True, **params)
    with pytest.raises(ValueError, match="uniform grid"):
        smile.calc_logsv_pdf(
            log_strikes=np.array([-0.2, -0.1, 0.1]),
            is_norm=True,
            **params,
        )


def test_approximate_logsv_delta_maps_and_grid_prices_preserve_conventions() -> None:
    params = dict(sigma0=0.25, beta=-0.2, volvol=0.4)
    deltas = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    strikes = smile.infer_strikes_from_deltas(
        deltas=deltas,
        forward=1.0,
        ttm=0.5,
        **params,
    )
    log_strikes = np.log(strikes.to_numpy())
    total_vols = np.sqrt(0.5) * smile.calc_logsv_ivols(log_strikes, **params)
    recovered_deltas = norm.cdf(-log_strikes / total_vols + 0.5 * total_vols)
    np.testing.assert_allclose(recovered_deltas, deltas, rtol=0.0, atol=2.0e-12)
    assert np.all(np.diff(strikes.to_numpy()) < 0.0)

    put_deltas = np.array([-0.9, -0.75, -0.5, -0.25, -0.1])
    put_strikes = smile.infer_strikes_from_deltas(
        deltas=put_deltas,
        forward=1.0,
        ttm=0.5,
        **params,
    )
    put_log_strikes = np.log(put_strikes.to_numpy())
    put_total_vols = np.sqrt(0.5) * smile.calc_logsv_ivols(put_log_strikes, **params)
    recovered_put_deltas = (
        norm.cdf(-put_log_strikes / put_total_vols + 0.5 * put_total_vols) - 1.0
    )
    np.testing.assert_allclose(recovered_put_deltas, put_deltas, rtol=0.0, atol=2.0e-12)
    assert np.all(np.diff(put_strikes.to_numpy()) < 0.0)
    raw_vols = smile.get_vols_delta_space(
        forward=1.0,
        ttm=0.5,
        deltas=deltas,
        is_remap_to_str_delta=False,
        **params,
    )
    display_vols = smile.get_vols_delta_space(
        forward=1.0,
        ttm=0.5,
        deltas=deltas,
        is_remap_to_str_delta=True,
        **params,
    )
    density = smile.get_pdf_delta_space(
        forward=1.0,
        ttm=0.5,
        deltas=deltas,
        is_remap_to_straddle_delta=False,
        is_analytic=False,
        **params,
    )
    assert len(raw_vols) == len(display_vols) == len(density) == deltas.size
    assert np.all(np.isfinite(raw_vols))
    assert np.all(np.isfinite(density))

    given_log_strikes = np.linspace(-0.2, 0.2, 9)
    vols = pd.Series(smile.calc_logsv_ivols(given_log_strikes, **params))
    grid = np.linspace(-0.15, 0.15, 7)
    puts, calls = smile.generate_grid_option_prices_from_slice(
        vols=vols,
        given_log_strikes=given_log_strikes,
        log_strike_grid=grid,
        p0_ref=1.0,
        ttm=0.5,
        vol_addon=0.01,
    )
    np.testing.assert_allclose(
        calls.to_numpy() - puts.to_numpy(),
        1.0 - calls.index.to_numpy(),
        rtol=0.0,
        atol=2.0e-14,
    )


@pytest.mark.parametrize("delta", [-1.0, 0.0, 1.0, np.nan])
def test_approximate_logsv_delta_inversion_rejects_invalid_delta(delta: float) -> None:
    with pytest.raises(ValueError, match="deltas"):
        smile.infer_strikes_from_deltas(
            deltas=np.array([delta]),
            forward=1.0,
            ttm=0.5,
            sigma0=0.25,
            beta=-0.2,
            volvol=0.4,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"deltas": np.array([[0.5]])}, "one-dimensional"),
        ({"forward": 0.0}, "forward"),
        ({"ttm": 0.0}, "ttm"),
        ({"sigma0": 0.0}, "sigma0"),
    ],
)
def test_approximate_logsv_delta_inversion_rejects_invalid_market_inputs(
    overrides: dict[str, object],
    message: str,
) -> None:
    kwargs: dict[str, object] = {
        "deltas": np.array([0.5]),
        "forward": 1.0,
        "ttm": 0.5,
        "sigma0": 0.25,
        "beta": -0.2,
        "volvol": 0.4,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        smile.infer_strikes_from_deltas(**kwargs)
