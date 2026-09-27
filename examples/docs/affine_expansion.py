"""Canonical script of docs/affine_expansion.md.

Every Python block on that page is a verbatim excerpt of this file, and every number the page
quotes is asserted here. The affine expansion of the moment generating function of Section 4 of
Sepp and Rakhmonov (2023) is solved with the package, checked against exact results (the
martingale conditions, the log-normal limit, the exact moments at kappa2 = 0), and compared with
the second-order system as printed in Eq. (4.25). Select a case in ``Locals`` or run the file to
execute every case.
"""
from enum import Enum

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

import stochvolmodels as svm
from stochvolmodels.pricers.logsv_pricer import set_vol_scaler
from stochvolmodels.utils.funcs import set_seed

# parameters of the paper's figure code for Figs. 4 and 5; the captions cite Eq. (6.4) instead
FIG45_PARAMS = svm.LogSvParams(sigma0=0.8327, theta=1.0139, kappa1=4.8606, kappa2=4.7938,
                               beta=0.1985, volvol=2.3690)
# Fig. 6: Eq. (6.4) with the vol-of-vol scaled by 0.6, as in the paper's figure code
FIG6_PARAMS = svm.LogSvParams(sigma0=0.4083, theta=0.3789, kappa1=2.21, kappa2=2.18, beta=0.5010,
                              volvol=0.6 * 3.0633)
# the fitted parameters of the Bitcoin case study
BTC_PARAMS = svm.LogSvParams(sigma0=0.8626, theta=1.0418, kappa1=2.21, kappa2=2.18, beta=0.1296,
                             volvol=1.6286)


def coefficients(params: svm.LogSvParams, phi: complex, ttm: float = 1.0,
                 order: svm.ExpansionOrder = svm.ExpansionOrder.FIRST) -> np.ndarray:
    """A(tau) of Eq. (4.17) at first order or Eq. (4.25) at second order, under the MMA measure."""
    solution = svm.solve_ode_for_a(ttm=ttm, theta=params.theta, kappa1=params.kappa1,
                                   kappa2=params.kappa2, beta=params.beta, volvol=params.volvol,
                                   phi=phi, psi=0j, expansion_order=order, is_stiff_solver=True)
    return solution.y[:, -1]


def leading_term(params: svm.LogSvParams, a: np.ndarray) -> complex:
    """E^[m] = exp(sum_k A^(k) Y^k) at Y = sigma0 - theta, Eqs. (4.16) and (4.24)."""
    y = params.sigma0 - params.theta
    return np.exp(a @ y ** np.arange(len(a)))


def solve_system(params: svm.LogSvParams, phi: complex = 0j, psi: complex = 0j,
                 theta_t: complex = 0j, ttm: float = 1.0,
                 order: svm.ExpansionOrder = svm.ExpansionOrder.SECOND,
                 is_spot_measure: bool = True, linear_terms=None) -> np.ndarray:
    """Package matrices of Eq. (4.14) integrated at tight tolerance; A(0) = (0, -Theta, 0, ...)."""
    quadratic, linear, free = svm.func_a_ode_quadratic_terms(
        theta=params.theta, kappa1=params.kappa1, kappa2=params.kappa2, beta=params.beta,
        volvol=params.volvol, phi=phi, psi=psi, is_spot_measure=is_spot_measure,
        expansion_order=order)
    if linear_terms is not None:
        linear = linear_terms(params, phi=phi, is_spot_measure=is_spot_measure, n=len(free))
    a0 = np.zeros(len(free), dtype=np.complex128)
    a0[1] = -theta_t
    solution = solve_ivp(svm.func_rhs, (0.0, ttm), a0, args=(quadratic, linear, free),
                         rtol=1e-12, atol=1e-14)
    return solution.y[:, -1]


def paper_linear_terms(params: svm.LogSvParams, phi: complex, is_spot_measure: bool = True,
                       n: int = 5) -> np.ndarray:
    """L^(k)(p) from matching powers of Y in PDE (4.6); row k is the equation for A^(k)."""
    vartheta2 = params.beta ** 2 + params.volvol ** 2
    theta, beta = params.theta, params.beta
    kappa2_p = params.kappa2 if is_spot_measure else params.kappa2 - beta
    kappa_p = params.kappa1 - params.kappa2 * theta + 2.0 * kappa2_p * theta
    lamda = (params.kappa2 - kappa2_p) * theta ** 2
    linear = np.zeros((n, n), dtype=np.complex128)
    for k in range(n):
        for j in range(1, n):
            if j == k + 2:  # vol-of-vol term theta^2 Y^(j-2)
                linear[k, j] = 0.5 * vartheta2 * theta ** 2 * j * (j - 1)
            elif j == k + 1:
                linear[k, j] = (vartheta2 * theta * j * (j - 1)
                                + (lamda - theta ** 2 * beta * phi) * j)
            elif j == k:
                linear[k, j] = (0.5 * vartheta2 * j * (j - 1)
                                - (kappa_p + 2.0 * theta * beta * phi) * j)
            elif j == k - 1:
                linear[k, j] = -(kappa2_p + beta * phi) * j
    return linear


def expansion_moments(params: svm.LogSvParams, ttm: float, order: svm.ExpansionOrder,
                      linear_terms=None, h: float = 1e-3) -> dict:
    """Means and variances of sigma, X and I from derivatives of log E^[m] at zero."""
    y_powers = (params.sigma0 - params.theta) ** np.arange(svm.get_expansion_n(order))
    moments = {}
    for name in ("sigma", "X", "I"):
        values = []
        for z in (h, 0.0, -h):
            kwargs = {"sigma": {"theta_t": z}, "X": {"phi": z}, "I": {"psi": z}}[name]
            a = solve_system(params, ttm=ttm, order=order, linear_terms=linear_terms, **kwargs)
            values.append(np.real(a @ y_powers))
        mean = -(values[0] - values[2]) / (2.0 * h) + (params.theta if name == "sigma" else 0.0)
        moments[name] = (mean, (values[0] - 2.0 * values[1] + values[2]) / h ** 2)
    return moments


def exact_moments(params: svm.LogSvParams, ttm: float) -> dict:
    """Exact means and variances at kappa2 = 0, where the moments of weight 4 close linearly."""
    vartheta2 = params.beta ** 2 + params.volvol ** 2
    basis = [(a, b, c) for a in range(3) for b in range(3) for c in range(5)
             if 2 * a + 2 * b + c <= 4]
    index = {monomial: i for i, monomial in enumerate(basis)}
    generator = np.zeros((len(basis), len(basis)))
    for (a, b, c), row in index.items():  # generator applied to X^a I^b sigma^c
        diagonal = 0.5 * vartheta2 * c * (c - 1) - params.kappa1 * c
        for (da, db, dc), coefficient in (((-1, 0, 2), -0.5 * a), ((0, -1, 2), b),
                                         ((0, 0, -1), params.kappa1 * params.theta * c),
                                         ((0, 0, 0), diagonal),
                                         ((-2, 0, 2), 0.5 * a * (a - 1)),
                                         ((-1, 0, 1), params.beta * a * c)):
            target = (a + da, b + db, c + dc)
            if coefficient != 0.0 and target in index:
                generator[row, index[target]] += coefficient
    start = np.array([params.sigma0 ** c if a == b == 0 else 0.0 for a, b, c in basis])
    m = expm(generator * ttm) @ start
    first, second = {"sigma": (0, 0, 1), "X": (1, 0, 0), "I": (0, 1, 0)}, \
        {"sigma": (0, 0, 2), "X": (2, 0, 0), "I": (0, 2, 0)}
    return {name: (m[index[first[name]]], m[index[second[name]]] - m[index[first[name]]] ** 2)
            for name in first}


def implied_vols(params: svm.LogSvParams, ttm: float, strikes: np.ndarray,
                 linear_terms=None) -> np.ndarray:
    """Second-order implied volatilities under the MMA measure, from package or paper L terms."""
    phi_grid = svm.get_phi_grid(vol_scaler=set_vol_scaler(sigma0=params.sigma0, ttm=ttm))
    log_mgf = np.array([solve_system(params, phi=phi, ttm=ttm, linear_terms=linear_terms)
                        @ (params.sigma0 - params.theta) ** np.arange(5) for phi in phi_grid])
    optiontypes = np.where(strikes >= 1.0, "C", "P")
    prices = svm.vanilla_slice_pricer_with_mgf_grid(log_mgf_grid=log_mgf, phi_grid=phi_grid,
                                                    forward=1.0, strikes=strikes,
                                                    optiontypes=optiontypes)
    chain = svm.OptionChain.slice_to_chain(ttm=ttm, forward=1.0, strikes=strikes,
                                           optiontypes=optiontypes)
    return chain.compute_model_ivols_from_chain_data(model_prices=[prices])[0]


def monte_carlo_vols(params: svm.LogSvParams, ttm: float, strikes: np.ndarray,
                     nb_path: int = 400000, seed: int = 3) -> tuple:
    """Monte Carlo implied volatilities with the bounds of the 95% interval of the price."""
    optiontypes = np.where(strikes >= 1.0, "C", "P")
    chain = svm.OptionChain.slice_to_chain(ttm=ttm, forward=1.0, strikes=strikes,
                                           optiontypes=optiontypes)
    set_seed(seed)
    out = svm.LogSVPricer().compute_mc_chain_implied_vols(option_chain=chain, params=params,
                                                          nb_path=nb_path, nb_steps=360)
    return out[3][0], out[4][0], out[5][0]


def expansion_densities(params: svm.LogSvParams, ttm: float = 1.0 / 12.0, n: int = 200,
                        variable_types: tuple = (svm.VariableType.LOG_RETURN,)) -> dict:
    """First- and second-order densities on grids of 4.5 standard deviations, Eq. (5.5)."""
    pricer = svm.LogSVPricer()
    densities = {}
    for variable_type in variable_types:
        grid = params.get_variable_space_grid(variable_type=variable_type, ttm=ttm, n=n,
                                              n_stdevs=4.5)
        densities[variable_type] = (grid, *(
            pricer.logsv_pdfs(params=params, ttm=ttm, space_grid=grid, variable_type=variable_type,
                              expansion_order=order, is_stiff_solver=True)
            for order in (svm.ExpansionOrder.FIRST, svm.ExpansionOrder.SECOND)))
    return densities


def monte_carlo_histograms(params: svm.LogSvParams, densities: dict, ttm: float = 1.0 / 12.0,
                           nb_path: int = 400000, seed: int = 37) -> dict:
    """Probability mass of simulated X, I / tau and sigma in the cells of each density grid."""
    set_seed(seed)
    x, sigma, qvar = svm.LogSVPricer().simulate_terminal_values(params=params, ttm=ttm,
                                                                nb_path=nb_path)
    samples = {svm.VariableType.LOG_RETURN: x, svm.VariableType.Q_VAR: qvar / ttm,
               svm.VariableType.SIGMA: sigma}
    histograms = {}
    for variable_type, (grid, *_) in densities.items():
        step = grid[1] - grid[0]
        edges = np.append(grid - 0.5 * step, grid[-1] + 0.5 * step)
        histograms[variable_type] = np.histogram(samples[variable_type], bins=edges)[0] / nb_path
    return histograms


SLOW_CASES = ("PRICE_IMPACT",)


class Locals(Enum):
    """Cases of this script; each asserts the numbers the page quotes."""
    COEFFICIENTS = 1
    MARTINGALE_AND_LIMIT = 2
    EXACT_MOMENTS = 3
    SECOND_ORDER_TERMS = 4
    PRICE_IMPACT = 5
    DENSITIES = 6
    SEMI_ANALYTIC_PATH = 7


def run_local(local: Locals) -> None:
    """Run one case and assert its documented results."""
    if local == Locals.COEFFICIENTS:
        phi = -0.5 + 2.0j
        first = coefficients(FIG45_PARAMS, phi)
        second = coefficients(FIG45_PARAMS, phi, order=svm.ExpansionOrder.SECOND)
        np.testing.assert_allclose(first, [-1.8986 + 0.1060j, -0.3691 + 0.0216j, 0.0], atol=1e-4)
        np.testing.assert_allclose(second[:2], first[:2], atol=3e-3)
        assert np.max(np.abs(second[2:])) < 1e-4
        np.testing.assert_allclose(leading_term(FIG45_PARAMS, first), 0.1593 + 0.0163j, atol=1e-4)
        np.testing.assert_allclose(leading_term(FIG45_PARAMS, second), 0.1589 + 0.0162j,
                                   atol=1e-4)
        # the parameters of Eq. (6.4) give a different A^(0)(1): the published Fig. 4 used the above
        eq64 = svm.LogSvParams(sigma0=0.4083, theta=0.3789, kappa1=2.21, kappa2=2.18, beta=0.5010,
                               volvol=3.0633)
        np.testing.assert_allclose(coefficients(eq64, phi)[0], -0.3482 + 0.0517j, atol=1e-4)

    elif local == Locals.MARTINGALE_AND_LIMIT:
        # Proposition 4.3: H vanishes and A stays at zero for Phi = 0, for Phi = -1 with p = 1
        # and for Phi = 1 with p = -1
        for phi, is_spot_measure in ((0j, True), (-1.0 + 0j, True), (1.0 + 0j, False)):
            a = solve_system(BTC_PARAMS, phi=phi, is_spot_measure=is_spot_measure)
            assert np.max(np.abs(a)) < 1e-14
        # beta = volvol = 0 and sigma0 = theta: log-normal MGF exp(theta^2 tau (Phi^2 + Phi) / 2)
        flat = svm.LogSvParams(sigma0=0.4, theta=0.4, kappa1=2.0, kappa2=2.0, beta=0.0,
                               volvol=0.0)
        phi = -0.5 + 2.0j
        for order in (svm.ExpansionOrder.FIRST, svm.ExpansionOrder.SECOND):
            log_e = solve_system(flat, phi=phi, order=order)[0]
            np.testing.assert_allclose(log_e, 0.5 * 0.16 * (phi * phi + phi), atol=1e-10)
        np.testing.assert_allclose(0.5 * 0.16 * (phi * phi + phi), -0.34, atol=1e-12)

    elif local == Locals.EXACT_MOMENTS:
        params = svm.LogSvParams(sigma0=1.3, theta=1.0, kappa1=3.0, kappa2=0.0, beta=0.4,
                                 volvol=1.2)
        exact = exact_moments(params, ttm=0.75)
        np.testing.assert_allclose([exact[name][0] for name in ("sigma", "X", "I")],
                                   [1.0316, -0.5981, 1.1961], atol=1e-4)
        np.testing.assert_allclose([exact[name][1] for name in ("sigma", "X", "I")],
                                   [0.3995, 1.2301, 1.6427], atol=1e-4)
        errors = {}
        for label, kwargs in (("first", {"order": svm.ExpansionOrder.FIRST}),
                              ("second", {"order": svm.ExpansionOrder.SECOND}),
                              ("paper", {"order": svm.ExpansionOrder.SECOND,
                                         "linear_terms": paper_linear_terms})):
            moments = expansion_moments(params, ttm=0.75, **kwargs)
            for name in moments:  # every system reproduces the means
                np.testing.assert_allclose(moments[name][0], exact[name][0], atol=1e-5)
            errors[label] = [moments[name][1] / exact[name][1] - 1.0
                             for name in ("sigma", "X", "I")]
        np.testing.assert_allclose(errors["first"], [0.0, -0.089, -0.404], atol=5e-4)
        np.testing.assert_allclose(errors["second"], [0.0, -0.027, -0.081], atol=5e-4)
        np.testing.assert_allclose(errors["paper"], [0.0, 0.0, 0.0], atol=1e-4)
        print(errors)

    elif local == Locals.SECOND_ORDER_TERMS:
        vartheta2 = FIG45_PARAMS.beta ** 2 + FIG45_PARAMS.volvol ** 2
        for order, n in ((svm.ExpansionOrder.FIRST, 3), (svm.ExpansionOrder.SECOND, 5)):
            for phi, is_spot_measure in ((-0.5 + 2.0j, True), (0.5 + 2.0j, False)):
                _, linear, _ = svm.func_a_ode_quadratic_terms(
                    theta=FIG45_PARAMS.theta, kappa1=FIG45_PARAMS.kappa1,
                    kappa2=FIG45_PARAMS.kappa2, beta=FIG45_PARAMS.beta,
                    volvol=FIG45_PARAMS.volvol, phi=phi, psi=0j,
                    is_spot_measure=is_spot_measure, expansion_order=order)
                paper = paper_linear_terms(FIG45_PARAMS, phi=phi, is_spot_measure=is_spot_measure,
                                           n=n)
                differing = [tuple(int(i) for i in ij)
                             for ij in np.argwhere(np.abs(linear - paper) > 1e-12)]
                if n == 3:
                    expected = []
                else:
                    expected = [(4, 4)] if is_spot_measure else [(2, 3), (3, 4), (4, 4)]
                assert differing == expected, (order, is_spot_measure, differing)
                if n == 5:  # package: vartheta^2 in the A^(4) entry; (4.25): 3 vartheta^2
                    np.testing.assert_allclose(paper[4, 4] - linear[4, 4], 4.0 * vartheta2,
                                               atol=1e-12)
        # (4.17), entry of A^(2): vartheta^2 - 2 kappa - 4 theta beta Phi in code and derivation
        _, linear, _ = svm.func_a_ode_quadratic_terms(
            theta=FIG45_PARAMS.theta, kappa1=FIG45_PARAMS.kappa1, kappa2=FIG45_PARAMS.kappa2,
            beta=FIG45_PARAMS.beta, volvol=FIG45_PARAMS.volvol, phi=-0.5 + 2.0j, psi=0j)
        kappa = FIG45_PARAMS.kappa1 + FIG45_PARAMS.kappa2 * FIG45_PARAMS.theta
        np.testing.assert_allclose(linear[2, 2], vartheta2 - 2.0 * kappa - 4.0 * FIG45_PARAMS.theta
                                   * FIG45_PARAMS.beta * (-0.5 + 2.0j), atol=1e-12)

    elif local == Locals.PRICE_IMPACT:
        for ttm, gap in ((1.0 / 12.0, 5e-5), (1.0, 1.42e-2)):
            deviation = BTC_PARAMS.sigma0 * np.sqrt(ttm)
            strikes = np.exp(np.linspace(-2.5 * deviation, 2.5 * deviation, 11))
            package = implied_vols(BTC_PARAMS, ttm, strikes)
            paper = implied_vols(BTC_PARAMS, ttm, strikes, linear_terms=paper_linear_terms)
            print(ttm, np.max(np.abs(package - paper)))
            np.testing.assert_allclose(np.max(np.abs(package - paper)), gap, rtol=0.1)
        mc, upper, lower = monte_carlo_vols(BTC_PARAMS, 1.0, strikes)
        half_width = 0.5 * (upper - lower)
        for vols, rmse, inside in ((package, 0.0099, 6), (paper, 0.0017, 11)):
            print(np.sqrt(np.mean((vols - mc) ** 2)), np.sum(np.abs(vols - mc) <= half_width))
            np.testing.assert_allclose(np.sqrt(np.mean((vols - mc) ** 2)), rmse, atol=5e-4)
            assert np.sum(np.abs(vols - mc) <= half_width) == inside

    elif local == Locals.DENSITIES:
        densities = expansion_densities(FIG6_PARAMS)
        histograms = monte_carlo_histograms(FIG6_PARAMS, densities)
        grid, first, second = densities[svm.VariableType.LOG_RETURN]
        mc = histograms[svm.VariableType.LOG_RETURN]
        np.testing.assert_allclose([first.sum(), second.sum()], mc.sum(), atol=1e-4)
        # L1 distance to the histogram, and its expected size from sampling noise alone
        noise = np.sum(np.sqrt(2.0 * second * (1.0 - second) / (np.pi * 400000)))
        distances = np.abs(first - mc).sum(), np.abs(second - mc).sum()
        print(distances, noise)
        np.testing.assert_allclose(distances, [0.0138, 0.0125], atol=5e-4)
        np.testing.assert_allclose(noise, 0.0127, atol=5e-4)

    elif local == Locals.SEMI_ANALYTIC_PATH:
        # the fixed-point integrator overflows at large transform values of the default grid
        pricer = svm.LogSVPricer()
        params = svm.LogSvParams(sigma0=1.0, theta=1.0, kappa1=5.0, kappa2=5.0, beta=0.2,
                                 volvol=2.0)
        price, _ = pricer.price_vanilla(params=params, ttm=0.25, forward=1.0, strike=1.0,
                                        optiontype="C", is_analytic=True)
        assert not np.isfinite(price)
        price, _ = pricer.price_vanilla(params=params, ttm=0.25, forward=1.0, strike=1.0,
                                        optiontype="C")
        np.testing.assert_allclose(price, 0.197331, atol=1e-6)


if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
