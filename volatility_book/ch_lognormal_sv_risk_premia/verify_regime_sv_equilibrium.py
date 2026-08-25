r"""Independent checks for the regime-switching LogSV equilibrium extension.

Run the default verification with::

    C:\Python\StochVolModels312\Scripts\python.exe verify_regime_sv_equilibrium.py

Use ``--full`` for the report-quality Monte Carlo sample.  The checks deliberately
combine algebraic identities, an exact frozen-volatility benchmark, physical-measure
Feynman--Kac Monte Carlo, and induced-risk-neutral option Monte Carlo.  Both the
log-linear/quadratic-Q and log-quadratic/full-cubic-Q equilibrium closures are checked.
"""

from __future__ import annotations

import argparse

import numpy as np
from regime_switch_logsv import (
    GROWTH,
    STRESS,
    RegimeSpec,
    RegimeSwitchLogSvParams,
    _equilibrium_rhs,
    compute_log_mgf_grid,
    mc_price_slice,
    price_slice,
    risk_neutral_state,
    simulate_equilibrium_feynman_kac,
    simulate_terminal_q,
    single_regime_closed_form,
    solve_equilibrium,
)
from scipy.linalg import expm


def _require(label: str, error: float, tolerance: float) -> None:
    status = "PASS" if error <= tolerance else "FAIL"
    print(f"{status:4s} {label:48s} error={error:.6g}  tolerance={tolerance:.6g}")
    if error > tolerance:
        raise AssertionError(f"{label}: error {error:.6g} exceeds {tolerance:.6g}")


def verify_jump_calibration() -> None:
    params = RegimeSwitchLogSvParams.equity_baseline()
    expected = np.array([-0.25, 0.15])
    observed = np.array([params.jump_compensator(i) for i in (GROWTH, STRESS)])
    _require("paper arithmetic jump means", np.max(np.abs(observed - expected)), 1.0e-14)
    _require(
        "paper transition intensities",
        np.max(np.abs(np.asarray(params.transition_intensities) - [0.1, 1.0])),
        1.0e-14,
    )


def verify_scalar_closed_form() -> None:
    """Differentiate the closed form and substitute it into both Riccati ODEs."""

    spec = RegimeSpec(theta=0.2, kappa1=2.7, kappa2=6.0, beta=-0.8, volvol=0.6)
    gamma = -0.5
    horizons = np.linspace(0.01, 3.0, 80)
    step = 1.0e-5
    a0, a1 = single_regime_closed_form(horizons, spec, gamma)
    a0_up, a1_up = single_regime_closed_form(horizons + step, spec, gamma)
    a0_down, a1_down = single_regime_closed_form(horizons - step, spec, gamma)
    derivative0 = (a0_up - a0_down) / (2.0 * step)
    derivative1 = (a1_up - a1_down) / (2.0 * step)
    rhs0 = 0.5 * spec.vartheta2 * spec.theta**2 * a1**2
    rhs0 += 0.5 * spec.theta**2 * gamma * (1.0 - gamma)
    rhs1 = spec.vartheta2 * spec.theta * a1**2 - spec.kappa_bar * a1
    rhs1 += spec.theta * gamma * (1.0 - gamma)
    residual = max(np.max(np.abs(derivative0 - rhs0)), np.max(np.abs(derivative1 - rhs1)))
    _require("single-regime closed-form Riccati residual", float(residual), 2.0e-9)


def verify_frozen_volatility_exact_solution() -> None:
    """Compare the coupled solution with a two-state matrix exponential."""

    frozen = RegimeSpec(theta=0.2, kappa1=2.0, kappa2=2.0, beta=0.0, volvol=0.0)
    params = RegimeSwitchLogSvParams(
        sigma0=0.2,
        regimes=(frozen, frozen),
        transition_intensities=(0.1, 1.0),
        jump_means=(0.25 / 0.75, 0.15 / 1.15),
        gamma=-0.5,
        agent_horizon=3.0,
    )
    potential = 0.5 * params.gamma * (1.0 - params.gamma) * params.sigma0**2
    matrix = np.array(
        [
            [potential - 0.1, 0.1 * params.transition_factor(GROWTH)],
            [1.0 * params.transition_factor(STRESS), potential - 1.0],
        ]
    )
    for degree, label in ((1, "log-linear"), (2, "log-quadratic")):
        solution = solve_equilibrium(params, degree=degree)
        largest = 0.0
        for horizon in (0.25, 1.0, 3.0):
            exact = expm(matrix * horizon) @ np.ones(2)
            approximate = np.array(
                [
                    np.exp(solution.log_g_hat(horizon, params.sigma0, GROWTH)),
                    np.exp(solution.log_g_hat(horizon, params.sigma0, STRESS)),
                ]
            )
            largest = max(largest, float(np.max(np.abs(exact - approximate))))
        _require(f"frozen-volatility matrix solution ({label})", largest, 2.0e-10)


def verify_degree_one_equilibrium_odes() -> None:
    """Check the four explicit log-linear equations against the generic projection."""

    params = RegimeSwitchLogSvParams.equity_baseline(gamma=-0.5, agent_horizon=3.0)
    coefficients = np.array([[0.021, -0.037], [-0.014, 0.029]])
    generic = _equilibrium_rhs(0.7, coefficients.ravel(), params, 1).reshape(2, 2)
    explicit = np.zeros_like(generic)
    qvar_loading = 0.5 * params.gamma * (1.0 - params.gamma)
    for regime, spec in enumerate(params.regimes):
        other = 1 - regime
        delta = spec.theta - params.regimes[other].theta
        d0 = coefficients[other, 0] + coefficients[other, 1] * delta
        d0 -= coefficients[regime, 0]
        d1 = coefficients[other, 1] - coefficients[regime, 1]
        a1 = coefficients[regime, 1]
        transition = params.transition_intensities[regime]
        coupling = transition * params.transition_factor(regime) * np.exp(d0)
        explicit[regime, 0] = 0.5 * spec.vartheta2 * spec.theta**2 * a1**2
        explicit[regime, 0] += qvar_loading * spec.theta**2
        explicit[regime, 0] += coupling - transition
        explicit[regime, 1] = -spec.kappa_bar * a1
        explicit[regime, 1] += spec.vartheta2 * spec.theta * a1**2
        explicit[regime, 1] += 2.0 * qvar_loading * spec.theta
        explicit[regime, 1] += coupling * d1
    _require(
        "explicit degree-one equilibrium ODEs",
        float(np.max(np.abs(generic - explicit))),
        2.0e-14,
    )


def verify_risk_neutral_drift_degrees() -> None:
    """Verify that degree one is quadratic and degree two has the derived cubic."""

    params = RegimeSwitchLogSvParams.equity_baseline(gamma=-0.5, agent_horizon=3.0)
    linear = solve_equilibrium(params, degree=1)
    quadratic = solve_equilibrium(params, degree=2)
    sigma_grid = np.linspace(0.08, 0.36, 31)
    for regime, spec in enumerate(params.regimes):
        drift_linear, *_ = risk_neutral_state(
            params, linear, params.agent_horizon, sigma_grid, regime
        )
        a1 = linear.coefficients(params.agent_horizon)[regime, 1]
        expected_linear = (spec.kappa1 + spec.kappa2 * sigma_grid) * (
            spec.theta - sigma_grid
        )
        expected_linear += (
            -spec.beta * (1.0 - params.gamma) + spec.vartheta2 * a1
        ) * sigma_grid**2
        _require(
            f"quadratic Q drift identity: regime={regime + 1}",
            float(np.max(np.abs(drift_linear - expected_linear))),
            2.0e-14,
        )

        drift_quadratic, *_ = risk_neutral_state(
            params, quadratic, params.agent_horizon, sigma_grid, regime
        )
        v_grid = sigma_grid - spec.theta
        fitted = np.polynomial.polynomial.polyfit(v_grid, drift_quadratic, 3)
        a2 = quadratic.coefficients(params.agent_horizon)[regime, 2]
        expected_cubic = 2.0 * spec.vartheta2 * a2
        _require(
            f"full cubic Q-drift coefficient: regime={regime + 1}",
            float(abs(fitted[3] - expected_cubic)),
            2.0e-10,
        )


def verify_equilibrium_monte_carlo(full: bool) -> None:
    params = RegimeSwitchLogSvParams.equity_baseline(agent_horizon=3.0)
    solutions = {
        "log-linear": solve_equilibrium(params, degree=1),
        "log-quadratic": solve_equilibrium(params, degree=2),
    }
    n_paths = 120_000 if full else 40_000
    steps_per_year = 1_440 if full else 720
    for horizon in (1.0, 3.0):
        for regime in (GROWTH, STRESS):
            sigma = params.regimes[regime].theta
            monte_carlo, standard_error = simulate_equilibrium_feynman_kac(
                params,
                horizon,
                sigma,
                regime,
                n_paths=n_paths,
                steps_per_year=steps_per_year,
                seed=181 + 17 * regime + int(10 * horizon),
            )
            for label, solution in solutions.items():
                analytic = np.exp(solution.log_g_hat(horizon, sigma, regime))
                error = abs(analytic - monte_carlo)
                relative_closure_allowance = 5.0e-3 if label == "log-linear" else 1.5e-3
                tolerance = 5.0 * standard_error + relative_closure_allowance * monte_carlo
                _require(
                    f"P-FK MC ({label}): h={horizon:g}, regime={regime + 1}",
                    error,
                    tolerance,
                )


def _empirical_mgf(log_return: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.empty(phi.size, dtype=complex)
    errors = np.empty(phi.size)
    for index, transform in enumerate(phi):
        values = np.exp(-transform * log_return)
        means[index] = np.mean(values)
        errors[index] = np.sqrt(np.mean(np.abs(values - means[index]) ** 2) / values.size)
    return means, errors


def verify_option_monte_carlo(full: bool) -> None:
    ttm = 0.25
    strikes = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    optiontypes = np.where(strikes < 1.0, "P", "C")
    n_paths = 160_000 if full else 50_000
    steps_per_year = 1_440 if full else 720
    phi = np.array([-0.5 + 1j * value for value in (0.0, 1.0, 2.0, 5.0, 10.0)])

    for equilibrium_degree, closure_label in ((1, "quadratic Q"), (2, "full cubic Q")):
        for regime in (GROWTH, STRESS):
            params = RegimeSwitchLogSvParams.equity_baseline(
                gamma=-0.5,
                initial_regime=regime,
                agent_horizon=3.0,
            )
            equilibrium = solve_equilibrium(params, degree=equilibrium_degree)
            invariant = compute_log_mgf_grid(
                params, equilibrium, ttm, np.array([0.0, -1.0]), degree=4
            )
            _require(
                f"MGF identities ({closure_label}): regime={regime + 1}",
                float(np.max(np.abs(invariant))),
                2.0e-11,
            )
            analytic = price_slice(
                params,
                equilibrium,
                ttm,
                strikes,
                optiontypes=optiontypes,
                degree=4,
                max_phi=1_601,
            )
            refined = price_slice(
                params,
                equilibrium,
                ttm,
                strikes,
                optiontypes=optiontypes,
                degree=4,
                max_phi=3_201,
            )
            _require(
                f"Fourier refinement ({closure_label}): regime={regime + 1}",
                float(np.max(np.abs(analytic.prices - refined.prices))),
                5.0e-6,
            )

            seed = 911 + regime + 100 * equilibrium_degree
            sample = simulate_terminal_q(
                params,
                equilibrium,
                ttm,
                n_paths=n_paths,
                steps_per_year=steps_per_year,
                seed=seed,
            )
            monte_carlo, standard_errors, _ = mc_price_slice(
                sample, params, ttm, strikes, optiontypes
            )
            price_errors = np.abs(analytic.prices - monte_carlo)
            price_tolerances = 5.0 * standard_errors + 1.5e-4
            _require(
                f"Q option MC ({closure_label}): regime={regime + 1}",
                float(np.max(price_errors / price_tolerances)),
                1.0,
            )

            martingale, martingale_error = sample.forward_martingale
            _require(
                f"forward martingale ({closure_label}): regime={regime + 1}",
                abs(martingale - 1.0),
                5.0 * martingale_error + 5.0e-4,
            )

            analytic_mgf = np.exp(
                compute_log_mgf_grid(params, equilibrium, ttm, phi, degree=4)
            )
            monte_carlo_mgf, mgf_errors = _empirical_mgf(sample.log_forward_return, phi)
            normalized = np.abs(analytic_mgf - monte_carlo_mgf) / (
                5.0 * mgf_errors + 1.0e-3
            )
            _require(
                f"complex MGF MC ({closure_label}): regime={regime + 1}",
                float(np.max(normalized)),
                1.0,
            )

            if regime == GROWTH:
                coarse_sample = simulate_terminal_q(
                    params,
                    equilibrium,
                    ttm,
                    n_paths=n_paths,
                    steps_per_year=max(1, steps_per_year // 2),
                    seed=seed + 1_000,
                )
                coarse_prices, coarse_errors, _ = mc_price_slice(
                    coarse_sample, params, ttm, strikes, optiontypes
                )
                atm = int(np.argmin(np.abs(strikes - 1.0)))
                refinement_error = abs(coarse_prices[atm] - monte_carlo[atm])
                refinement_tolerance = (
                    5.0 * np.hypot(coarse_errors[atm], standard_errors[atm]) + 2.0e-4
                )
                _require(
                    f"Q MC time-step ({closure_label}): growth near-ATM",
                    float(refinement_error),
                    float(refinement_tolerance),
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="use report-quality path counts and a finer time grid",
    )
    args = parser.parse_args()
    verify_jump_calibration()
    verify_scalar_closed_form()
    verify_frozen_volatility_exact_solution()
    verify_degree_one_equilibrium_odes()
    verify_risk_neutral_drift_degrees()
    verify_equilibrium_monte_carlo(args.full)
    verify_option_monte_carlo(args.full)
    print("All regime-switching LogSV verification checks passed.")


if __name__ == "__main__":
    main()
