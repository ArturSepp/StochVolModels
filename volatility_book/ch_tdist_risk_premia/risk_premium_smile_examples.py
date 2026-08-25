"""Illustrate risk-premium effects in the inverse-gamma Student-t mixture.

Under the physical measure,

    V ~ IG(alpha_P, beta_P),       X | V ~ N(0, c V).

The normalized terminal pricing kernel is

    Z(V, X) = Z_{p, eta}(V) exp(q X / c - q^2 V / (2 c)),

which gives, directly under the pricing measure,

    V ~ IG(alpha_P - p, beta_P + eta),
    X | V ~ N(q V, c V).

The terminal asset is S_T = (A + X)^+ with forward one.  The script solves A
separately in every scenario, prices calls by generalized Gauss-Laguerre
quadrature, converts them to Black implied volatilities, validates the numerical
prices independently, and writes the two vector figures used by the LaTeX note.

Run from the repository root with

    python volatility_book/ch_tdist_risk_premia/risk_premium_smile_examples.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import vanilla_option_pricers as bsm
from matplotlib.ticker import PercentFormatter
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gammaln, kve, ndtr, roots_genlaguerre
from scipy.stats import t as student_t

CHAPTER_DIR = Path(__file__).resolve().parent
DEFAULT_FIGURE_DIR = CHAPTER_DIR / "figures"


@dataclass(frozen=True)
class ModelSetup:
    """Physical baseline and numerical controls for the one-period example."""

    alpha_p: float = 4.0
    beta_p: float = 0.12
    c: float = 1.0
    ttm: float = 1.0
    forward: float = 1.0
    quadrature_order: int = 256

    @property
    def physical_mean_v(self) -> float:
        return self.beta_p / (self.alpha_p - 1.0)


@dataclass(frozen=True)
class Scenario:
    """One pricing-kernel parameter combination."""

    panel: str
    label: str
    p: float = 0.0
    eta: float = 0.0
    q: float = 0.0


@dataclass(frozen=True)
class ScenarioResult:
    """Smile and diagnostics for one scenario."""

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


def _normal_positive_part(mean: np.ndarray, stdev: np.ndarray) -> np.ndarray:
    """Return E[(mean + stdev Z)^+] elementwise, including a stable left tail."""
    d = mean / stdev
    h = np.empty_like(d, dtype=float)
    middle = np.abs(d) <= 8.0
    right = d > 8.0
    left = d < -8.0

    middle_d = d[middle]
    density = np.exp(-0.5 * middle_d * middle_d) / math.sqrt(2.0 * math.pi)
    h[middle] = density + middle_d * ndtr(middle_d)
    h[right] = d[right]

    z = -d[left]
    inverse_square = 1.0 / np.square(z)
    mills_remainder = inverse_square * (
        1.0
        + inverse_square
        * (
            -3.0
            + inverse_square
            * (15.0 + inverse_square * (-105.0 + 945.0 * inverse_square))
        )
    )
    h[left] = (
        np.exp(-0.5 * z * z)
        * mills_remainder
        / math.sqrt(2.0 * math.pi)
    )
    return np.maximum(stdev * h, 0.0)


class MixturePricer:
    """One-dimensional conditional-normal pricer under the transformed Q law."""

    def __init__(self, setup: ModelSetup, scenario: Scenario) -> None:
        self.setup = setup
        self.scenario = scenario
        self.alpha_q = setup.alpha_p - scenario.p
        self.beta_q = setup.beta_p + scenario.eta
        if self.alpha_q <= 1.0:
            raise ValueError("alpha_Q must exceed one for finite first moments")
        if self.beta_q <= 0.0:
            raise ValueError("beta_Q must be positive")

        nodes, weights = roots_genlaguerre(
            setup.quadrature_order,
            self.alpha_q - 1.0,
        )
        self.weights = weights / np.sum(weights)
        self.variances = self.beta_q / nodes
        self.stdevs = np.sqrt(setup.c * self.variances)
        self.conditional_means = scenario.q * self.variances
        self.shift = self._solve_shift()

    def expect_positive(self, intercept: float) -> float:
        means = intercept + self.conditional_means
        conditional_values = _normal_positive_part(means, self.stdevs)
        return float(np.dot(self.weights, conditional_values))

    def _solve_shift(self) -> float:
        def objective(shift: float) -> float:
            return self.expect_positive(shift) - self.setup.forward

        lower, upper = -2.0, 2.0
        while objective(lower) > 0.0:
            lower *= 2.0
        while objective(upper) < 0.0:
            upper *= 2.0
        return float(
            brentq(objective, lower, upper, xtol=1.0e-13, rtol=1.0e-13)
        )

    def call_price(self, strike: float) -> float:
        return self.expect_positive(self.shift - strike)

    def put_price(self, strike: float) -> float:
        raw_put = float(
            np.dot(
                self.weights,
                _normal_positive_part(
                    strike - self.shift - self.conditional_means,
                    self.stdevs,
                ),
            )
        )
        floor_adjustment = float(
            np.dot(
                self.weights,
                _normal_positive_part(
                    -self.shift - self.conditional_means,
                    self.stdevs,
                ),
            )
        )
        return raw_put - floor_adjustment

    def default_probability(self) -> float:
        d = (-self.shift - self.conditional_means) / self.stdevs
        return float(np.dot(self.weights, ndtr(d)))

    def mean_v(self) -> float:
        return self.beta_q / (self.alpha_q - 1.0)


def _direct_symmetric_student_call(
    strike: float,
    shift: float,
    alpha_q: float,
    beta_q: float,
    c: float,
) -> float:
    """Independent q=0 Student partial-moment formula for validation."""
    nu = 2.0 * alpha_q
    scale = math.sqrt(c * beta_q / alpha_q)
    threshold = (strike - shift) / scale
    survival = student_t.sf(threshold, df=nu)
    first_tail_moment = (
        (nu + threshold * threshold)
        * student_t.pdf(threshold, df=nu)
        / (nu - 1.0)
    )
    return (shift - strike) * survival + scale * first_tail_moment


def _skew_marginal_density(pricer: MixturePricer, shock: float) -> float:
    """Evaluate the q != 0 Bessel-K marginal density in stable log form."""
    q = pricer.scenario.q
    if abs(q) < 1.0e-15:
        raise ValueError("the skew marginal density requires non-zero q")

    alpha = pricer.alpha_q
    beta = pricer.beta_q
    c = pricer.setup.c
    radius_squared = shock * shock + 2.0 * beta * c
    radius = math.sqrt(radius_squared)
    bessel_argument = abs(q) * radius / c

    if q * shock >= 0.0:
        exponential_term = -2.0 * beta * abs(q) / (radius + abs(shock))
    else:
        exponential_term = -abs(q) * (radius + abs(shock)) / c

    log_density = (
        math.log(2.0)
        + alpha * math.log(beta)
        - gammaln(alpha)
        - 0.5 * math.log(2.0 * math.pi * c)
        + exponential_term
        + 0.5
        * (alpha + 0.5)
        * (math.log(q * q) - math.log(radius_squared))
        + math.log(kve(alpha + 0.5, bessel_argument))
    )
    return math.exp(log_density)


def _direct_skew_marginal_call(pricer: MixturePricer, strike: float) -> float:
    """Integrate the skew Bessel-K density, independently of conditional pricing."""
    threshold = strike - pricer.shift

    def integrand(shock: float) -> float:
        payoff = pricer.shift + shock - strike
        return max(payoff, 0.0) * _skew_marginal_density(pricer, shock)

    value = 0.0
    if threshold < 0.0:
        value += quad(
            integrand,
            threshold,
            0.0,
            epsabs=2.0e-11,
            epsrel=2.0e-11,
            limit=300,
        )[0]
        threshold = 0.0
    value += quad(
        integrand,
        threshold,
        np.inf,
        epsabs=2.0e-11,
        epsrel=2.0e-11,
        limit=300,
    )[0]
    return float(value)


def _adaptive_call(pricer: MixturePricer, strike: float) -> float:
    """Independent adaptive integral in the Gamma precision variable."""
    intercept = pricer.shift - strike

    def integrand(precision: float) -> float:
        if precision <= 0.0:
            return 0.0
        variance = pricer.beta_q / precision
        stdev = math.sqrt(pricer.setup.c * variance)
        mean = intercept + pricer.scenario.q * variance
        positive_part = float(
            _normal_positive_part(np.array([mean]), np.array([stdev]))[0]
        )
        if positive_part == 0.0:
            return 0.0
        log_density = (
            (pricer.alpha_q - 1.0) * math.log(precision)
            - precision
            - gammaln(pricer.alpha_q)
        )
        return math.exp(log_density) * positive_part

    lower = quad(
        integrand,
        0.0,
        1.0,
        epsabs=2.0e-11,
        epsrel=2.0e-11,
        limit=300,
    )[0]
    upper = quad(
        integrand,
        1.0,
        np.inf,
        epsabs=2.0e-11,
        epsrel=2.0e-11,
        limit=300,
    )[0]
    return float(lower + upper)


def _implied_volatility(
    setup: ModelSetup,
    strike: float,
    call_price: float,
) -> float:
    return float(
        bsm.infer_bsm_implied_vol(
            forward=setup.forward,
            ttm=setup.ttm,
            strike=strike,
            given_price=call_price,
            discfactor=1.0,
            optiontype="C",
            tol=1.0e-12,
            vol_lower=1.0e-8,
            vol_upper=5.0,
            is_bounds_to_nan=False,
        )
    )


def compute_scenario(
    setup: ModelSetup,
    scenario: Scenario,
    log_moneyness: np.ndarray,
) -> tuple[ScenarioResult, MixturePricer]:
    """Price one scenario and calculate smile diagnostics."""
    pricer = MixturePricer(setup, scenario)
    strikes = setup.forward * np.exp(log_moneyness)
    call_prices = np.array([pricer.call_price(float(strike)) for strike in strikes])
    implied_volatilities = np.array(
        [
            _implied_volatility(setup, float(strike), float(price))
            for strike, price in zip(strikes, call_prices)
        ]
    )

    atm_index = int(np.argmin(np.abs(log_moneyness)))
    step = float(log_moneyness[atm_index + 1] - log_moneyness[atm_index])
    atm_skew = float(
        (
            implied_volatilities[atm_index + 1]
            - implied_volatilities[atm_index - 1]
        )
        / (2.0 * step)
    )
    left_index = int(np.argmin(np.abs(log_moneyness + 0.25)))
    right_index = int(np.argmin(np.abs(log_moneyness - 0.25)))
    rr_025 = float(
        implied_volatilities[right_index] - implied_volatilities[left_index]
    )
    bf_025 = float(
        0.5
        * (
            implied_volatilities[left_index]
            + implied_volatilities[right_index]
        )
        - implied_volatilities[atm_index]
    )

    result = ScenarioResult(
        scenario=scenario,
        alpha_q=pricer.alpha_q,
        beta_q=pricer.beta_q,
        mean_v=pricer.mean_v(),
        shift=pricer.shift,
        default_probability=pricer.default_probability(),
        log_moneyness=log_moneyness,
        strikes=strikes,
        call_prices=call_prices,
        implied_volatilities=implied_volatilities,
        atm_iv=float(implied_volatilities[atm_index]),
        atm_skew=atm_skew,
        rr_025=rr_025,
        bf_025=bf_025,
    )
    return result, pricer


def _raw_scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario("p", "p = -0.75", p=-0.75),
        Scenario("p", "p = 0"),
        Scenario("p", "p = +0.75", p=0.75),
        Scenario("eta", "eta = -0.024", eta=-0.024),
        Scenario("eta", "eta = 0"),
        Scenario("eta", "eta = +0.024", eta=0.024),
        Scenario("q", "q = -2", q=-2.0),
        Scenario("q", "q = 0"),
        Scenario("q", "q = +2", q=2.0),
    )


def _fixed_variance_p_scenarios(setup: ModelSetup) -> tuple[Scenario, ...]:
    mean_v = setup.physical_mean_v
    return tuple(
        Scenario(
            "p_fixed_variance",
            (
                "p = 0, eta = 0"
                if abs(p) < 1.0e-15
                else f"p = {p:+.2f}, eta = {-mean_v * p:+.3f}"
            ),
            p=p,
            eta=-mean_v * p,
        )
        for p in (-0.75, 0.0, 0.75)
    )


def compute_examples(
    setup: ModelSetup,
) -> tuple[list[ScenarioResult], list[ScenarioResult], dict[str, float]]:
    """Compute raw and variance-neutral examples and all validation diagnostics."""
    log_moneyness = np.linspace(-0.35, 0.35, 29)
    scenarios = _raw_scenarios() + _fixed_variance_p_scenarios(setup)
    results: list[ScenarioResult] = []
    pricers: list[MixturePricer] = []
    for scenario in scenarios:
        result, pricer = compute_scenario(setup, scenario, log_moneyness)
        results.append(result)
        pricers.append(pricer)

    martingale_errors: list[float] = []
    independent_martingale_errors: list[float] = []
    parity_errors: list[float] = []
    symmetric_errors: list[float] = []
    adaptive_errors: list[float] = []
    marginal_density_errors: list[float] = []
    black_errors: list[float] = []
    convergence_errors: list[float] = []
    monotonicity_violations: list[float] = []
    convexity_violations: list[float] = []

    for result, pricer in zip(results, pricers):
        martingale_errors.append(abs(pricer.call_price(0.0) - setup.forward))
        if abs(result.scenario.q) < 1.0e-15:
            independent_forward = _direct_symmetric_student_call(
                0.0,
                pricer.shift,
                pricer.alpha_q,
                pricer.beta_q,
                setup.c,
            )
        else:
            independent_forward = _direct_skew_marginal_call(pricer, 0.0)
        independent_martingale_errors.append(
            abs(independent_forward - setup.forward)
        )
        monotonicity_violations.append(
            float(max(0.0, np.max(np.diff(result.call_prices))))
        )
        slopes = np.diff(result.call_prices) / np.diff(result.strikes)
        convexity_violations.append(float(max(0.0, -np.min(np.diff(slopes)))))
        for strike, call, volatility in zip(
            result.strikes,
            result.call_prices,
            result.implied_volatilities,
        ):
            put = pricer.put_price(float(strike))
            parity_errors.append(abs(call - put - (setup.forward - strike)))
            repriced = bsm.compute_bsm_vanilla_price(
                forward=setup.forward,
                strike=float(strike),
                ttm=setup.ttm,
                vol=float(volatility),
                optiontype="C",
                discfactor=1.0,
            )
            black_errors.append(abs(float(repriced) - call))
            if abs(result.scenario.q) < 1.0e-15:
                reference = _direct_symmetric_student_call(
                    float(strike),
                    pricer.shift,
                    pricer.alpha_q,
                    pricer.beta_q,
                    setup.c,
                )
                symmetric_errors.append(abs(float(call) - reference))

        if abs(result.scenario.q) > 1.0e-15:
            for strike in (math.exp(-0.25), 1.0, math.exp(0.25)):
                adaptive_errors.append(
                    abs(pricer.call_price(strike) - _adaptive_call(pricer, strike))
                )
                marginal_density_errors.append(
                    abs(
                        pricer.call_price(strike)
                        - _direct_skew_marginal_call(pricer, strike)
                    )
                )

        lower_order = MixturePricer(
            replace(setup, quadrature_order=128),
            result.scenario,
        )
        convergence_errors.extend(
            abs(pricer.call_price(float(strike)) - lower_order.call_price(float(strike)))
            for strike in (math.exp(-0.25), 1.0, math.exp(0.25))
        )

    validation = {
        "max_martingale_error": max(martingale_errors),
        "max_independent_martingale_error": max(independent_martingale_errors),
        "max_put_call_parity_error": max(parity_errors),
        "max_symmetric_closed_form_error": max(symmetric_errors),
        "max_adaptive_integration_error": max(adaptive_errors),
        "max_marginal_density_integration_error": max(marginal_density_errors),
        "max_black_round_trip_error": max(black_errors),
        "max_128_vs_256_node_error": max(convergence_errors),
        "max_call_monotonicity_violation": max(monotonicity_violations),
        "max_call_convexity_violation": max(convexity_violations),
    }
    fixed_results = results[9:]
    validation["max_variance_neutral_mean_v_error"] = max(
        abs(result.mean_v - setup.physical_mean_v) for result in fixed_results
    )
    baseline_results = [
        result
        for result in results
        if abs(result.scenario.p) < 1.0e-15
        and abs(result.scenario.eta) < 1.0e-15
        and abs(result.scenario.q) < 1.0e-15
    ]
    baseline_curve = baseline_results[0].implied_volatilities
    validation["max_duplicate_baseline_curve_error"] = max(
        float(np.max(np.abs(result.implied_volatilities - baseline_curve)))
        for result in baseline_results[1:]
    )
    return results[:9], results[9:], validation


def _enforce_validation(validation: dict[str, float]) -> None:
    thresholds = {
        "max_martingale_error": 1.0e-10,
        "max_independent_martingale_error": 1.0e-7,
        "max_put_call_parity_error": 1.0e-10,
        "max_symmetric_closed_form_error": 5.0e-8,
        "max_adaptive_integration_error": 5.0e-8,
        "max_marginal_density_integration_error": 1.0e-7,
        "max_black_round_trip_error": 1.0e-10,
        "max_128_vs_256_node_error": 3.0e-7,
        "max_call_monotonicity_violation": 1.0e-12,
        "max_call_convexity_violation": 1.0e-12,
        "max_variance_neutral_mean_v_error": 1.0e-13,
        "max_duplicate_baseline_curve_error": 1.0e-13,
    }
    failures = {
        name: (validation[name], threshold)
        for name, threshold in thresholds.items()
        if validation[name] > threshold
    }
    if failures:
        details = ", ".join(
            f"{name}={value:.3e}>{threshold:.3e}"
            for name, (value, threshold) in failures.items()
        )
        raise RuntimeError(f"risk-premium example validation failed: {details}")


def _configure_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )


def _legend_label(scenario: Scenario) -> str:
    def signed(value: float, digits: int) -> str:
        return "0" if abs(value) < 1.0e-15 else f"{value:+.{digits}f}"

    if scenario.panel == "p":
        return rf"$p={signed(scenario.p, 2)}$"
    if scenario.panel == "eta":
        return rf"$\eta={signed(scenario.eta, 3)}$"
    if scenario.panel == "q":
        return rf"$q={signed(scenario.q, 0)}$"
    return (
        rf"$p={signed(scenario.p, 2)},\ "
        rf"\eta={signed(scenario.eta, 3)}$"
    )


def _save_raw_figure(results: list[ScenarioResult], output_path: Path) -> None:
    _configure_figure_style()
    colors = ("#2878B5", "#F28E2B", "#3AA255")
    linestyles = ("-", "--", ":")
    panel_specs = (
        ("p", r"$p$: tails and variance"),
        ("eta", r"$\eta$: variance level"),
        ("q", r"$q$: skew direction"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(8.4, 2.9), sharex=True, sharey=True)
    for axis, (panel, title) in zip(axes, panel_specs):
        panel_results = [result for result in results if result.scenario.panel == panel]
        for result, color, linestyle in zip(panel_results, colors, linestyles):
            axis.plot(
                result.log_moneyness,
                result.implied_volatilities,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                label=_legend_label(result.scenario),
            )
        axis.axvline(0.0, color="0.75", linewidth=0.7, zorder=0)
        axis.grid(color="0.88", linewidth=0.6)
        axis.set_title(title)
        axis.set_xlabel(r"log-moneyness $k=\log(K/F)$")
        axis.legend(frameon=False, loc="upper right")
    axes[0].set_ylabel("Black implied volatility")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_ylim(0.155, 0.285)
    figure.tight_layout(w_pad=1.0)
    figure.savefig(
        output_path,
        bbox_inches="tight",
        metadata={"Title": "Risk-premium parameters and the Black implied-volatility smile"},
    )
    plt.close(figure)


def _save_fixed_variance_figure(
    results: list[ScenarioResult],
    output_path: Path,
) -> None:
    _configure_figure_style()
    colors = ("#2878B5", "#F28E2B", "#3AA255")
    linestyles = ("-", "--", ":")
    figure, axis = plt.subplots(figsize=(5.8, 3.45))
    for result, color, linestyle in zip(results, colors, linestyles):
        axis.plot(
            result.log_moneyness,
            result.implied_volatilities,
            color=color,
            linestyle=linestyle,
            linewidth=1.9,
            label=_legend_label(result.scenario),
        )
    axis.axvline(0.0, color="0.75", linewidth=0.7, zorder=0)
    axis.grid(color="0.88", linewidth=0.6)
    axis.set_title(r"Tail premium $p$ with $E_Q[V]=4\%$ held fixed")
    axis.set_xlabel(r"log-moneyness $k=\log(K/F)$")
    axis.set_ylabel("Black implied volatility")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.legend(frameon=False, loc="upper right")
    figure.tight_layout()
    figure.savefig(
        output_path,
        bbox_inches="tight",
        metadata={"Title": "Tail-risk premium at fixed mean variance"},
    )
    plt.close(figure)


def _print_results(
    raw_results: list[ScenarioResult],
    fixed_results: list[ScenarioResult],
    validation: dict[str, float],
) -> None:
    print(
        "scenario                E[V]       A        p0       ATM       skew     RRk.25    BFk.25"
    )
    print("-" * 98)
    for result in raw_results + fixed_results:
        print(
            f"{result.scenario.label:<23} "
            f"{result.mean_v:>7.4f} "
            f"{result.shift:>8.5f} "
            f"{result.default_probability:>8.2e} "
            f"{result.atm_iv:>8.4f} "
            f"{result.atm_skew:>9.4f} "
            f"{result.rr_025:>9.4f} "
            f"{result.bf_025:>9.4f}"
        )
    print("\nvalidation")
    for name, value in validation.items():
        print(f"{name}: {value:.6e}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Student-t risk-premium implied-volatility figures."
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Output directory for the two PDF figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    setup = ModelSetup()
    raw_results, fixed_results, validation = compute_examples(setup)
    _enforce_validation(validation)
    _save_raw_figure(raw_results, args.figure_dir / "risk_premium_smiles.pdf")
    _save_fixed_variance_figure(
        fixed_results,
        args.figure_dir / "p_tail_premium_fixed_variance.pdf",
    )
    _print_results(raw_results, fixed_results, validation)


if __name__ == "__main__":
    main()
