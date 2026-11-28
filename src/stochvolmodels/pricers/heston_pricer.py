"""Heston transform pricing, calibration and the maintained Monte Carlo kernel.

The model state is log forward return, annualized variance and integrated variance.
Chain pricing combines that law with forwards, discounts and option payoffs. The
private increment-driven simulator is shared with the path adapter in ``models.heston``;
legacy callers retain their seeded draw ordering and historical variance floor.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Literal, Tuple

import numpy as np
from scipy.optimize import minimize
from numba import njit
from numba.typed import List

# stochvolmodels
from stochvolmodels.utils.funcs import to_flat_np_array, set_time_grid, timer
from stochvolmodels.utils.config import VariableType
from stochvolmodels.utils import _fourier_quadrature as quadrature
from stochvolmodels.utils.mc_payoffs import compute_conditional_mc_payoff, compute_mc_vars_payoff
from stochvolmodels.pricers._fourier_calibration import (
    FourierCalibration, validate_calibration_options,
)
from stochvolmodels.pricers._prepared_fourier import _PreparedFourierGrid, _VanillaQuotes
from stochvolmodels.pricers.model_pricer import (
    ModelPricer,
    validate_optimization_result,
)
import stochvolmodels.utils.mgf_pricer as mgfp

# data
from stochvolmodels.data.option_chain import OptionChain


from stochvolmodels.models.heston.model import HestonModel, _real
from stochvolmodels.models.heston.params import HestonParams as HestonParams


BTC_HESTON_PARAMS = HestonParams(v0=0.8, theta=1.0, kappa=2.0, rho=0.0, volvol=2.0)


def _qe_paths(params, ttms, count, *, seed=None, steps_per_year=360, is_spot_measure=True):
    """Run one zero-drift QE-M path set for all requested observations."""
    if not is_spot_measure:
        raise ValueError("Heston QE-M supports the money-market measure")
    times = np.asarray(ttms, dtype=float)
    if (times.ndim != 1 or not times.size or not np.isfinite(times).all()
            or times[0] < 0 or np.any(np.diff(times) <= 0)):
        raise ValueError("QE-M requires increasing nonnegative finite maturities")
    observations = times if times[0] == 0.0 else np.r_[0.0, times]
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    return HestonModel(params).simulate_paths(
        measure="Q_MMA", observation_times=observations, spot0=1., n_paths=count,
        steps_per_year=steps_per_year, seed=seed, mc_scheme="qe_m",
    )


def _controlled_heston_chain(option_chain: OptionChain, params: HestonParams,
                             fourier: Mapping[str, object]) -> List[np.ndarray]:
    """Price each expiry from time zero on its own grid, sharing transforms across strikes."""
    settings = quadrature.vanilla_settings_by_expiry(fourier, option_chain.strikes_ttms)
    model = HestonModel(params)
    prices_ttms = List()
    for i, ttm in enumerate(option_chain.ttms):
        _, prices, _ = quadrature.integrate_vanilla_slice(
            lambda phi, maturity=ttm: model.log_mgf_grid(ttm=maturity, phi_grid=phi),
            maturity=ttm, forward=option_chain.forwards[i], discfactor=option_chain.discfactors[i],
            strikes=option_chain.strikes_ttms[i], optiontypes=option_chain.optiontypes_ttms[i],
            **settings[i],
        )
        prices_ttms.append(prices)
    return prices_ttms


@dataclass(frozen=True, eq=False, repr=False)
class _HestonQuotes(_VanillaQuotes):
    """Vanilla premiums or derivatives at one initial annualized variance."""

    v0: float

    def same_output(self, other):
        """A certificate applies to one initial variance and one output quantity."""
        return self.v0 == other.v0 and self.quantity == other.quantity

    def evaluate(self, coefficients, nodes):
        """Return log M for premiums or the differentiated transform numerator for risk."""
        intercept, loading = coefficients.T
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            log_mgf = intercept + self.v0 * loading
            if self.quantity == "state":
                return loading * np.exp(log_mgf)
        return super().evaluate(log_mgf, nodes)


@dataclass(frozen=True, eq=False, repr=False)
class PreparedHestonGrid(_PreparedFourierGrid):
    """Caller-owned affine coefficients for Heston prices and sensitivities.

    Construct with HestonPricer.prepare_fourier. All prepared maturities share
    one accepted frequency grid. Diagnostics count frequencies, including discarded
    parent panels. Each frequency is evaluated once per prepared maturity.
    """

    _seed_panels = 1
    _v0: float

    def _row_factory(self, v0):
        """Bind all quote rows in a request to the selected variance."""
        state = self._v0 if v0 is None else _real(v0, "v0")
        return partial(_HestonQuotes, v0=state)

    def price_chain(self, quotes: OptionChain, *, v0: float | None = None,
                    fourier: Mapping | None = None) -> List[np.ndarray]:
        """Price discounted premiums in input order using the saved transforms.

        Forwards, discounts, strikes and call/put codes can change at prepared
        maturities. v0 is initial annualized variance, defaulting to the preparation
        snapshot. A new state, quote or stricter budget must pass resolution and tail
        checks on the stored grid. On failure, call refined_for explicitly.
        """
        rows = self._quote_requests(quotes, fourier, row_factory=self._row_factory(v0))
        if all(row.covered_by(self._quotes) for row in rows):
            return List([self._plan.price_vanillas(
                row.evaluate(self._samples[row.expiry], self._plan.nodes),
                forward=row.forward, discfactor=row.discount,
                strikes=row.strikes, optiontypes=row.types,
            ) for row in rows])
        return List(self._query(rows))

    def v0_sensitivity(self, quotes: OptionChain, *, v0: float | None = None,
                       atol: float = 1e-6) -> List[np.ndarray]:
        """Return d(discounted premium)/d(v0) per unit of initial annualized variance.

        Hold forwards, strikes, discounts and the other Heston parameters fixed.
        The default state is the preparation snapshot. The absolute tolerance applies
        to the derivative. Use refined_for with sensitivity_atol if a grid check fails.
        """
        rows = self._sensitivity_requests(
            quotes, "state", atol, "atol", row_factory=self._row_factory(v0)
        )
        if not all(row.covered_by(self._quotes) for row in rows):
            return List(self._query(rows))
        return List([self._plan.vanilla_sensitivities(
            row.evaluate(self._samples[row.expiry], self._plan.nodes),
            forward=row.forward, strikes=row.strikes, discfactor=row.discount,
        ) for row in rows])

    def forward_greeks(self, quotes: OptionChain, *, v0: float | None = None, delta_atol: float = 1e-6,
                       gamma_atol: float = 1e-6) -> tuple[List, List]:
        """Return forward delta and gamma of discounted premiums in expiry and strike order.

        Strike, discount and the selected variance state stay fixed as the forward changes. Tolerances
        are absolute premium per forward unit and per squared forward unit, respectively.
        Queries use saved transforms. If a check fails, pass the same tolerances to refined_for.
        """
        factory = self._row_factory(v0)
        deltas = self._sensitivity_requests(quotes, "delta", delta_atol, "delta_atol", row_factory=factory)
        gammas = self._sensitivity_requests(quotes, "gamma", gamma_atol, "gamma_atol", row_factory=factory)
        if not all(row.covered_by(self._quotes) for row in deltas + gammas):
            values = self._query(deltas + gammas)
            return List(values[:len(deltas)]), List(values[len(deltas):])

        # Both integrals at a strike share the complex phase. Apply both weights in one pass.
        delta_values, gamma_values = List(), List()
        for delta, gamma in zip(deltas, gammas, strict=True):
            coefficients = self._samples[delta.expiry]
            nodes = self._plan.nodes
            derivatives = np.column_stack((delta.evaluate(coefficients, nodes),
                                           gamma.evaluate(coefficients, nodes)))
            pair = self._plan.vanilla_sensitivities(
                derivatives, forward=delta.forward, strikes=delta.strikes,
                discfactor=delta.discount, quantity=("delta", "gamma"),
            )
            pair[delta.types == "C", 0] += delta.discount
            delta_values.append(pair[:, 0].copy())
            gamma_values.append(pair[:, 1].copy())
        return delta_values, gamma_values

    def refined_for(self, quotes: OptionChain, *, v0: float | None = None,
                    fourier: Mapping | None = None, sensitivity_atol: float | None = None,
                    delta_atol: float | None = None,
                    gamma_atol: float | None = None) -> "PreparedHestonGrid":
        """Certify original and additional quotes, evaluating only new frequencies.

        Preparation work limits apply cumulatively. The original handle remains
        usable, including when refinement exhausts its budget. At the requested v0,
        sensitivity_atol checks the initial-variance derivative, and delta_atol and
        gamma_atol check the forward derivatives.
        """
        factory = self._row_factory(v0)
        rows = self._quote_requests(quotes, fourier, row_factory=factory)
        for quantity, atol, name in (("state", sensitivity_atol, "sensitivity_atol"),
                                     ("delta", delta_atol, "delta_atol"),
                                     ("gamma", gamma_atol, "gamma_atol")):
            if atol is not None:
                rows += self._sensitivity_requests(quotes, quantity, atol, name, row_factory=factory)
        return self._refine(rows)


class HestonPricer(ModelPricer):

    """
    ModelPricer for the Heston model, valued by Fourier inversion of the analytic MGF.
    """
    def prepare_fourier(self, option_chain: OptionChain, params: HestonParams, *,
                        fourier: Mapping | None = None, is_spot_measure: bool = True,
                        ) -> PreparedHestonGrid:
        """Prepare a reusable controlled grid for money-market European calls and puts.

        Supply increasing maturities in years. The handle snapshots Heston dynamics and
        checks discounted premium errors using the same settings as controlled price_chain.
        The default absolute premium tolerance is 1e-6. Saved affine coefficients support
        initial-variance scenarios. A new maturity or change to theta, kappa, rho or
        volvol requires a separate preparation.
        """
        if fourier is not None and not isinstance(fourier, Mapping):
            raise ValueError("preparation requires a controlled Fourier settings mapping")
        if not is_spot_measure:
            raise ValueError("controlled Heston requires log returns under the spot measure")
        model = HestonModel(params)
        ttms = np.array(option_chain.ttms, dtype=float, copy=True)

        def transform(phi):
            return (np.column_stack(model._affine_coefficients(ttm=t, phi_grid=phi)) for t in ttms)

        # Match the initial panel used by the fresh controlled Heston slice.
        controls = {"initial_cutoff": 1.0, **({} if fourier is None else fourier)}
        v0 = model.params.v0
        return PreparedHestonGrid._prepare_grid(option_chain, transform, True, controls, (2,),
                                                row_factory=partial(_HestonQuotes, v0=v0), _v0=v0)

    def price_chain(self, option_chain: OptionChain, params: HestonParams, *,
                    fourier: Mapping[str, object] | Literal["legacy"] | None = None,
                    **kwargs) -> List[np.ndarray]:
        """Price a Heston option chain by Fourier inversion.

        Parameters
        ----------
        option_chain : OptionChain
            Expiries, forwards, discounts, strikes and option types.
        params : HestonParams
            Initial variance and Heston dynamics under the money-market measure.
        fourier : mapping or "legacy", optional
            Omission or "legacy" selects the historical grid. A mapping selects adaptive
            quadrature for ordinary calls and puts. An empty mapping targets 1e-6 in
            discounted premium units. Supply ``price_atol`` and ``time_values`` as scalars,
            strike vectors for one expiry or a sequence of strike vectors for a chain.
            A positive ``price_rtol`` scales the supplied time values. Frequency range and
            work controls are ``initial_cutoff``, ``max_cutoff``, ``max_nodes`` and
            ``max_refinements``.

        Returns
        -------
        List[np.ndarray]
            Discounted premiums, one array per expiry in input strike order.

        Raises
        ------
        FloatingPointError
            Controlled quadrature exhausted a work limit or encountered a nonfinite integral.
            The exception's ``diagnostics`` attribute records the work and error estimates.
        """
        variable_type = kwargs.get("variable_type", VariableType.LOG_RETURN)
        if quadrature.uses_controlled_fourier(fourier):
            if variable_type != VariableType.LOG_RETURN or not kwargs.get("is_spot_measure", True):
                raise ValueError("controlled Heston requires log returns under the spot measure")
            legacy_controls = {"vol_scaler", "max_phi", "real_phi", "phi_grid", "is_simpson"}
            conflicts = kwargs.keys() & legacy_controls
            if conflicts:
                raise ValueError(f"fourier conflicts with legacy settings: {sorted(conflicts)}")
            return _controlled_heston_chain(option_chain, params, fourier)
        model_prices_ttms = heston_chain_pricer(v0=params.v0,
                                                theta=params.theta,
                                                kappa=params.kappa,
                                                volvol=params.volvol,
                                                rho=params.rho,
                                                ttms=option_chain.ttms,
                                                forwards=option_chain.forwards,
                                                discfactors=option_chain.discfactors,
                                                strikes_ttms=option_chain.strikes_ttms,
                                                optiontypes_ttms=option_chain.optiontypes_ttms,
                                                variable_type=variable_type)
        return model_prices_ttms

    def model_mc_price_chain(self, option_chain: OptionChain, params: HestonParams,
                             nb_path: int = 100000,
                             variable_type: VariableType = VariableType.LOG_RETURN,
                             *, mc_scheme: str = "euler",
                             **kwargs
                             ) -> (List[np.ndarray], List[np.ndarray]):
        """
        Price with historical Euler or opt-in QE-M for ordinary discounted calls and puts.

        QE-M accepts seed and steps_per_year (default 360), simulates all expiries
        on the same paths and returns raw sample means and independent-path standard errors.
        """
        if quadrature.uses_controlled_fourier(kwargs.pop("fourier", None)):
            raise ValueError("controlled Fourier quadrature applies to analytic vanilla pricing")
        if mc_scheme == "qe_m":
            if variable_type != VariableType.LOG_RETURN or nb_path < 2:
                raise ValueError("QE-M chain pricing requires log returns and at least two paths")
            rows = [quadrature._checked_quotes(f, d, k, types, True) for f, d, k, types in zip(
                option_chain.forwards, option_chain.discfactors, option_chain.strikes_ttms,
                option_chain.optiontypes_ttms, strict=True)]
            if len(rows) != len(option_chain.ttms):
                raise ValueError("supply one quote row per expiry")
            paths = _qe_paths(params, option_chain.ttms, nb_path, **kwargs)
            prices, errors = List(), List()
            zero_variance = np.zeros(nb_path)
            for t, f, d, (strikes, types) in zip(option_chain.ttms, option_chain.forwards,
                                                 option_chain.discfactors, rows):
                index = np.searchsorted(paths.observation_times, t)
                # Zero residual variance selects the realized payoff with raw forwards.
                means, ses = compute_conditional_mc_payoff(
                    paths.states["log_zero_drift_return"][:, index], zero_variance,
                    f, strikes, types, d,
                )
                if not (np.isfinite(means).all() and np.isfinite(ses).all()):
                    raise FloatingPointError("nonfinite QE-M premium or sampling error")
                prices.append(means)
                errors.append(ses)
            return prices, errors
        if mc_scheme != "euler":
            raise ValueError("mc_scheme must be 'euler' or 'qe_m'")
        return heston_mc_chain_pricer(v0=params.v0,
                                      theta=params.theta,
                                      kappa=params.kappa,
                                      rho=params.rho,
                                      volvol=params.volvol,
                                      ttms=option_chain.ttms,
                                      forwards=option_chain.forwards,
                                      discfactors=option_chain.discfactors,
                                      strikes_ttms=option_chain.strikes_ttms,
                                      optiontypes_ttms=option_chain.optiontypes_ttms,
                                      nb_path=nb_path,
                                      variable_type=variable_type)

    @timer
    def simulate_terminal_values(self,
                                 params: HestonParams,
                                 ttm: float = 1.0,
                                 nb_path: int = 100000,
                                 x0: float = 0.0,
                                 *, mc_scheme: str = "euler",
                                 **kwargs
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        """Return terminal log return, annualized variance and integrated variance.

        Opt-in QE-M accepts the same seed and steps_per_year as the chain simulator.
        """
        if mc_scheme == "qe_m":
            if not np.isfinite(x0):
                raise ValueError("initial log return must be finite")
            paths = _qe_paths(params, [ttm], nb_path, **kwargs)
            return (paths.states["log_zero_drift_return"][:, -1] + x0,
                    paths.states["variance"][:, -1].copy(),
                    paths.states["quadratic_variance"][:, -1].copy())
        if mc_scheme != "euler":
            raise ValueError("mc_scheme must be 'euler' or 'qe_m'")
        x0, var0, qvar0 = simulate_heston_x_vol_terminal(ttm=ttm,
                                                         x0=np.full(nb_path, x0, dtype=float),
                                                         var0=params.v0 * np.ones(nb_path),
                                                         qvar0=np.zeros(nb_path),
                                                         theta=params.theta,
                                                         kappa=params.kappa,
                                                         rho=params.rho,
                                                         volvol=params.volvol,
                                                         nb_path=nb_path)
        return x0, var0, qvar0

    @timer
    def calibrate_model_params_to_chain(self,
                                        option_chain: OptionChain,
                                        params0: HestonParams = None,
                                        is_vega_weighted: bool = True,
                                        is_unit_ttm_vega: bool = False,
                                        *,
                                        fourier: Mapping[str, object] | Literal["legacy"] | None = None,
                                        **kwargs) -> HestonParams:
        """Fit the five Heston parameters to weighted market IVs under the Feller constraint.

        A ``fourier`` settings mapping freezes a common grid before each optimizer
        run and checks the fitted premiums on a wider, finer grid. Omission or
        ``"legacy"`` selects the historical objective.
        """
        controlled = quadrature.uses_controlled_fourier(fourier)
        if controlled:
            validate_calibration_options(kwargs)
        if params0 is not None:
            p0 = np.array([params0.v0, params0.theta, params0.kappa, params0.rho, params0.volvol])
        else:
            p0 = np.array([0.1, 0.1, 2.0, -0.2, 1.0])
        bounds = ((0.01, 2.0), (0.01, 2.0), (0.1, 30.0), (-0.99, 0.99), (0.1, 5.0))

        x, y = option_chain.get_chain_data_as_xy()
        market_vols = to_flat_np_array(y)  # market mid quotes
        if is_vega_weighted:
            vegas_ttms = option_chain.get_chain_vegas(is_unit_ttm_vega=is_unit_ttm_vega)
            vegas_ttms = [vegas_ttm/sum(vegas_ttm) for vegas_ttm in vegas_ttms]
            weights = to_flat_np_array(vegas_ttms)
        else:
            weights = np.ones_like(market_vols)

        def parse_model_params(pars: np.ndarray) -> HestonParams:
            """map the optimizer parameter vector onto a model parameter object."""
            v0, theta, kappa, rho, volvol = pars[0], pars[1], pars[2], pars[3], pars[4]
            return HestonParams(v0=v0, theta=theta, kappa=kappa, rho=rho, volvol=volvol)

        def objective(pars: np.ndarray, args: np.ndarray) -> float:
            """weighted mean squared error between model and market implied volatilities."""
            params = parse_model_params(pars=pars)
            model_vols = self.compute_model_ivols_for_chain(
                option_chain=option_chain,
                params=params,
            )
            resid = np.nansum(weights * np.square(to_flat_np_array(model_vols) - market_vols))
            return resid

        def feller(pars: np.ndarray) -> float:
            """
            Feller condition 2 kappa theta - volvol^2, as an inequality constraint.

            Non-negative values keep the variance process from reaching zero.
            """
            params = parse_model_params(pars=pars)
            return 2.0*params.kappa * params.theta - params.volvol*params.volvol

        constraints = ({'type': 'ineq', 'fun': feller})
        options = {'disp': True, 'ftol': 1e-8}

        def optimize(fun, start):
            """Fit the supplied implied-volatility objective under the Feller constraint."""
            return minimize(fun, start, args=None, method='SLSQP', constraints=constraints,
                            bounds=bounds, options=options)

        if controlled:
            def transforms(vector, ttms, phi):
                """Evaluate the trial Heston log transforms on the common grid."""
                model = HestonModel(parse_model_params(vector))
                return (model.log_mgf_grid(ttm=t, phi_grid=phi) for t in ttms)

            calibration = FourierCalibration(option_chain, market_vols, weights, fourier, transforms)
            fitted = calibration.fit(p0, bounds, constraints, [0, 1], optimize)
        else:
            fitted = validate_optimization_result(optimize(objective, p0), bounds)
        return parse_model_params(fitted)


from stochvolmodels.models.heston.kernels import compute_heston_mgf_grid as compute_heston_mgf_grid


@njit(cache=False, fastmath=True)
def heston_chain_pricer(v0: float,
                        theta: float,
                        kappa: float,
                        volvol: float,
                        rho: float,
                        ttms: np.ndarray,
                        forwards: np.ndarray,
                        strikes_ttms: Tuple[np.ndarray, ...],
                        optiontypes_ttms: Tuple[np.ndarray, ...],
                        discfactors: np.ndarray,
                        variable_type: VariableType = VariableType.LOG_RETURN,
                        vol_scaler: float = None  # run calibration on same vol_scaler
                        ) -> List[np.ndarray]:

    # starting values
    """price an option chain by Fourier inversion of the Heston MGF."""
    if vol_scaler is None:
        vol_scaler = np.minimum(0.3, np.sqrt(v0*ttms[0]))
    phi_grid, psi_grid, theta_grid = mgfp.get_transform_var_grid(
        variable_type=variable_type,
        vol_scaler=vol_scaler,
    )
    a_t0 = np.zeros(phi_grid.shape[0], dtype=np.complex128)
    b_t0 = np.zeros(phi_grid.shape[0], dtype=np.complex128)
    ttm0 = 0.0

    # outputs as numpy lists
    model_prices_ttms = List()
    for ttm, forward, discfactor, strikes_ttm, optiontypes_ttm in zip(
        ttms, forwards, discfactors, strikes_ttms, optiontypes_ttms
    ):
        log_mgf_grid, a_t0, b_t0 = compute_heston_mgf_grid(ttm=ttm - ttm0,
                                                           v0=v0,
                                                           theta=theta,
                                                           kappa=kappa,
                                                           volvol=volvol,
                                                           rho=rho,
                                                           phi_grid=phi_grid,
                                                           psi_grid=psi_grid,
                                                           a_t0=a_t0,
                                                           b_t0=b_t0)

        if variable_type == VariableType.LOG_RETURN:
            option_prices = mgfp.vanilla_slice_pricer_with_mgf_grid(log_mgf_grid=log_mgf_grid,
                                                                    phi_grid=phi_grid,
                                                                    forward=forward,
                                                                    strikes=strikes_ttm,
                                                                    optiontypes=optiontypes_ttm,
                                                                    discfactor=discfactor)

        elif variable_type == VariableType.Q_VAR:
            option_prices = mgfp.slice_qvar_pricer_with_a_grid(log_mgf_grid=log_mgf_grid,
                                                               psi_grid=psi_grid,
                                                               ttm=ttm,
                                                               forward=forward,
                                                               strikes=strikes_ttm,
                                                               optiontypes=optiontypes_ttm,
                                                               discfactor=discfactor)
        else:
            raise NotImplementedError(f"variable_type={variable_type}")

        model_prices_ttms.append(option_prices)
        ttm0 = ttm

    return model_prices_ttms


@njit(cache=False, fastmath=True)
def heston_mc_chain_pricer(ttms: np.ndarray,
                           forwards: np.ndarray,
                           discfactors: np.ndarray,
                           strikes_ttms: Tuple[np.ndarray,...],
                           optiontypes_ttms: Tuple[np.ndarray, ...],
                           v0: float,
                           theta: float,
                           kappa: float,
                           rho: float,
                           volvol: float,
                           nb_path: int = 100000,
                           variable_type: VariableType = VariableType.LOG_RETURN
                           ) -> (List[np.ndarray], List[np.ndarray]):

    # starting values
    """price an option chain by simulating the Heston dynamics."""
    ttm0 = 0.0
    x0 = np.zeros(nb_path)
    qvar0 = np.zeros(nb_path)
    var0 = v0*np.ones(nb_path)
    # outputs as numpy lists
    option_prices_ttm, option_std_ttm = List(), List()
    for ttm, forward, discfactor, strikes_ttm, optiontypes_ttm in zip(
        ttms, forwards, discfactors, strikes_ttms, optiontypes_ttms
    ):
        x0, var0, qvar0 = simulate_heston_x_vol_terminal(ttm=ttm - ttm0,
                                                         x0=x0,
                                                         var0=var0,
                                                         qvar0=qvar0,
                                                         theta=theta,
                                                         kappa=kappa,
                                                         rho=rho,
                                                         volvol=volvol,
                                                         nb_path=nb_path)
        ttm0 = ttm
        option_prices, option_std = compute_mc_vars_payoff(x0=x0, sigma0=np.sqrt(var0), qvar0=qvar0,
                                                           ttm=ttm,
                                                           forward=forward,
                                                           strikes_ttm=strikes_ttm,
                                                           optiontypes_ttm=optiontypes_ttm,
                                                           discfactor=discfactor,
                                                           variable_type=variable_type)
        option_prices_ttm.append(option_prices)
        option_std_ttm.append(option_std)

    return option_prices_ttm, option_std_ttm


@njit(cache=False, fastmath=True)
def simulate_heston_x_vol_terminal(ttm: float,
                                   x0:  np.ndarray,
                                   var0: np.ndarray,
                                   qvar0: np.ndarray,
                                   theta: float,
                                   kappa: float,
                                   rho: float,
                                   volvol: float,
                                   nb_path: int = 100000,
                                   nb_steps_per_year: int = 360
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    simulate terminal log-return, variance and quadratic variance under Heston.

    The historical Euler scheme floors each updated variance at ``1e-4``.
    It uses left-point integrated variance and draws the complete first Brownian
    matrix before the second. Length-one initial states broadcast their values.
    """
    if x0.shape[0] == 1:  # initial value
        x0 = x0 * np.ones(nb_path)
    else:
        assert x0.shape[0] == nb_path

    if var0.shape[0] == 1:
        var0 = var0 * np.ones(nb_path)
    else:
        assert var0.shape[0] == nb_path

    if qvar0.shape[0] == 1:  # initial value
        qvar0 = qvar0 * np.ones(nb_path)
    else:
        assert qvar0.shape[0] == nb_path

    nb_steps, dt, grid_t = set_time_grid(ttm=ttm, nb_steps_per_year=nb_steps_per_year)
    w0 = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path))
    w1 = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path))

    return _simulate_heston_with_increments(
        x0, var0, qvar0, theta, kappa, rho, volvol, dt, w0, w1
    )


from stochvolmodels.models.heston.kernels import _simulate_heston_with_increments as _simulate_heston_with_increments


def v0_implied(v0: float, volvol: float, ttm: float):
    """
    to do: find good approximation for heston
    """
    v0 = v0 - volvol*volvol*ttm / 8.0
    # print(f"{ttm}, {v0}")
    return v0


# Manual scenarios are available in ``examples.pricers.run_heston_pricer``.
