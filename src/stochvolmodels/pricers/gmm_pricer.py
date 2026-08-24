"""Gaussian-mixture terminal-distribution pricing and calibration."""
from dataclasses import dataclass
from numbers import Real
from typing import ClassVar, Tuple

import numpy as np
from numba import njit
from numba.typed import List
from scipy.optimize import minimize, minimize_scalar

import vanilla_option_pricers as bsm
from stochvolmodels.data.option_chain import OptionChain, OptionSlice
from stochvolmodels.pricers.model_pricer import (
    CalibrationError,
    ModelParams,
    ModelPricer,
    validate_optimization_result,
)
from stochvolmodels.utils.config import VariableType
from stochvolmodels.utils.funcs import npdf, timer, to_flat_np_array


def _require_finite_reported_objective(result: object) -> None:
    """Reject optimizer success that does not carry a finite objective value."""
    try:
        objective = float(getattr(result, "fun"))
    except (AttributeError, TypeError, ValueError) as error:
        raise CalibrationError("Calibration returned no finite objective value") from error
    if not np.isfinite(objective):
        raise CalibrationError("Calibration returned a non-finite objective value")


@dataclass
class GmmParams(ModelParams):
    """
    parameters of a Gaussian mixture model for terminal log-returns.

    The terminal density is a weighted sum of normals, one per state, each with its
    own drift and volatility. Weights sum to one and the mixture must reprice the
    forward, which are the two equality constraints imposed during calibration.
    """
    gmm_weights: np.ndarray
    gmm_mus: np.ndarray
    gmm_vols: np.ndarray
    # TTM is fixed for each fitted slice and is not itself calibrated.
    ttm: float

    def sort_by_mus(self):
        """order the mixture states by drift, so fitted states stay comparable across slices."""
        indices = np.argsort(self.gmm_mus)
        self.gmm_weights = self.gmm_weights[indices]
        self.gmm_mus = self.gmm_mus[indices]
        self.gmm_vols = self.gmm_vols[indices]

    def get_get_avg_vol(self) -> float:
        """weight-averaged volatility, sqrt(sum w_i vol_i^2)."""
        return np.sqrt(np.sum(self.gmm_weights*np.square(self.gmm_vols)))

    def compute_state_pdfs(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """per-state densities and their weighted aggregate on a log-return grid."""
        state_pdfs = np.zeros((len(x), len(self.gmm_weights)))
        agg_pdf = np.zeros_like(x)
        states = zip(self.gmm_weights, self.gmm_mus, self.gmm_vols)
        for idx, (gmm_weight, mu, vol) in enumerate(states):
            state_pdf = npdf(x, mu=mu*self.ttm, vol=vol*np.sqrt(self.ttm))
            state_pdfs[:, idx] = state_pdf
            agg_pdf += gmm_weight*state_pdf
        return state_pdfs, agg_pdf

    def compute_pdf(self, x: np.ndarray):
        """aggregate mixture density on a log-return grid."""
        pdfs = np.zeros_like(x)
        for gmm_weight, mu, vol in zip(self.gmm_weights, self.gmm_mus, self.gmm_vols):
            pdfs = pdfs + gmm_weight*npdf(x, mu=mu*self.ttm, vol=vol*np.sqrt(self.ttm))
        return pdfs


class GmmPricer(ModelPricer):

    """ModelPricer valuing options as a weighted sum of Black-Scholes prices."""
    def price_chain(self, option_chain: OptionChain, params: GmmParams, **kwargs) -> np.ndarray:
        """
        implementation of generic method price_chain using heston wrapper for heston chain
        """
        model_prices_ttms = gmm_vanilla_chain_pricer(gmm_weights=params.gmm_weights,
                                                     gmm_mus=params.gmm_mus,
                                                     gmm_vols=params.gmm_vols,
                                                     ttms=option_chain.ttms,
                                                     forwards=option_chain.forwards,
                                                     strikes_ttms=option_chain.strikes_ttms,
                                                     optiontypes_ttms=option_chain.optiontypes_ttms,
                                                     discfactors=option_chain.discfactors)

        return model_prices_ttms

    def model_mc_price_chain(self, option_chain: OptionChain, params: GmmParams,
                             nb_path: int = 100000,
                             variable_type: VariableType = VariableType.LOG_RETURN,
                             **kwargs
                             ) -> (List[np.ndarray], List[np.ndarray]):
        """price an option chain by Monte Carlo rather than the analytic solution."""
        raise NotImplementedError

    @timer
    def calibrate_model_params_to_chain_slice(self,
                                              option_chain: OptionChain,
                                              params0: GmmParams = None,
                                              is_vega_weighted: bool = True,
                                              is_unit_ttm_vega: bool = False,
                                              n_mixtures: int = 4,
                                              **kwargs
                                              ) -> GmmParams:
        """
        implementation of model calibration interface
        fit: GmmParams
        nb: always use option_chain with one slice because we need martingale condition per slice
        """

        ttms = option_chain.ttms
        if len(ttms) > 1:
            raise NotImplementedError("cannot calibrate to multiple slices")
        ttm = ttms[0]

        # p0 = (gmm_weights, gmm_mus, gmm_vols)
        if params0 is not None:
            p0 = np.concatenate((params0.gmm_weights, params0.gmm_mus, params0.gmm_vols))
            n_mixtures = len(params0.gmm_weights)
        else:
            gmm_weights = np.ones(n_mixtures) / n_mixtures
            gmm_mus = np.zeros(n_mixtures)
            gmm_vols = np.linspace(0.2, 1.0, n_mixtures)
            p0 = np.concatenate((gmm_weights, gmm_mus, gmm_vols))

        gmm_weights_bounds = [(0.0, 1.0)]*n_mixtures
        gmm_mus_bounds = [(-10.0, 10.0)]*n_mixtures
        gmm_vols_bounds = [(0.01, 4.0)]*n_mixtures
        bounds = np.concatenate((gmm_weights_bounds, gmm_mus_bounds, gmm_vols_bounds))

        _, y = option_chain.get_chain_data_as_xy()
        market_vols = to_flat_np_array(y)  # market mid quotes
        if is_vega_weighted:
            vegas_ttms = option_chain.get_chain_vegas(is_unit_ttm_vega=is_unit_ttm_vega)
            normalized_vegas = []
            for vegas_ttm in vegas_ttms:
                vegas_array = np.asarray(vegas_ttm, dtype=float)
                vega_sum = float(np.sum(vegas_array))
                if (
                    not np.all(np.isfinite(vegas_array))
                    or np.any(vegas_array < 0.0)
                    or not np.isfinite(vega_sum)
                    or vega_sum <= 0.0
                ):
                    raise CalibrationError("Calibration received invalid vega weights")
                normalized_vegas.append(vegas_array / vega_sum)
            weights = to_flat_np_array(normalized_vegas)
        else:
            weights = np.ones_like(market_vols)
        if weights.shape != market_vols.shape:
            raise CalibrationError("Calibration quote values and weights have different shapes")
        active_quotes = np.isfinite(market_vols) & np.isfinite(weights) & (weights > 0.0)
        if not np.any(active_quotes):
            raise CalibrationError("Calibration has no finite positive-weight quotes")

        def parse_model_params(pars: np.ndarray) -> GmmParams:
            """map the optimizer parameter vector onto a model parameter object."""
            gmm_weights = pars[:n_mixtures]
            gmm_mus = pars[n_mixtures:2*n_mixtures]
            gmm_vols = pars[2*n_mixtures:]
            return GmmParams(gmm_weights=gmm_weights, gmm_mus=gmm_mus, gmm_vols=gmm_vols, ttm=ttm)

        def objective(pars: np.ndarray, args: np.ndarray) -> float:
            """Weighted SSE, with non-finite repricing treated as an invalid trial."""
            params = parse_model_params(pars=pars)
            model_vols = self.compute_model_ivols_for_chain(
                option_chain=option_chain,
                params=params,
            )
            model_vols_flat = to_flat_np_array(model_vols)
            if model_vols_flat.shape != market_vols.shape:
                return np.inf
            active_model_vols = model_vols_flat[active_quotes]
            if not np.all(np.isfinite(active_model_vols)):
                return np.inf
            residual_terms = weights[active_quotes] * np.square(
                active_model_vols - market_vols[active_quotes]
            )
            if not np.all(np.isfinite(residual_terms)):
                return np.inf
            return float(np.sum(residual_terms))

        if n_mixtures == 1:

            def objective_vol(vol: float) -> float:
                """One-state objective with weight and martingale drift imposed exactly."""
                pars = np.array([1.0, -0.5 * vol * vol, vol])
                return objective(pars, args=None)

            scalar_result = minimize_scalar(
                objective_vol,
                bounds=gmm_vols_bounds[0],
                method='bounded',
                options={'xatol': 1.0e-10, 'maxiter': 500},
            )
            if not scalar_result.success or not np.isfinite(scalar_result.x):
                raise CalibrationError(
                    f"Calibration failed: {getattr(scalar_result, 'message', 'no message')}"
                )
            _require_finite_reported_objective(scalar_result)
            fitted_vol = float(scalar_result.x)
            if not gmm_vols_bounds[0][0] <= fitted_vol <= gmm_vols_bounds[0][1]:
                raise CalibrationError("Calibration returned volatility outside its bounds")
            if not np.isfinite(objective_vol(fitted_vol)):
                raise CalibrationError("Calibration returned a non-finite repricing objective")
            fit_params = GmmParams(
                gmm_weights=np.array([1.0]),
                gmm_mus=np.array([-0.5 * fitted_vol * fitted_vol]),
                gmm_vols=np.array([fitted_vol]),
                ttm=ttm,
            )
            try:
                GmmTerminalModel(params=fit_params)
            except (TypeError, ValueError) as error:
                raise CalibrationError(
                    f"Calibration returned invalid GMM parameters: {error}"
                ) from error
            return fit_params

        def weights_sum(pars: np.ndarray) -> float:
            """equality constraint sum of mixture weights minus one."""
            params = parse_model_params(pars=pars)
            return np.sum(params.gmm_weights) - 1.0

        def martingale(pars: np.ndarray) -> float:
            # we set to 1.0, mutplication with foward will be set by pricing
            """
            equality constraint forcing the mixture to reprice the forward.

            Returns sum_i w_i exp((mu_i + 0.5 vol_i^2) ttm) - 1, normalised to a unit
            forward; the scaling by the actual forward happens at pricing time.
            """
            params = parse_model_params(pars=pars)
            terminal_means = np.exp(
                (params.gmm_mus + 0.5 * params.gmm_vols * params.gmm_vols) * ttm
            )
            return np.sum(params.gmm_weights * terminal_means) - 1.0

        constraints = ({'type': 'eq', 'fun': weights_sum}, {'type': 'eq', 'fun': martingale})
        options = {'disp': True, 'ftol': 1e-10, 'maxiter': 500}

        res = minimize(
            objective,
            p0,
            args=None,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options=options,
        )
        fit_params = parse_model_params(
            pars=validate_optimization_result(res, bounds)
        )
        _require_finite_reported_objective(res)
        fit_params.sort_by_mus()
        if not np.isfinite(
            objective(
                np.concatenate(
                    (fit_params.gmm_weights, fit_params.gmm_mus, fit_params.gmm_vols)
                ),
                args=None,
            )
        ):
            raise CalibrationError("Calibration returned a non-finite repricing objective")
        try:
            GmmTerminalModel(params=fit_params)
        except (TypeError, ValueError) as error:
            raise CalibrationError(
                f"Calibration returned invalid GMM parameters: {error}"
            ) from error

        return fit_params

    @timer
    def calibrate_model_params_to_chain(self,
                                        option_chain: OptionChain,
                                        is_vega_weighted: bool = True,
                                        is_unit_ttm_vega: bool = False,
                                        n_mixtures: int = 4,
                                        **kwargs
                                        ) -> List[str, GmmParams]:
        """
        model params are fitted per slice
        need to splic chain to slices
        """
        fit_params = {}
        params0 = None
        for ids_ in option_chain.ids:
            option_chain0 = OptionChain.get_slices_as_chain(option_chain, ids=[ids_])
            params0 = self.calibrate_model_params_to_chain_slice(option_chain=option_chain0,
                                                                 params0=params0,
                                                                 is_vega_weighted=is_vega_weighted,
                                                                 is_unit_ttm_vega=is_unit_ttm_vega,
                                                                 n_mixtures=n_mixtures,
                                                                 **kwargs)
            fit_params[ids_] = params0
        return fit_params


@dataclass(frozen=True, init=False)
class GmmTerminalModel:
    """Validated Gaussian-mixture law for one-maturity European options.

    The state parameters describe normalized terminal log-return
    ``X = log(S_T / F_T)``. The adapter enforces both probability normalization and
    ``E[exp(X)] = 1``, while the supplied option-slice discount factor discounts the
    standard call or put payoff exactly once.
    """

    _gmm_weights: np.ndarray
    _gmm_mus: np.ndarray
    _gmm_vols: np.ndarray
    _ttm: float

    _CONSTRAINT_ATOL: ClassVar[float] = 5.0e-10

    def __init__(self, params: GmmParams) -> None:
        """Validate and snapshot one legacy Gaussian-mixture parameter payload."""
        if not isinstance(params, GmmParams):
            raise TypeError("params must be a GmmParams instance")

        weights = self._snapshot_real_array("gmm_weights", params.gmm_weights)
        mus = self._snapshot_real_array("gmm_mus", params.gmm_mus)
        vols = self._snapshot_real_array("gmm_vols", params.gmm_vols)
        if not weights.size == mus.size == vols.size:
            raise ValueError("gmm_weights, gmm_mus, and gmm_vols must have the same length")
        if np.any(weights < 0.0):
            raise ValueError("gmm_weights must be nonnegative")
        if np.any(vols <= 0.0):
            raise ValueError("gmm_vols must be positive")

        ttm = params.ttm
        if (
            isinstance(ttm, (bool, np.bool_))
            or not isinstance(ttm, Real)
            or not np.isfinite(ttm)
            or ttm <= 0.0
        ):
            raise ValueError("ttm must be a finite positive real scalar")
        ttm_float = float(ttm)

        weight_error = abs(float(np.sum(weights)) - 1.0)
        if weight_error > self._CONSTRAINT_ATOL:
            raise ValueError(
                "gmm_weights must sum to one within "
                f"{self._CONSTRAINT_ATOL:.0e}"
            )
        log_martingale = self._evaluate_log_mgf(
            weights=weights,
            mus=mus,
            vols=vols,
            ttm=ttm_float,
            phi_grid=np.array([1.0]),
        )[0]
        if not np.isfinite(log_martingale) or abs(float(log_martingale)) > self._CONSTRAINT_ATOL:
            raise ValueError(
                "GMM martingale condition abs(log M(1)) must not exceed "
                f"{self._CONSTRAINT_ATOL:.0e}"
            )

        for array in (weights, mus, vols):
            array.setflags(write=False)
        object.__setattr__(self, "_gmm_weights", weights)
        object.__setattr__(self, "_gmm_mus", mus)
        object.__setattr__(self, "_gmm_vols", vols)
        object.__setattr__(self, "_ttm", ttm_float)

    @staticmethod
    def _snapshot_real_array(name: str, values: object) -> np.ndarray:
        """Return a detached finite one-dimensional real floating array."""
        object_array = np.asarray(values, dtype=object)
        if any(isinstance(value, (bool, np.bool_)) for value in object_array.flat):
            raise ValueError(f"{name} must contain finite real numbers, not booleans")
        array = np.asarray(values)
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional array")
        if array.dtype.kind not in "iuf":
            raise ValueError(f"{name} must contain finite real numbers")
        snapshot = np.array(array, dtype=float, copy=True)
        if not np.all(np.isfinite(snapshot)):
            raise ValueError(f"{name} must contain finite real numbers")
        return snapshot

    @staticmethod
    def _evaluate_log_mgf(
        *,
        weights: np.ndarray,
        mus: np.ndarray,
        vols: np.ndarray,
        ttm: float,
        phi_grid: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the mixture log-MGF with a real-part exponential shift."""
        phi_array = np.asarray(phi_grid)
        flat_phi = phi_array.reshape(-1)
        positive_weights = weights > 0.0
        active_weights = weights[positive_weights]
        active_mus = mus[positive_weights]
        active_vols = vols[positive_weights]
        exponents = ttm * (
            active_mus[:, np.newaxis] * flat_phi[np.newaxis, :]
            + 0.5
            * np.square(active_vols[:, np.newaxis])
            * np.square(flat_phi[np.newaxis, :])
        )
        real_shifts = np.max(np.real(exponents), axis=0)
        shifted_sum = np.sum(
            active_weights[:, np.newaxis]
            * np.exp(exponents - real_shifts[np.newaxis, :]),
            axis=0,
        )
        return (real_shifts + np.log(shifted_sum)).reshape(phi_array.shape)

    @property
    def params(self) -> GmmParams:
        """Return a detached legacy parameter payload for inspection or facade calls."""
        return GmmParams(
            gmm_weights=self._gmm_weights.copy(),
            gmm_mus=self._gmm_mus.copy(),
            gmm_vols=self._gmm_vols.copy(),
            ttm=self._ttm,
        )

    @property
    def ttm(self) -> float:
        """Return the terminal law's time to maturity in years."""
        return self._ttm

    def _option_chain(self, option_slice: OptionSlice) -> OptionChain:
        """Validate a standard-payoff slice and convert it to the legacy facade input."""
        if not isinstance(option_slice, OptionSlice):
            raise TypeError("option_slice must be an OptionSlice instance")
        if option_slice.ttm != self.ttm:
            raise ValueError("option_slice.ttm must exactly match the bound params.ttm")
        unsupported = set(np.asarray(option_slice.optiontypes).astype(str)) - {"C", "P"}
        if unsupported:
            raise NotImplementedError(
                "GmmTerminalModel supports C/P only; inverse IC/IP settlement is not implemented"
            )
        return OptionChain.slice_to_chain(
            ttm=option_slice.ttm,
            forward=option_slice.forward,
            strikes=np.asarray(option_slice.strikes, dtype=float),
            optiontypes=np.asarray(option_slice.optiontypes),
            discfactor=option_slice.discfactor,
            id=option_slice.id,
        )

    def price_european(self, option_slice: OptionSlice) -> np.ndarray:
        """Return standard-call/put prices shaped like ``option_slice.strikes``."""
        option_chain = self._option_chain(option_slice)
        prices = GmmPricer().price_chain(option_chain=option_chain, params=self.params)
        return np.asarray(prices[0], dtype=float)

    def implied_vols(self, option_slice: OptionSlice) -> np.ndarray:
        """Return Black implied volatilities shaped like ``option_slice.strikes``."""
        option_chain = self._option_chain(option_slice)
        ivols = GmmPricer().compute_model_ivols_for_chain(
            option_chain=option_chain,
            params=self.params,
        )
        return np.asarray(ivols[0], dtype=float)

    def log_mgf_grid(self, *, phi_grid: np.ndarray) -> np.ndarray:
        """Return ``log E[exp(phi*X)]`` on a finite real or complex grid."""
        raw_grid = np.asarray(phi_grid)
        if raw_grid.size == 0 or raw_grid.dtype.kind not in "iufc":
            raise ValueError("phi_grid must be a non-empty finite real or complex numeric grid")
        if raw_grid.dtype.kind == "c":
            validated_grid = np.asarray(raw_grid, dtype=np.complex128)
        else:
            validated_grid = np.asarray(raw_grid, dtype=float)
        if not np.all(np.isfinite(validated_grid)):
            raise ValueError("phi_grid must be a non-empty finite real or complex numeric grid")
        return self._evaluate_log_mgf(
            weights=self._gmm_weights,
            mus=self._gmm_mus,
            vols=self._gmm_vols,
            ttm=self._ttm,
            phi_grid=validated_grid,
        )


@njit
def compute_gmm_vanilla_price(gmm_weights: np.ndarray,
                              gmm_mus: np.ndarray,
                              gmm_vols: np.ndarray,
                              ttm: float,
                              forward: float,
                              strike: float,
                              optiontype: str,
                              discfactor: float = 1.0
                              ) -> float:
    """
    bsm deltas for strikes and vols
    """
    price = 0.0
    for gmm_weight, gmm_mu, gmm_vol in zip(gmm_weights, gmm_mus, gmm_vols):
        forward_i = forward*np.exp((gmm_mu+0.5*gmm_vol*gmm_vol)*ttm)
        # forward is vol-adjusted
        price_i = bsm.compute_bsm_vanilla_price(forward=forward_i,
                                                strike=strike,
                                                ttm=ttm,
                                                vol=gmm_vol,
                                                optiontype=optiontype,
                                                discfactor=1.0)
        price += gmm_weight * price_i
    return discfactor*price


@njit
def compute_gmm_vanilla_slice_prices(gmm_weights: np.ndarray,
                                     gmm_mus: np.ndarray,
                                     gmm_vols: np.ndarray,
                                     ttm: float,
                                     forward: float,
                                     strikes: np.ndarray,
                                     optiontypes: np.ndarray,
                                     discfactor: float = 1.0
                                     ) -> np.ndarray:
    """
    vectorised bsm deltas for array of aligned strikes, vols, and optiontypes
    """
    def f(strike: float, optiontype: str) -> float:
        """helper evaluated inside the enclosing routine."""
        return compute_gmm_vanilla_price(gmm_weights=gmm_weights,
                                         gmm_mus=gmm_mus,
                                         gmm_vols=gmm_vols,
                                         forward=forward,
                                         ttm=ttm,
                                         strike=strike,
                                         optiontype=optiontype,
                                         discfactor=discfactor)

    gmm_prices = np.zeros_like(strikes)
    for idx, (strike, optiontype) in enumerate(zip(strikes, optiontypes)):
        gmm_prices[idx] = f(strike, optiontype)
    return gmm_prices


@njit
def gmm_vanilla_chain_pricer(gmm_weights: np.ndarray,
                             gmm_mus: np.ndarray,
                             gmm_vols: np.ndarray,
                             ttms: np.ndarray,
                             forwards: np.ndarray,
                             strikes_ttms: Tuple[np.ndarray, ...],
                             optiontypes_ttms: Tuple[np.ndarray, ...],
                             discfactors: np.ndarray,
                             ) -> np.ndarray:
    """
    vectorised bsm deltas for array of aligned strikes, vols, and optiontypes
    """
    # outputs as numpy lists
    model_prices_ttms = List()
    for ttm, forward, discfactor, strikes_ttm, optiontypes_ttm in zip(
        ttms,
        forwards,
        discfactors,
        strikes_ttms,
        optiontypes_ttms,
    ):
        option_prices_ttm = compute_gmm_vanilla_slice_prices(gmm_weights=gmm_weights,
                                                             gmm_mus=gmm_mus,
                                                             gmm_vols=gmm_vols,
                                                             ttm=ttm,
                                                             forward=forward,
                                                             strikes=strikes_ttm,
                                                             optiontypes=optiontypes_ttm,
                                                             discfactor=discfactor)
        model_prices_ttms.append(option_prices_ttm)

    return model_prices_ttms


# Manual scenarios are available in ``stochvolmodels.pricers.run_local.gmm_pricer_run``.
