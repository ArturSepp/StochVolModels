"""Fixed-grid analytic calibration and independent premium validation.

Market IVs, weights and discounted premium tolerances are fixed at entry. The
model supplies its parameter codec, transform, constraints and optimizer. Only
grid geometry survives a parameter change. Each optimizer run owns one plan.
"""

import logging
import warnings

import numpy as np
import vanilla_option_pricers as vop
from numba.typed import List

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.pricers.model_pricer import CalibrationError, validate_optimization_result
from stochvolmodels.utils import _fourier_quadrature as fq
from stochvolmodels.utils.config import VariableType


_LOGGER = logging.getLogger(__name__)


def validate_calibration_options(options):
    """Reject grid overrides and alternate payoffs for the fitted spot-vanilla IV objective."""
    conflicts = options.keys() & {"vol_scaler", "max_phi", "real_phi", "phi_grid", "is_simpson"}
    if conflicts:
        raise ValueError(f"controlled Fourier calibration conflicts with {sorted(conflicts)}")
    if (not options.get("is_spot_measure", True)
            or options.get("variable_type", VariableType.LOG_RETURN) != VariableType.LOG_RETURN):
        raise ValueError("controlled Fourier calibration requires ordinary log-return vanillas")


def _feasible(vector, constraints):
    items = (constraints,) if isinstance(constraints, dict) else constraints or ()
    return all(item["fun"](vector) >= -1e-8 for item in items)


def _stress_vectors(start, bounds, constraints, level_indices):
    """Sample lower initial/long-run vol and stronger leverage/vol-of-vol within fit bounds.

    Both parameterizations put leverage and vol-of-vol last. VARSWAP fits have no
    free volatility level. Short maturities are included directly in every scenario.
    """
    low, high = np.asarray(bounds, dtype=float).T
    base = np.clip(start, low, high)
    samples = [base]
    origin = base.copy()
    if not _feasible(origin, constraints):
        # The historical Heston starting point can lie outside the Feller region.
        origin[-2:] = [np.clip(0., low[-2], high[-2]), low[-1]]
        if not _feasible(origin, constraints):
            origin[:-2] = high[:-2]
        if not _feasible(origin, constraints):
            raise CalibrationError("no feasible preflight stress anchor within calibration bounds")
    levels = base.copy()
    levels[level_indices] += .25 * (low[level_indices] - levels[level_indices])
    skew = base.copy()
    endpoint = low[-2] if base[-2] <= 0 else high[-2]
    skew[-2] += .15 * (endpoint - skew[-2])
    skew[-1] += .15 * (high[-1] - skew[-1])
    for target in (levels, skew):
        # A boundary start may require lowering diffusion to keep a level or leverage stress.
        candidates = [origin + (target - origin) * step for step in (1., .5, .25, .125, .0625)]
        lower_diffusion = target.copy()
        lower_diffusion[-1] = low[-1]
        candidates.extend([lower_diffusion, origin])
        for candidate in candidates:
            trial = np.clip(candidate, low, high)
            if _feasible(trial, constraints):
                if not any(np.array_equal(trial, prior) for prior in samples):
                    samples.append(trial)
                break
    return samples


class FourierCalibration:
    """Calibrate against fixed active quotes with an independently checked final grid."""

    def __init__(self, chain, market, weights, fourier, transform):
        active = np.isfinite(market) & np.isfinite(weights) & (weights != 0)
        if not np.any(active):
            raise CalibrationError("calibration has no finite nonzero-weight market quotes")
        self.market = market[active]
        self.weights = weights[active]
        self.transform = transform
        sizes = [len(k) for k in chain.strikes_ttms]
        masks = np.split(active, np.cumsum(sizes)[:-1])
        keep = [i for i, mask in enumerate(masks) if mask.any()]
        self.chain = OptionChain(
            ttms=np.asarray(chain.ttms)[keep], forwards=np.asarray(chain.forwards)[keep],
            discfactors=np.asarray(chain.discfactors)[keep],
            strikes_ttms=tuple(np.asarray(chain.strikes_ttms[i])[masks[i]] for i in keep),
            optiontypes_ttms=tuple(np.asarray(chain.optiontypes_ttms[i])[masks[i]] for i in keep),
            ids=None,
        )
        settings = fq.vanilla_settings_by_expiry(fourier, chain.strikes_ttms)
        self.settings = [settings[i] for i in keep]
        for row, i in zip(self.settings, keep):
            for key in ("price_atol", "time_values"):
                if row.get(key) is not None:
                    row[key] = row[key][masks[i]]
        self.tolerance = np.concatenate([
            2 * fq._premium_budget(k, s.get("price_atol", 1e-6), s.get("price_rtol", 0),
                                   s.get("time_values"))
            for k, s in zip(self.chain.strikes_ttms, self.settings)
        ])

    def _plan(self, vectors, edges=None, refinements=0):
        chain = self.chain

        def batches(phi):
            for vector in vectors:
                try:
                    yield from self.transform(vector, chain.ttms, phi)
                except FloatingPointError as exc:
                    raise FloatingPointError(
                        f"transform failed at parameters {vector}: {exc}"
                    ) from exc

        return fq.integrate_vanilla_chain(
            batches, ttms=chain.ttms, forwards=chain.forwards, discfactors=chain.discfactors,
            strikes_ttms=chain.strikes_ttms, optiontypes_ttms=chain.optiontypes_ttms,
            is_spot_measure=True, settings=self.settings, scenario_count=len(vectors),
            initial_edges=edges, initial_refinements=refinements,
        )

    def _fixed_prices(self, vector, plan):
        chain = self.chain
        transforms = self.transform(vector, chain.ttms, -0.5 + 1j * plan.nodes)
        return List([
            plan.price_vanillas(values, forward=f, discfactor=d, strikes=k, optiontypes=types)
            for values, f, d, k, types in zip(
                transforms, chain.forwards, chain.discfactors,
                chain.strikes_ttms, chain.optiontypes_ttms, strict=True,
            )
        ])

    def _fixed_prices_and_jacobian(self, vector, plan, transform_jacobian):
        chain = self.chain
        samples = transform_jacobian(vector, chain.ttms, -0.5 + 1j * plan.nodes)
        prices, jacobian = List(), []
        for (logs, deriv), f, d, k, types in zip(samples, chain.forwards, chain.discfactors,
                                                chain.strikes_ttms, chain.optiontypes_ttms, strict=True):
            prices.append(plan.price_vanillas(logs, forward=f, discfactor=d, strikes=k, optiontypes=types))
            jacobian.append(plan.vanilla_sensitivities(deriv, forward=f, discfactor=d, strikes=k))
        return prices, np.concatenate(jacobian)

    def _loss(self, prices, price_jacobian=None):
        chain = self.chain
        model_ivs = chain.compute_model_ivols_from_chain_data(model_prices=prices)
        vols = np.concatenate(model_ivs)
        if not np.isfinite(vols).all():
            raise FloatingPointError("nonfinite implied volatility at an active market quote")
        residual = vols - self.market
        loss = np.dot(self.weights, np.square(residual))
        if not np.isfinite(loss):
            raise FloatingPointError("nonfinite weighted implied-volatility objective")
        if price_jacobian is None:
            return float(loss)
        vegas = np.concatenate([d * vop.compute_bsm_slice_vegas(t, f, k, iv)
                                for t, f, k, iv, d in zip(chain.ttms, chain.forwards, chain.strikes_ttms,
                                                         model_ivs, chain.discfactors, strict=True)])
        if not np.isfinite(vegas).all() or np.any(vegas <= 0):
            raise FloatingPointError("invalid discounted model Black vega at an active quote")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            gradient = (2. * self.weights * residual / vegas) @ price_jacobian
        if not np.isfinite(gradient).all():
            raise FloatingPointError("nonfinite calibration gradient")
        return float(loss), gradient

    def _validate(self, vector, plan):
        # Bisect every fitted panel and the extra shell. All validation transforms
        # are recomputed, including any coincident nodes. Recheck resolution and tail.
        edges = np.r_[plan.edges, 2 * plan.edges[-1]]
        mids = edges[:-1] + .5 * np.diff(edges)
        return self._plan([vector], np.sort(np.r_[edges, mids]), refinements=len(mids))

    def fit(self, start, bounds, constraints, level_indices, optimize, *, penalize_trials=False,
            transform_jacobian=None):
        """Optimize once, then restart once for a finite premium tolerance miss."""
        invalid = 0
        first_error = last_error = ""
        phase = "preflight"
        try:
            scenarios = _stress_vectors(start, bounds, constraints, level_indices)
            _LOGGER.info("Preflight parameter vectors in optimizer order: %s",
                         [vector.tolist() for vector in scenarios])
            plan, _, preflight = self._plan(scenarios)
            _LOGGER.info("Accepted preflight grid with %d nodes: %s", plan.nodes.size, preflight)
            for attempt in range(2):
                phase = "optimizer"
                valid_gradient_at = None

                def objective(vector, args=None, *, fixed_plan=plan):
                    """Evaluate the loss and optional gradient on the fixed fit grid."""
                    nonlocal invalid, first_error, last_error, valid_gradient_at
                    try:
                        if transform_jacobian is not None:
                            result = self._loss(*self._fixed_prices_and_jacobian(vector, fixed_plan, transform_jacobian))
                            valid_gradient_at = vector.copy()
                            return result
                        return self._loss(self._fixed_prices(vector, fixed_plan))
                    except FloatingPointError as exc:
                        if not penalize_trials:
                            raise
                        invalid += 1
                        last_error = str(exc)
                        first_error = first_error or last_error
                        if transform_jacobian is not None:
                            return 1e12, np.zeros_like(vector)
                        return 1e12

                fitted = validate_optimization_result(optimize(objective, start), bounds)
                if not _feasible(fitted, constraints):
                    raise CalibrationError("fitted parameters violate calibration constraints")
                phase = "final fixed-grid valuation"
                prices = self._fixed_prices(fitted, plan)
                phase = "independent premium validation"
                finer, reference, diagnostics = self._validate(fitted, plan)
                delta = np.abs(np.concatenate(prices) - np.concatenate(reference))
                worst = int(np.argmax(delta / self.tolerance))
                _LOGGER.info("Fit %d: %d fixed nodes, %d validation nodes, maximum premium "
                             "difference %.9g, largest tolerance ratio %.9g",
                             attempt + 1, plan.nodes.size, finer.nodes.size,
                             np.max(delta), delta[worst] / self.tolerance[worst])
                if delta[worst] <= self.tolerance[worst]:
                    # Compare finite premiums first: a truncated wing can be outside
                    # Black's IV bounds and still be recoverable on the refined grid.
                    phase = "final implied-volatility validation"
                    self._loss(prices)
                    self._loss(List(reference))
                    if transform_jacobian is not None and not np.array_equal(valid_gradient_at, fitted):
                        phase = "final gradient validation"
                        self._loss(*self._fixed_prices_and_jacobian(fitted, plan, transform_jacobian))
                    return fitted
                if attempt:
                    sizes = [len(k) for k in self.chain.strikes_ttms]
                    maturity = np.repeat(self.chain.ttms, sizes)[worst]
                    strike = np.concatenate(self.chain.strikes_ttms)[worst]
                    raise CalibrationError(
                        f"premium mismatch after restart at maturity {maturity:g}, "
                        f"strike {strike:g}: "
                        f"difference {delta[worst]:g}, tolerance {self.tolerance[worst]:g}, "
                        f"fit cutoff {plan.edges[-1]:g}, validation {diagnostics}"
                    )
                start, plan = fitted, finer
        except FloatingPointError as exc:
            raise CalibrationError(f"Fourier calibration {phase} failed: {exc}") from exc
        finally:
            if invalid:
                warnings.warn(f"LogSV calibration rejected {invalid} numerical trials. "
                              f"First reason: {first_error}; last reason: {last_error}",
                              RuntimeWarning, stacklevel=2)
