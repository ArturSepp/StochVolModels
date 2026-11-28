"""Equity local-volatility marginals bound to native forwards and a settlement measure."""

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from stochvolmodels.data.model_paths import ModelPaths

from stochvolmodels.models.lsv.local_vol import LocalVolCoefficientProvider1D, LocalVolSurface
from stochvolmodels.models.lsv.simulation import LSVMonteCarloConfig
from stochvolmodels.pde_solvers import CoefficientProvider1D, Grid1D, PDE1DCoefficients
from stochvolmodels.utils._arrays import readonly_float_array
from stochvolmodels.utils.config import CoordinateSystem


@dataclass(frozen=True)
class _QuantoCoefficients(LocalVolCoefficientProvider1D):
    """Native log-forward generator under the settlement money-market measure."""

    fx_loading: float

    def evaluate(self, time: float, grid: Grid1D) -> PDE1DCoefficients:
        native = super().evaluate(time, grid)
        return PDE1DCoefficients(
            variance=native.variance,
            drift=native.drift - self.fx_loading * np.sqrt(2 * native.variance),
            discount=native.discount,
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class CorrelatedLocalVolModel:
    """Local-vol equity model with deterministic log-linear native forwards.

    forward_levels has shape (curve times, assets), including today's spots at zero.
    FX is settlement currency per native currency. FX vols and equity-FX correlations
    are explicit constant annualized inputs in asset order. Domestic adjustments are zero.
    """

    asof: str
    asset_ids: tuple[str, ...]
    currencies: tuple[str, ...]
    settlement_currency: str
    surfaces: tuple[LocalVolSurface, ...]
    curve_times: np.ndarray
    forward_levels: np.ndarray
    correlation: np.ndarray
    fx_vols: np.ndarray
    equity_fx_correlations: np.ndarray
    root: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        for name in ("asset_ids", "currencies", "surfaces"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        n = len(self.asset_ids)
        labels = (*self.asset_ids, *self.currencies, self.asof, self.settlement_currency)
        if not all(isinstance(x, str) and x and x.strip() == x for x in labels):
            raise ValueError("market identifiers and currencies must be explicit")
        if (
            not n
            or len(set(self.asset_ids)) != n
            or len(self.currencies) != n
            or len(self.surfaces) != n
        ):
            raise ValueError("asset IDs, currencies and surfaces must align uniquely")
        if any(
            not isinstance(v, LocalVolSurface) or v.coordinate is not CoordinateSystem.LOG_FORWARD
            for v in self.surfaces
        ):
            raise ValueError("basket marginals require log-forward LocalVolSurface")
        for name, ndim in (
            ("curve_times", 1),
            ("forward_levels", 2),
            ("correlation", 2),
            ("fx_vols", 1),
            ("equity_fx_correlations", 1),
        ):
            object.__setattr__(
                self, name, readonly_float_array(getattr(self, name), name=name, ndim=ndim)
            )
        t = self.curve_times
        if (
            len(t) < 2
            or t[0] != 0
            or np.any(np.diff(t) <= 0)
            or self.forward_levels.shape != (len(t), n)
            or np.any(self.forward_levels <= 0)
        ):
            raise ValueError("require positive forward levels on increasing times from zero")
        if (
            self.fx_vols.shape != (n,)
            or self.equity_fx_correlations.shape != (n,)
            or np.any(self.fx_vols < 0)
            or np.any(np.abs(self.equity_fx_correlations) > 1)
        ):
            raise ValueError("invalid equity-FX correlation or FX volatility")
        if np.any(self.fx_vols[np.asarray(self.currencies) == self.settlement_currency] != 0):
            raise ValueError("domestic FX volatility must be zero")
        r = self.correlation
        if (
            r.shape != (n, n)
            or not np.allclose(r, r.T, atol=1e-12, rtol=0)
            or not np.allclose(np.diag(r), 1, atol=1e-12, rtol=0)
        ):
            raise ValueError("require a symmetric correlation matrix with unit diagonal")
        eig, vectors = np.linalg.eigh(r)
        if eig.min() < -1e-12:
            raise ValueError("equity correlation must be positive semidefinite")
        root = vectors * np.sqrt(np.maximum(eig, 0.0))
        root.setflags(write=False)
        object.__setattr__(self, "root", root)
        for currency in set(self.currencies):
            selected = np.asarray(self.currencies) == currency
            vols = self.fx_vols[selected]
            if np.any(vols != vols[0]):
                raise ValueError("equities in one currency must share its FX volatility")
            rho = self.equity_fx_correlations[selected]
            block = r[np.ix_(selected, selected)] - np.outer(rho, rho)
            if np.linalg.eigvalsh(block).min() < -1e-12:
                raise ValueError("infeasible correlations for shared FX factor")

    @property
    def spots(self):
        """Native asset levels at valuation."""
        return self.forward_levels[0]

    def log_forwards(self, times):
        """Interpolate native log forwards within supplied curve coverage."""
        times = np.asarray(times)
        if not np.all((times >= 0) & (times <= self.curve_times[-1])):
            raise ValueError("forward curve does not cover observation grid")
        return np.column_stack(
            [np.interp(times, self.curve_times, np.log(column)) for column in self.forward_levels.T]
        )

    def marginal_pde_coefficient_provider(self, asset_id: str) -> CoefficientProvider1D:
        """Return one asset's log-forward generator under the settlement measure.

        FX is settlement currency per native currency. The quanto adjustment to
        log drift is minus local volatility times FX volatility times correlation.
        """
        idx = self.asset_ids.index(asset_id)
        loading = self.fx_vols[idx] * self.equity_fx_correlations[idx]
        if loading == 0:
            return LocalVolCoefficientProvider1D(self.surfaces[idx])
        return _QuantoCoefficients(self.surfaces[idx], loading)

    def simulate_paths(self, *, observation_times, config: LSVMonteCarloConfig, refinement=0):
        """Emit the supplied grid, forward and surface knots, and any dyadic refinements."""
        from stochvolmodels.models.basket.simulation import simulate_block

        return simulate_block(
            self, observation_times, config, block_id=0, size=config.paths, refinement=refinement
        )

    def iter_path_blocks(
        self,
        *,
        observation_times: ArrayLike,
        config: LSVMonteCarloConfig,
        groups_per_block: int,
        refinement: int = 0,
    ) -> Iterator[ModelPaths]:
        """Yield complete independent groups on reproducible, coupled refinement streams.

        Keep seed and logical block size fixed for a paired grid comparison. The
        last block may be smaller. Group identifiers are unique across blocks.
        """
        from stochvolmodels.models.basket.simulation import simulate_block

        if type(groups_per_block) is not int or groups_per_block < 1:
            raise ValueError("groups_per_block must be a positive integer")
        group_size = 2 if config.antithetic else 1
        if config.paths % group_size:
            raise ValueError("path count must contain complete independent groups")
        groups = config.paths // group_size
        for block, first in enumerate(range(0, groups, groups_per_block)):
            size = group_size * min(groups_per_block, groups - first)
            yield simulate_block(
                self,
                observation_times,
                config,
                block_id=block,
                size=size,
                refinement=refinement,
                first_group_id=first,
            )
