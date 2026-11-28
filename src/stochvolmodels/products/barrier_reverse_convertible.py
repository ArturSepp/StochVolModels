"""Cash-settled worst-of barrier reverse convertibles with one maturity payment.

Levels are native index prices. The coupon and redemption loss are fractions of face.
Contract time arguments are year fractions measured from the current valuation cutoff.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from stochvolmodels.data.model_paths import ModelPaths
from stochvolmodels.products.payoffs import _finite_float, _positive_float, _validated_label
from stochvolmodels.utils._arrays import readonly_float_array


class Monitoring(str, Enum):
    """Contract observation conventions supported by the pricing model."""

    SYNCHRONIZED = "continuous_synchronized"
    MATURITY = "maturity_only"


class TouchState(str, Enum):
    """Historical evidence available at the valuation cutoff."""

    CLEAR = "clear"
    HIT = "hit"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class BrcTerms:
    """Original fixing and absolute barrier per asset, remaining times and settlement unit."""

    asset_ids: tuple[str, ...]
    fixings: np.ndarray
    barriers: np.ndarray
    fixing_time: float
    payment_time: float
    currency: str
    monitoring: Monitoring = Monitoring.SYNCHRONIZED
    face: float = 1.0

    def __post_init__(self):
        ids = tuple(_validated_label(x, "asset_id") for x in self.asset_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("unique asset IDs are required")
        object.__setattr__(self, "asset_ids", ids)
        for field in ("fixings", "barriers"):
            levels = readonly_float_array(getattr(self, field), name=field, ndim=1)
            if levels.size != len(ids) or np.any(levels <= 0):
                raise ValueError(f"{field} must contain one positive level per asset")
            object.__setattr__(self, field, levels)
        if np.any(self.barriers > self.fixings):
            raise ValueError("barriers must not exceed original fixings")
        _finite_float(self.fixing_time, "fixing_time")
        _finite_float(self.payment_time, "payment_time")
        if self.payment_time < self.fixing_time:
            raise ValueError("payment precedes final fixing")
        _positive_float(self.face, "face")
        _validated_label(self.currency, "currency")
        object.__setattr__(self, "monitoring", Monitoring(self.monitoring))

    def loss_at_touch(self, levels):
        """Loss per unit face if a knock-in has occurred, in contractual asset order."""
        return np.maximum(1.0 - np.min(levels / self.fixings, axis=-1), 0.0)


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class BrcState:
    """Historical touch evidence and optional final fixings in contractual asset order."""

    asof: str
    touch: TouchState = TouchState.UNRESOLVED
    final_levels: np.ndarray | None = None
    paid: bool = False

    def __post_init__(self):
        _validated_label(self.asof, "asof")
        object.__setattr__(self, "touch", TouchState(self.touch))
        if self.final_levels is not None:
            levels = readonly_float_array(self.final_levels, name="final_levels", ndim=1)
            if np.any(levels <= 0):
                raise ValueError("final_levels must be positive")
            object.__setattr__(self, "final_levels", levels)


@dataclass(frozen=True, slots=True)
class BrcPayoff:
    """Undiscounted cash amount using resolved touch indicators or conditional weights."""

    terms: BrcTerms
    coupon: float = 0.0

    def __post_init__(self):
        _finite_float(self.coupon, "coupon")

    @property
    def required_asset_ids(self):
        return self.terms.asset_ids

    @property
    def expiry(self):
        return self.terms.fixing_time

    @property
    def payoff_unit(self):
        return self.terms.currency

    @property
    def settlement_unit(self):
        return self.terms.currency

    def __call__(self, paths: ModelPaths):
        matches = np.flatnonzero(paths.observation_times == self.expiry)
        if len(matches) != 1:
            raise ValueError("final fixing must occur on the observation grid")
        idx = matches[0]
        assets = [paths.asset_ids.index(name) for name in self.terms.asset_ids]
        levels = paths.assets[:, idx, :][:, assets]
        if np.any(levels <= 0) or not np.isfinite(levels).all():
            raise ValueError("invalid final fixing levels")
        if self.terms.monitoring is Monitoring.MATURITY:
            hit = np.any(levels <= self.terms.barriers, axis=1)
        else:
            if paths.state_units.get("knock_in_weight") != "probability":
                raise ValueError("continuous payoff requires resolved knock_in_weight")
            hit = paths.states["knock_in_weight"][:, idx]
            if hit.ndim != 1 or not np.all((hit >= 0) & (hit <= 1)):
                raise ValueError("invalid knock-in weights")
        return self.terms.face * (1 + self.coupon - hit * self.terms.loss_at_touch(levels))
