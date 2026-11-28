"""Scheduled single-equity accumulators with close-only knock-out and signed settlement cash.

Times on the immutable contract are ACT/365 year fractions from issue. State is after
fixings and payments at its cutoff. Pricing rebases the remaining times to that cutoff.
"""

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stochvolmodels.utils._arrays import readonly_float_array


_TIME_TOL = 1e-12  # ACT/365 years, for independently constructed copies of the same date.


class AccumulatorDirection(str, Enum):
    """Buy under an upper knock-out or sell under a lower knock-out."""

    ACCUMULATOR = "accumulator"
    DECUMULATOR = "decumulator"


@dataclass(frozen=True, kw_only=True, eq=False)
class AccumulatorTerms:
    """Original terms. Payment indices are zero-based and refer to distinct payment times."""

    asset_id: str
    direction: AccumulatorDirection
    original_spot: float
    barrier: float
    notional: float
    currency: str
    fixing_times: ArrayLike
    payment_times: ArrayLike
    payment_indices: ArrayLike

    def __post_init__(self) -> None:
        if not isinstance(self.direction, AccumulatorDirection):
            raise TypeError("direction must be an AccumulatorDirection")
        for name in ("asset_id", "currency"):
            label = getattr(self, name)
            if not isinstance(label, str) or not label or label.strip() != label:
                raise ValueError(f"{name} must be an explicit identifier")
        for name in ("original_spot", "barrier", "notional"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        upper = self.direction is AccumulatorDirection.ACCUMULATOR
        if (upper and self.barrier <= self.original_spot) or (
            not upper and self.barrier >= self.original_spot
        ):
            raise ValueError("barrier must be beyond original spot in the knock-out direction")
        for name in ("fixing_times", "payment_times"):
            a = readonly_float_array(getattr(self, name), name=name, ndim=1)
            if not len(a) or a[0] <= _TIME_TOL or np.any(np.diff(a) <= _TIME_TOL):
                raise ValueError(f"{name} must be distinct, positive and increasing")
            object.__setattr__(self, name, a)
        idx = np.array(self.payment_indices, copy=True)
        if (
            idx.shape != self.fixing_times.shape
            or idx.dtype.kind not in "iu"
            or np.any(np.diff(idx.astype(np.int64)) < 0)
            or not np.array_equal(np.unique(idx), np.arange(len(self.payment_times)))
        ):
            raise ValueError("payment_indices must map fixings in order onto every payment")
        if np.any(self.payment_times[idx] + _TIME_TOL < self.fixing_times):
            raise ValueError("a payment precedes its fixing")
        last_fixings = self.fixing_times[np.r_[idx[:-1] != idx[1:], True]]
        payments = np.where(
            np.abs(self.payment_times - last_fixings) <= _TIME_TOL, last_fixings, self.payment_times
        )
        if np.any(np.diff(payments) <= _TIME_TOL):
            raise ValueError("payment mapping has ambiguous dates within the schedule tolerance")
        payments.setflags(write=False)
        object.__setattr__(self, "payment_times", payments)
        idx.setflags(write=False)
        object.__setattr__(self, "payment_indices", idx)

    def knocked_out(self, levels: ArrayLike) -> NDArray[np.bool_]:
        """Evaluate the inclusive barrier on observed native spot levels."""
        spots = np.asarray(levels)
        if self.direction is AccumulatorDirection.ACCUMULATOR:
            return spots >= self.barrier
        return spots <= self.barrier

    def fixing_cash(self, levels: ArrayLike, strike_ratio: float) -> NDArray[np.float64]:
        """Signed settlement amount on a surviving fixing at each supplied native spot."""
        if not np.isfinite(strike_ratio) or strike_ratio <= 0:
            raise ValueError("strike_ratio must be positive and finite")
        difference = np.asarray(levels) / self.original_spot - strike_ratio
        if self.direction is AccumulatorDirection.DECUMULATOR:
            difference = -difference
        return (self.notional / len(self.fixing_times)) * (difference + np.minimum(difference, 0))


@dataclass(frozen=True, kw_only=True, eq=False)
class AccumulatorState:
    """Post-close state with signed unpaid amounts in original payment-index order.

    A missing native close while live makes history unresolved. Reconstruct from the
    last complete state when the evidence arrives. Stopped trades retain earned cash.
    """

    asof: str
    elapsed: float
    fixed_unpaid: ArrayLike
    stopped: bool = False
    history_complete: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.asof, str) or not self.asof.strip():
            raise ValueError("state requires an asof identifier")
        if not np.isfinite(self.elapsed) or self.elapsed < 0:
            raise ValueError("elapsed must be a nonnegative ACT/365 time from issue")
        if type(self.stopped) is not bool or type(self.history_complete) is not bool:
            raise TypeError("state flags must be booleans")
        cash = readonly_float_array(self.fixed_unpaid, name="fixed_unpaid", ndim=1)
        object.__setattr__(self, "fixed_unpaid", cash)

    @classmethod
    def at_issue(cls, terms: AccumulatorTerms, asof: str) -> "AccumulatorState":
        """Open the original schedule with no observed fixings or balances."""
        return cls(asof=asof, elapsed=0.0, fixed_unpaid=np.zeros(len(terms.payment_times)))

    def next_fixing(self, terms: AccumulatorTerms) -> int:
        """Index of the first fixing strictly after this valuation cutoff."""
        return int(np.searchsorted(terms.fixing_times, self.elapsed + _TIME_TOL, side="right"))

    def outstanding_payments(self, terms: AccumulatorTerms) -> NDArray[np.intp]:
        """Original indices requiring discounts, in increasing payment-date order."""
        if len(self.fixed_unpaid) != len(terms.payment_times):
            raise ValueError("fixed_unpaid must align with original payment dates")
        observed = self.next_fixing(terms)
        active = self.fixed_unpaid != 0
        if np.any(active & (terms.payment_times <= self.elapsed + _TIME_TOL)):
            raise ValueError("a settled payment cannot remain unpaid")
        if np.any(active & ~np.isin(np.arange(len(active)), terms.payment_indices[:observed])):
            raise ValueError("cash cannot be fixed before its observation")
        if not self.stopped:
            active[terms.payment_indices[observed:]] = True
        return np.flatnonzero(active)

    def advance(
        self,
        terms: AccumulatorTerms,
        *,
        asof: str,
        elapsed: float,
        closes: ArrayLike,
        strike_ratio: float,
    ) -> tuple["AccumulatorState", float]:
        """Apply new native closes followed by due payments. Return state and net cash paid.

        Supply one spot per newly scheduled fixing, using NaN for missing observations.
        Cash returned with incomplete history includes only amounts established with certainty.
        """
        self.outstanding_payments(terms)
        if not np.isfinite(elapsed) or elapsed < self.elapsed:
            raise ValueError("elapsed must advance monotonically")
        if not np.isfinite(strike_ratio) or strike_ratio <= 0:
            raise ValueError("strike_ratio must be positive and finite")
        first = self.next_fixing(terms)
        end = np.searchsorted(terms.fixing_times, elapsed + _TIME_TOL, side="right")
        closes = np.asarray(closes, dtype=float)
        if closes.shape != (end - first,) or np.any(np.isinf(closes) | (closes <= 0)):
            raise ValueError("require a positive close or NaN per newly scheduled fixing")
        cash = self.fixed_unpaid.copy()
        stopped, complete = self.stopped, self.history_complete
        for j, close in enumerate(closes, first):
            if stopped:
                break
            if np.isnan(close):
                complete = False
            elif terms.knocked_out(close):
                stopped = True
            elif complete:
                cash[terms.payment_indices[j]] += terms.fixing_cash(close, strike_ratio)
        due = terms.payment_times <= elapsed + _TIME_TOL
        paid = float(np.sum(cash[due]))
        cash[due] = 0
        return replace(
            self,
            asof=asof,
            elapsed=elapsed,
            fixed_unpaid=cash,
            stopped=stopped,
            history_complete=complete,
        ), paid
