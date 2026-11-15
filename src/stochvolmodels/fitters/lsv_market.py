"""Prepare ragged European Black-IV quotes for normalized LSV calibration and Dupire.

Examples
--------
Inspect the normalized BTC market and the number of Dupire nodes repaired in its interior.

>>> from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
>>> from stochvolmodels.fitters.dupire import DupireRepairPolicy
>>> from stochvolmodels.fitters.lsv_market import LSVOptionChainAdapter
>>> market = LSVOptionChainAdapter(get_btc_test_chain_data()).build_calibration_input(grid_size=11)
>>> result = market.recover_dupire(repair_policy=DupireRepairPolicy(0.04, 4.0))
>>> market.target.target_calls.shape, int((result.repaired_mask & result.interior_mask).sum())
((4, 11), 0)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.fitters.dupire import (
    CallSurface,
    DupireRecoverer,
    DupireRepairPolicy,
    DupireResult,
)
from stochvolmodels.fitters.lsv_parametric import ImpliedVolCalibrationTarget
from stochvolmodels.models.local_vol import _readonly_array

FloatArray = NDArray[np.float64]


class LSVQuoteSelection(str, Enum):
    """OTM uses puts below forward and calls at or above forward. ALL retains both sides."""

    OTM = "otm"
    ALL = "all"


@dataclass(frozen=True, slots=True)
class MarketQuoteSlice:
    """Original-order quote snapshot and the indices used for interpolation, sorted by strike."""

    slice_id: str
    maturity: float
    forward: float
    discount_factor: float
    strikes: FloatArray
    option_types: NDArray[np.str_]
    bid_ivs: FloatArray
    ask_ivs: FloatArray
    selected_indices: NDArray[np.intp]
    selected_log_moneyness: FloatArray


@dataclass(frozen=True, slots=True)
class LSVCalibrationInput:
    """Original quotes and normalized bid, ask and mid Black IVs on a common grid."""

    source_slices: tuple[MarketQuoteSlice, ...]
    quote_selection: LSVQuoteSelection
    log_moneyness: FloatArray
    bid_ivs: FloatArray
    ask_ivs: FloatArray
    target: ImpliedVolCalibrationTarget

    def recover_dupire(self, *, repair_policy: DupireRepairPolicy | None = None) -> DupireResult:
        """Recover local variance at fixed K/F, retaining raw calls, variances and repair masks.

        The target calls are PV / (D F) with unit forward and discount. At least three
        maturities and three distinct grid points are required for Dupire differentiation.
        """
        calls = CallSurface(self.target.times, self.target.strikes, self.target.target_calls)
        return DupireRecoverer().continuous(calls, repair_policy=repair_policy)


@dataclass(frozen=True, slots=True, init=False)
class LSVOptionChainAdapter:
    """Snapshot ordinary European bid/ask Black IVs for fitting on overlapping log(K/F)."""

    source_slices: tuple[MarketQuoteSlice, ...]
    quote_selection: LSVQuoteSelection

    def __init__(
        self, option_chain: OptionChain, quote_selection: LSVQuoteSelection = LSVQuoteSelection.OTM
    ) -> None:
        """Copy each original expiry and select the ordered source quotes for interpolation."""
        if not isinstance(option_chain, OptionChain):
            raise TypeError("option_chain must be an OptionChain")
        selection = LSVQuoteSelection(quote_selection)
        if option_chain.bid_ivs is None or option_chain.ask_ivs is None:
            raise ValueError("OptionChain bid and ask implied volatilities are required")
        slices = tuple(
            _snapshot_slice(option_chain, i, selection) for i in range(len(option_chain.ttms))
        )
        object.__setattr__(self, "source_slices", slices)
        object.__setattr__(self, "quote_selection", selection)

    @property
    def _common_domain(self) -> tuple[float, float]:
        """Intersection of the selected observed log-moneyness intervals."""
        lower = max(s.selected_log_moneyness[0] for s in self.source_slices)
        upper = min(s.selected_log_moneyness[-1] for s in self.source_slices)
        if lower >= upper:
            raise ValueError("selected quote slices have no common log-moneyness domain")
        return lower, upper

    def common_log_moneyness_grid(self, size: int = 21) -> FloatArray:
        """Space nodes across the common observed domain, pinning the nearest node to ATM."""
        if isinstance(size, bool) or not isinstance(size, Integral) or size < 2:
            raise ValueError("size must be an integer of at least two")
        lower, upper = self._common_domain
        nodes = np.linspace(lower, upper, size)
        if lower <= 0 <= upper:
            nodes[np.argmin(np.abs(nodes))] = 0.0
        if np.any(np.diff(nodes) <= 0):
            raise ValueError("grid cannot resolve distinct log-moneyness nodes within the overlap")
        nodes.setflags(write=False)
        return nodes

    def build_calibration_input(
        self, log_moneyness: ArrayLike | None = None, *, grid_size: int = 21
    ) -> LSVCalibrationInput:
        """Interpolate Black IVs within the common domain, snapping roundoff at its endpoints."""
        if log_moneyness is None:
            nodes = self.common_log_moneyness_grid(grid_size)
        else:
            nodes = _readonly_array(log_moneyness, name="log_moneyness", ndim=1)
            if nodes.size < 2 or np.any(np.diff(nodes) <= 0):
                raise ValueError("log_moneyness must contain at least two increasing points")
            lower, upper = self._common_domain
            tolerance = 64 * np.finfo(float).eps * max(1.0, abs(lower), abs(upper))
            if nodes[0] < lower - tolerance or nodes[-1] > upper + tolerance:
                raise ValueError("log_moneyness lies outside the common observed domain")
            if nodes[0] < lower or nodes[-1] > upper:
                nodes = np.clip(nodes, lower, upper)
                if np.any(np.diff(nodes) <= 0):
                    raise ValueError("log_moneyness nodes coincide at an observed domain endpoint")
                nodes.setflags(write=False)

        bids, asks, mids = [], [], []
        for row in self.source_slices:
            x = row.selected_log_moneyness
            bid, ask = row.bid_ivs[row.selected_indices], row.ask_ivs[row.selected_indices]
            bids.append(np.interp(nodes, x, bid))
            asks.append(np.interp(nodes, x, ask))
            mids.append(np.interp(nodes, x, 0.5 * (bid + ask)))
        target = ImpliedVolCalibrationTarget(
            [row.maturity for row in self.source_slices], np.exp(nodes), mids
        )
        return LSVCalibrationInput(
            self.source_slices,
            self.quote_selection,
            nodes,
            _readonly_array(bids, name="bid_ivs", ndim=2),
            _readonly_array(asks, name="ask_ivs", ndim=2),
            target,
        )


def _snapshot_slice(
    chain: OptionChain, index: int, selection: LSVQuoteSelection
) -> MarketQuoteSlice:
    """Detach a full market slice and retain the permutation from normalized to original quotes."""
    t, f, d = (
        float(chain.ttms[index]),
        float(chain.forwards[index]),
        float(chain.discfactors[index]),
    )
    label = str(chain.ids[index]) if chain.ids is not None else f"T={t:.6g}"
    strikes = _readonly_array(chain.strikes_ttms[index], name="strikes", ndim=1)
    kinds = np.array(chain.optiontypes_ttms[index], dtype=str, copy=True)
    bids = _readonly_array(chain.bid_ivs[index], name="bid_ivs", ndim=1)
    asks = _readonly_array(chain.ask_ivs[index], name="ask_ivs", ndim=1)
    if not (strikes.shape == kinds.shape == bids.shape == asks.shape):
        raise ValueError(f"{label}: strikes, option types, bids and asks must align")
    if not np.all(np.isin(kinds, ["C", "P"])):
        raise ValueError(f"{label}: only standard C/P quotes are supported")
    if not (np.isfinite([t, f, d]).all() and min(t, f, d) > 0 and np.all(strikes > 0)):
        raise ValueError(f"{label}: maturity, forward, discount and strikes must be positive")
    if np.any(bids <= 0) or np.any(bids > asks):
        raise ValueError(f"{label}: bid/ask IVs must be positive and ordered")
    indices = np.arange(len(strikes))
    if selection is LSVQuoteSelection.OTM:
        indices = indices[((kinds == "P") & (strikes < f)) | ((kinds == "C") & (strikes >= f))]
    if indices.size < 2:
        raise ValueError(f"{label}: fewer than two selected quotes")
    indices = indices[np.argsort(strikes[indices], kind="stable")]
    x = _readonly_array(np.log(strikes[indices] / f), name="selected_log_moneyness", ndim=1)
    if np.any(np.diff(x) <= 0):
        raise ValueError(f"{label}: duplicate normalized strikes among selected quotes")
    kinds.setflags(write=False)
    indices.setflags(write=False)
    return MarketQuoteSlice(label, t, f, d, strikes, kinds, bids, asks, indices, x)
