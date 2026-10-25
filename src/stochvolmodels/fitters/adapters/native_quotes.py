"""Native quote scales and conditional additive-price admissibility diagnostics.

The caller supplies quoted units, recorded ticks and physical bounds; this module
does not infer exchange tick rules, reconstruct market observations or convert
premiums. Reports retain every input row, including missing or invalid evidence.
An optional analytic helper supplies bounds only for the explicitly declared
forward-normalized vanilla coordinate q = p / (discount * forward). Open shock
intervals are conditional arithmetic, not noise probabilities or trading policy.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

from stochvolmodels.fitters.diagnostics import DiagnosticReport


def normalized_vanilla_price_bounds(
    forward: float | np.ndarray, strikes: ArrayLike, optiontypes: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Return vanilla bounds in the explicit coordinate q = p / (D * F).

    Parameters
    ----------
    forward : float or array
        Positive forward F, scalar or one per strike, in the strike's units.
    strikes : array-like
        One-dimensional positive finite strike vector.
    optiontypes : array-like
        Aligned vector containing only ``C`` and ``P``.

    Returns
    -------
    tuple of arrays
        Lower and upper normalized-price bounds. Positive finite implied-vol
        inversion requires strictly interior prices. This coordinate is not
        asserted to equal a provider's coin premium; that mapping is the
        caller's explicit convention. No spot or contract multiplier is applied.
    """
    strike = np.asarray(strikes, dtype=float)
    sides = np.asarray(optiontypes)
    anchor = np.asarray(forward, dtype=float)
    if strike.ndim != 1 or sides.shape != strike.shape:
        raise ValueError("strikes and optiontypes must be aligned one-dimensional vectors")
    if anchor.ndim > 1 or (anchor.ndim == 1 and anchor.shape != strike.shape):
        raise ValueError("forward must be scalar or align with strikes")
    if (np.any(~np.isfinite(strike) | (strike <= 0))
            or np.any(~np.isfinite(anchor) | (anchor <= 0))):
        raise ValueError("forward and strikes must be finite and positive")
    if not np.isin(sides, ["C", "P"]).all():
        raise ValueError("optiontypes must contain only C and P")
    with np.errstate(over="ignore", invalid="ignore"):
        ratio = strike / anchor
    if not np.isfinite(ratio).all():
        raise ValueError("strike/forward must be finite")
    calls = sides == "C"
    return (
        np.where(calls, np.maximum(1.0 - ratio, 0.0), np.maximum(ratio - 1.0, 0.0)),
        np.where(calls, 1.0, ratio),
    )


def _vectors(values: Mapping[str, ArrayLike]) -> dict[str, np.ndarray]:
    """Copy flat input vectors, refusing implicit broadcasting of quote evidence."""
    result = {name: np.asarray(value, dtype=float).copy() for name, value in values.items()}
    if any(value.ndim != 1 for value in result.values()):
        raise ValueError("native quote inputs must be one-dimensional vectors")
    if len({len(value) for value in result.values()}) != 1:
        raise ValueError("native quote inputs must have identical lengths")
    return result


def _quote_reason(bid: float, ask: float) -> str:
    """Preserve known quote failures separately from absent observations."""
    if not np.isfinite(bid) or not np.isfinite(ask):
        return "missing_or_nonfinite_quote"
    if min(bid, ask) < 0:
        return "invalid_quote"
    if bid > ask:
        return "crossed_quote"
    return "valid"


def diagnose_native_quotes(
    *,
    bid_prices: ArrayLike,
    ask_prices: ArrayLike,
    tick_sizes: ArrayLike,
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    price_unit: str,
    quote_ids: Sequence[str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> DiagnosticReport:
    """Observe native scales and open bounds for an equal shift of both endpoints.

    Parameters
    ----------
    bid_prices, ask_prices, tick_sizes : array-like
        Aligned flat vectors in ``price_unit``. Ticks are actual caller-supplied
        positive increments, never inferred from quote levels. Missing ticks
        leave scale evidence unavailable, including when the spread is nonzero.
    lower_bounds, upper_bounds : array-like
        Aligned physical price bounds in the same unit. The caller owns the
        payoff, discount and normalization convention. Nonfinite or unordered
        bounds are retained as unavailable; equality of a quote and a bound
        fails the strictly open inversion domain.
    price_unit : str
        Explicit common price unit, e.g. ``coin per unit``.
    quote_ids : sequence of str, optional
        Unique aligned source identities. Defaults to input row ordinals.
    metadata : mapping, optional
        JSON-compatible provenance, copied without interpretation.

    Returns
    -------
    DiagnosticReport
        ``quotes`` and ``summary`` tables. Scale s is max((ask-bid)/2, tick/2).
        For q_bid+s*u and q_ask+s*u, admissibility is the open interval
        (lower-q_bid)/s < u < (upper-q_ask)/s. Intervals are reported only when
        the original quotes and scales are admissible. ``pass`` is conditional
        input arithmetic; freshness, parity and model accuracy are not checked.

        Group ``spread_scale`` by expiry and pass the vectors as ``spread_floor``
        to ``diagnose_fit`` with the same quotes, units and identities. Since the
        effective scale already dominates each half-spread, it is unchanged by
        that operation. Null scales restore to NaN and remain unavailable.
    """
    if not isinstance(price_unit, str) or not price_unit.strip():
        raise ValueError("price_unit must explicitly name the common quote unit")
    arrays = _vectors(dict(bid_price=bid_prices, ask_price=ask_prices, tick_size=tick_sizes,
                           lower_price_bound=lower_bounds, upper_price_bound=upper_bounds))
    count = len(arrays["bid_price"])
    ids = [str(i) for i in range(count)] if quote_ids is None else [str(i) for i in quote_ids]
    if len(ids) != count or len(set(ids)) != count:
        raise ValueError("quote_ids must align with inputs and be unique")
    rows = []
    for i, identity in enumerate(ids):
        row = {name: values[i] for name, values in arrays.items()}
        bid, ask, tick = row["bid_price"], row["ask_price"], row["tick_size"]
        low, high = row["lower_price_bound"], row["upper_price_bound"]
        market = _quote_reason(bid, ask)
        tick_reason = ("missing_or_nonfinite_tick" if not np.isfinite(tick) else
                       "nonpositive_tick" if tick <= 0 else "valid")
        scale_reason = market if market != "valid" else tick_reason
        scale_status = ("pass" if scale_reason == "valid" else
                        "unavailable" if scale_reason.startswith("missing") else "fail")
        half = (ask - bid) / 2 if market == "valid" else np.nan
        floor = tick / 2 if tick_reason == "valid" else np.nan
        scale = max(half, floor) if scale_status == "pass" else np.nan
        if scale_status == "pass" and (not np.isfinite(scale) or scale <= 0):
            scale_status, scale_reason = "unavailable", "unrepresentable_scale"
            scale = np.nan
        bounds_valid = bool(np.isfinite(low) and np.isfinite(high) and low < high)
        outside = market == "valid" and bounds_valid and (bid <= low or ask >= high)
        if market in ("invalid_quote", "crossed_quote"):
            status, reason = "fail", market
        elif outside:
            status, reason = "fail", "outside_open_price_bounds"
        elif not bounds_valid:
            status, reason = "unavailable", "invalid_or_missing_price_bounds"
        else:
            status, reason = scale_status, scale_reason
        lower_coefficient, upper_coefficient = np.nan, np.nan
        if status == "pass":
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                lower_coefficient, upper_coefficient = (low - bid) / scale, (high - ask) / scale
            if not np.isfinite([lower_coefficient, upper_coefficient]).all():
                status, reason = "unavailable", "unrepresentable_coefficient_interval"
                lower_coefficient, upper_coefficient = np.nan, np.nan
        row.update(
            quote_id=identity, half_spread=half, spread_floor=floor, spread_scale=scale,
            scale_status=scale_status, scale_reason=scale_reason,
            admissibility_status=status, admissibility_reason=reason,
            lower_coefficient=lower_coefficient, upper_coefficient=upper_coefficient,
        )
        rows.append(row)
    columns = [*arrays, "quote_id", "half_spread", "spread_floor", "spread_scale",
               "scale_status", "scale_reason", "admissibility_status", "admissibility_reason",
               "lower_coefficient", "upper_coefficient"]
    quotes = pd.DataFrame(rows, columns=columns)
    summary = {"n_quotes": count}
    for facet in ("scale", "admissibility"):
        for status in ("pass", "fail", "unavailable"):
            summary[f"n_{facet}_{status}"] = int(quotes[f"{facet}_status"].eq(status).sum())
    return DiagnosticReport.from_tables(
        {"quotes": quotes, "summary": pd.DataFrame([summary])},
        dict(kind="native_quote_admissibility", price_unit=price_unit,
             scale_rule="max(half_spread, recorded_tick/2)",
             coefficient_interval="strictly open", coefficient_unit="dimensionless",
             bounds_source="caller supplied in price_unit", probability_model=None,
             automatic_policy=None, provenance=dict(metadata or {})),
    )
