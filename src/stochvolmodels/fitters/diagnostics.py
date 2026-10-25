"""Immutable fit diagnostics in explicitly declared price units.

These reports observe final model outputs; they never fit, filter or repair a chain.
Vanilla pricing and quote normalization remain owned by their existing providers.
Inputs contain one model-price or decimal-IV vector per OptionChain expiry, with
weights, scalar or per-quote scale floors, and quote identities aligned to strikes.
Missing per-quote floors withhold only normalized residuals. Reports snapshot quote residuals
and expiry/chain aggregates as strict JSON, retaining explicit price units and
failure reasons when non-finite results serialize as null.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping, Sequence, TypeAlias

import numpy as np
import pandas as pd

from stochvolmodels.data.option_chain import OptionChain


_JsonValue: TypeAlias = (
    str | int | float | bool | None | list["_JsonValue"] | dict[str, "_JsonValue"]
)


def _json_value(value: object) -> _JsonValue:
    """Copy supported metadata and table values into strict JSON primitives."""
    if value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported report value: {type(value).__name__}")


@dataclass(frozen=True)
class DiagnosticReport:
    """An immutable JSON snapshot; table and metadata access returns fresh copies.

    Construct with ``from_tables`` or ``from_dict``. Non-finite numeric results
    serialize as null; status/reason columns distinguish why a result is absent.
    """

    _payload: str

    def __post_init__(self) -> None:
        """Reject malformed serialized reports at construction."""
        data = json.loads(self._payload)
        if data.get("schema_version") != 1 or not isinstance(data.get("metadata"), dict):
            raise ValueError("unsupported diagnostic report schema")
        if not isinstance(data.get("tables"), dict):
            raise ValueError("report tables must be a mapping")
        for table in data["tables"].values():
            columns, rows = table["columns"], table["data"]
            if len(set(columns)) != len(columns) or any(len(row) != len(columns) for row in rows):
                raise ValueError("invalid diagnostic table dimensions")

    @classmethod
    def from_tables(
        cls, tables: Mapping[str, pd.DataFrame], metadata: Mapping[str, object]
    ) -> DiagnosticReport:
        """Snapshot tables and JSON-compatible metadata without retaining mutable inputs."""
        data = {
            "schema_version": 1,
            "metadata": metadata,
            "tables": {
                name: {
                    "columns": list(frame.columns),
                    "data": frame.to_numpy(dtype=object).tolist(),
                }
                for name, frame in tables.items()
            },
        }
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> DiagnosticReport:
        """Restore a report from its strict JSON representation."""
        return cls(json.dumps(_json_value(data), allow_nan=False, sort_keys=True))

    def to_dict(self) -> dict[str, _JsonValue]:
        """Return an independent JSON-compatible representation."""
        return json.loads(self._payload)

    @property
    def metadata(self) -> dict[str, _JsonValue]:
        """Return an independent copy of conventions and provenance."""
        return self.to_dict()["metadata"]

    def table(self, name: str) -> pd.DataFrame:
        """Return a fresh table; unknown names raise KeyError."""
        table = self.to_dict()["tables"][name]
        return pd.DataFrame(table["data"], columns=table["columns"])


def _aligned(
    chain: OptionChain,
    values: Sequence[np.ndarray] | None,
    name: str,
    *,
    fill: float = np.nan,
) -> list[np.ndarray]:
    """Validate one vector per expiry and copy all input arrays."""
    if values is None:
        return [np.full(len(strikes), fill, dtype=float) for strikes in chain.strikes_ttms]
    if len(values) != len(chain.ttms):
        raise ValueError(f"{name} must contain one vector per expiry")
    result = []
    for strikes, row in zip(chain.strikes_ttms, values):
        array = np.asarray(row, dtype=float)
        if array.ndim != 1 or len(array) != len(strikes):
            raise ValueError(f"{name} must align with the chain strikes")
        result.append(array.copy())
    return result


def _market_status(bid: float, ask: float, *, positive: bool) -> str:
    """Classify quote intervals without treating zero bids as invalid prices."""
    if not np.isfinite(bid) or not np.isfinite(ask):
        return "missing_or_nonfinite_quote"
    if min(bid, ask) < 0 or (positive and min(bid, ask) == 0):
        return "invalid_quote"
    if bid > ask:
        return "crossed_quote"
    return "valid"


def _fit_summary(frame: pd.DataFrame, label: str) -> dict[str, str | int | float | None]:
    """Aggregate raw residuals, preserving actual weight sums across expiries."""
    result = {"expiry_id": label, "n_quotes": len(frame)}
    for kind in ("iv", "price"):
        valid = frame[f"{kind}_reason"].eq("valid")
        subset = frame.loc[valid]
        residuals = subset[f"{kind}_residual"].to_numpy(dtype=float)
        weights = subset["weight"].to_numpy(dtype=float)
        total_weight = float(weights.sum())
        result.update(
            {
                f"n_{kind}": len(subset),
                f"n_{kind}_excluded": len(frame) - len(subset),
                f"{kind}_status": "pass" if len(subset) else "not_checked",
                f"{kind}_weight_sum": total_weight,
                f"{kind}_rmse": float(np.sqrt(np.mean(residuals**2))) if len(subset) else np.nan,
                f"{kind}_weighted_rmse": float(
                    np.sqrt(np.dot(weights, residuals**2) / total_weight)
                )
                if total_weight > 0
                else np.nan,
                f"{kind}_weighted_status": "pass" if total_weight > 0 else "not_checked",
                f"{kind}_max_abs_error": float(np.max(np.abs(residuals)))
                if len(subset)
                else np.nan,
                f"{kind}_worst_quote_id": str(subset.iloc[np.argmax(np.abs(residuals))]["quote_id"])
                if len(subset)
                else None,
            }
        )
    prices = frame.loc[frame.price_reason.eq("valid")]
    result["price_inside_bid_ask_fraction"] = (
        float(prices.inside_bid_ask.mean()) if len(prices) else np.nan
    )
    result["n_locked_prices"] = int(
        frame.price_market.eq("valid").mul(frame.bid_price.eq(frame.ask_price)).sum()
    )
    result["n_one_sided_prices"] = int(
        (np.isfinite(frame.bid_price) ^ np.isfinite(frame.ask_price)).sum()
    )
    result["n_missing_prices"] = int(frame.price_market.eq("missing_or_nonfinite_quote").sum())
    result["n_crossed_prices"] = int(frame.price_market.eq("crossed_quote").sum())
    return result


def diagnose_fit(
    chain: OptionChain,
    *,
    model_prices: Sequence[np.ndarray] | None = None,
    model_ivols: Sequence[np.ndarray] | None = None,
    price_unit: str,
    spread_floor: float | Sequence[np.ndarray],
    weights: Sequence[np.ndarray] | None = None,
    weight_label: str | None = None,
    quote_ids: Sequence[Sequence[str]] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> DiagnosticReport:
    """Measure final fitted values against chain bid/ask intervals.

    Parameters
    ----------
    chain : OptionChain
        Quotes, strikes and maturities. Quote fields may be absent.
    model_prices, model_ivols : sequence of arrays, optional
        One aligned vector per expiry. At least one must be supplied. Prices
        must already share the quoted units; IVs are decimal. No conversion or
        pricing is performed here. Invalid outputs are counted, not clipped.
    price_unit : str
        Explicit common unit of model and quoted prices, e.g. ``USD per unit``.
    spread_floor : float or sequence of arrays
        Positive floor for the half-spread denominator, in price_unit. A scalar
        applies to every quote; vectors align with each expiry's strikes. NaN
        vector entries explicitly mark unavailable scales and withhold normalized
        errors without discarding raw price residuals. Inf, zero and negative
        floors are rejected. An already computed effective scale may be supplied
        if it dominates the same quote's half-spread; it remains unchanged.
    weights : sequence of arrays, optional
        Finite nonnegative weights. Zero weights exclude only weighted metrics.
    weight_label : str, optional
        Required description when weights are supplied; otherwise ``uniform``.
    quote_ids : sequence of sequences of str, optional
        Unique IDs aligned to the input. Defaults to expiry/row ordinal IDs.
    metadata : mapping, optional
        JSON-compatible provenance stored separately from metric conventions.

    Returns
    -------
    DiagnosticReport
        ``quotes``, ``expiries`` and ``summary`` tables. Metric status ``pass``
        means evaluated, not economically acceptable. Coverage is a fraction.
        Both aggregate RMSEs are computed from individual observations.
    """
    if model_prices is None and model_ivols is None:
        raise ValueError("provide model_prices or model_ivols")
    if not isinstance(price_unit, str) or not price_unit.strip():
        raise ValueError("price_unit must explicitly name the common price unit")
    if spread_floor is None:
        raise ValueError("spread_floor must be scalar or contain one vector per expiry")
    if np.isscalar(spread_floor) or (
        isinstance(spread_floor, np.ndarray) and spread_floor.ndim == 0
    ):
        if not np.isfinite(spread_floor) or spread_floor <= 0:
            raise ValueError("spread_floor must be finite and positive")
        floor_arrays = _aligned(chain, None, "spread_floor", fill=float(spread_floor))
        floor_metadata = float(spread_floor)
    else:
        floor_arrays = _aligned(chain, spread_floor, "spread_floor")
        if any(np.any(np.isinf(row) | (row <= 0)) for row in floor_arrays):
            raise ValueError("spread_floor vectors must contain positive values or NaN")
        floor_metadata = floor_arrays
    if weights is not None and (not isinstance(weight_label, str) or not weight_label.strip()):
        raise ValueError("explicit weights require weight_label")
    arrays = {
        name: _aligned(chain, value, name)
        for name, value in {
            "model_price": model_prices,
            "model_iv": model_ivols,
            "bid_price": chain.bid_prices,
            "ask_price": chain.ask_prices,
            "bid_iv": chain.bid_ivs,
            "ask_iv": chain.ask_ivs,
        }.items()
    }
    weight_arrays = _aligned(chain, weights, "weights", fill=1.0)
    if any(np.any(~np.isfinite(row) | (row < 0)) for row in weight_arrays):
        raise ValueError("weights must be finite and nonnegative")
    if quote_ids is not None and (
        len(quote_ids) != len(chain.ttms)
        or any(len(ids) != len(strikes) for ids, strikes in zip(quote_ids, chain.strikes_ttms))
    ):
        raise ValueError("quote_ids must align with the chain")
    rows = []
    for i, strikes in enumerate(chain.strikes_ttms):
        for j, strike in enumerate(strikes):
            row = {name: values[i][j] for name, values in arrays.items()}
            row.update(
                expiry_index=i,
                expiry_id=str(chain.ids[i]) if chain.ids is not None else str(i),
                quote_id=str(quote_ids[i][j]) if quote_ids is not None else f"{i}:{j}",
                strike=float(strike),
                optiontype=str(chain.optiontypes_ttms[i][j]),
                ttm=float(chain.ttms[i]),
                forward=float(chain.forwards[i]),
                discount=float(chain.discfactors[i]),
                weight=float(weight_arrays[i][j]),
            )
            for kind in ("iv", "price"):
                bid, ask, model = row[f"bid_{kind}"], row[f"ask_{kind}"], row[f"model_{kind}"]
                market = _market_status(bid, ask, positive=kind == "iv")
                reason = market
                if market == "valid" and (
                    not np.isfinite(model) or model < 0 or (kind == "iv" and model == 0)
                ):
                    reason = "invalid_or_missing_model"
                row[f"{kind}_market"] = market
                row[f"{kind}_reason"] = reason
                row[f"{kind}_residual"] = model - (bid + ask) / 2 if reason == "valid" else np.nan
            valid = row["price_reason"] == "valid"
            bid, ask, model = row["bid_price"], row["ask_price"], row["model_price"]
            row["inside_bid_ask"] = bool(bid <= model <= ask) if valid else None
            row["outside_spread"] = max(bid - model, model - ask, 0.0) if valid else np.nan
            floor = floor_arrays[i][j]
            scale_reason = row["price_market"] if row["price_market"] != "valid" else (
                "missing_spread_floor" if np.isnan(floor) else "valid"
            )
            row["spread_floor"] = floor
            row["scale_reason"] = scale_reason
            row["scale_status"] = (
                "pass" if scale_reason == "valid" else
                "unavailable" if scale_reason.startswith("missing") else "fail"
            )
            row["spread_scale"] = (
                max((ask - bid) / 2, floor) if scale_reason == "valid" else np.nan
            )
            row["spread_scaled_error"] = (
                row["price_residual"] / row["spread_scale"]
                if valid and scale_reason == "valid" else np.nan
            )
            rows.append(row)
    quotes = pd.DataFrame(rows)
    if quotes.quote_id.duplicated().any():
        raise ValueError("quote_ids must be unique across the chain")
    expiries = pd.DataFrame(
        [
            _fit_summary(frame, str(frame.expiry_id.iloc[0]))
            for _, frame in quotes.groupby("expiry_index", sort=True)
        ]
    )
    summary = pd.DataFrame([_fit_summary(quotes, "all")])
    return DiagnosticReport.from_tables(
        {"quotes": quotes, "expiries": expiries, "summary": summary},
        {
            "kind": "fit",
            "price_unit": price_unit,
            "iv_unit": "decimal",
            "ttm_unit": "years",
            "spread_floor": floor_metadata,
            "weighting": weight_label if weights is not None else "uniform",
            "coverage_denominator": "finite model prices with valid two-sided price quotes",
            "provenance": dict(metadata or {}),
        },
    )
