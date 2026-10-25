"""Convert OCA quotes and their quality evidence into SVM diagnostic inputs.

Selection reuses the legacy point-in-time adapter, then applies the caller's
explicit price-unit conversion and joins source-row provenance when supplied.
The returned OptionChain and immutable DiagnosticReport retain maturity, quote,
and timestamp alignment; quality flags never silently alter quote selection.
OptionChainAnalytics remains optional and is imported only when an adapter runs.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Mapping

import numpy as np
import pandas as pd

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.fitters.adapters.oca import option_chain_from_oca
from stochvolmodels.fitters.diagnostics import DiagnosticReport

if TYPE_CHECKING:
    from option_chain_analytics import SlicesChain
    from option_chain_analytics.diagnostics import QuoteDiagnosticReport


def diagnostic_chain_from_oca(
    chain: SlicesChain,
    *,
    price_conversion: str,
    price_unit: str,
    value_time: pd.Timestamp | None = None,
    days_map: dict[str, int] | None = None,
    delta_bounds: tuple[float | None, float | None] = (-0.1, 0.1),
    is_filtered: bool = True,
    quality_metadata: Mapping[str, object] | None = None,
    quality_report: QuoteDiagnosticReport | None = None,
) -> tuple[OptionChain, DiagnosticReport]:
    """Return an SVM chain and selected-quote provenance from a reconstructed OCA chain.

    Parameters
    ----------
    chain : option_chain_analytics.SlicesChain
        Already reconstructed observations. No new snapshot selection is performed.
    price_conversion : {'quoted', 'usd_per_unit'}
        Explicit unit policy. The latter multiplies bid/ask by OCA USD_MULTIPLIER
        once, without multiplying by CONTRACT_SIZE. This is a unit conversion,
        not verification of the provider's payoff or valuation measure.
    price_unit : str
        Resulting price unit; must be ``USD per unit`` for usd_per_unit conversion.
    value_time, days_map, delta_bounds, is_filtered
        Selection arguments of the existing option_chain_from_oca adapter.
    quality_metadata : mapping, optional
        Caller-supplied OCA diagnostic provenance, copied without interpretation.
        Its absence is explicitly recorded; this helper does not create OCA checks.
    quality_report : option_chain_analytics.diagnostics.QuoteDiagnosticReport, optional
        Reconstruction sidecar from OCA 5.3 or later. Join its source-row identities,
        quality reasons and upstream exclusions to the selected SVM rows. Units,
        timestamps and selected quote values must agree. Cannot combine with
        quality_metadata. Quality flags do not change the legacy selection policy.

    Returns
    -------
    tuple of OptionChain and DiagnosticReport
        A new chain and ``quotes``/``expiries`` provenance tables. Quote IDs match
        diagnose_fit's default expiry/row IDs. Pre-reconstruction duplicate counts
        remain unavailable, rather than being inferred from a filtered chain.
    """
    if price_conversion not in ("quoted", "usd_per_unit"):
        raise ValueError("price_conversion must be quoted or usd_per_unit")
    if not isinstance(price_unit, str) or not price_unit.strip():
        raise ValueError("price_unit must be explicit")
    if price_conversion == "usd_per_unit" and price_unit != "USD per unit":
        raise ValueError('usd_per_unit conversion requires price_unit="USD per unit"')
    if quality_report is not None and quality_metadata is not None:
        raise ValueError("supply quality_report or quality_metadata, not both")
    result = option_chain_from_oca(
        chain,
        value_time=value_time,
        days_map=days_map,
        delta_bounds=delta_bounds,
        is_filtered=is_filtered,
    )
    # The existing adapter checks the optional dependency and container type first.
    from option_chain_analytics import SliceColumn

    source_time = pd.Timestamp(chain.value_time)
    requested_time = source_time if value_time is None else pd.Timestamp(value_time)
    if source_time.tzinfo is None or requested_time.tzinfo is None:
        raise ValueError("source and requested timestamps must be timezone-aware")
    if source_time > requested_time:
        raise ValueError("source observation must not be later than requested time")
    quote_rows, expiry_rows, bids, asks = [], [], [], []
    for i, ttm in enumerate(result.ttms):
        matches = [
            s
            for s in chain.expiry_slices.values()
            if np.isclose(s.get_ttm(), ttm, rtol=0, atol=1e-12)
        ]
        if len(matches) != 1:
            raise ValueError("cannot uniquely map selected maturity back to OCA")
        source = matches[0]
        selected = source.get_joint_slice(delta_bounds=delta_bounds, is_filtered=is_filtered)
        if not np.array_equal(
            selected.index.to_numpy(), result.strikes_ttms[i]
        ) or not np.array_equal(
            selected[SliceColumn.OPTION_TYPE].to_numpy(), result.optiontypes_ttms[i]
        ):
            raise ValueError("OCA selection changed during conversion")
        if price_conversion == "usd_per_unit":
            scales = selected[SliceColumn.USD_MULTIPLIER].to_numpy(dtype=float)
            if np.any(~np.isfinite(scales) | (scales <= 0)):
                raise ValueError("USD multiplier must be finite and positive")
        else:
            scales = np.ones(len(selected))
        bids.append(result.bid_prices[i] * scales)
        asks.append(result.ask_prices[i] * scales)
        expiry_rows.append(
            dict(
                expiry_index=i,
                expiry_id=str(source.expiry_id),
                n_source_rows=len(source.options_df),
                n_selected=len(selected),
                n_excluded=len(source.options_df) - len(selected),
                discount_source="median_of_selected_quotes"
                if SliceColumn.DISCOUNT in selected
                else "legacy_default_one",
            )
        )
        for j, (_, row) in enumerate(selected.iterrows()):
            quote_rows.append(
                dict(
                    quote_id=f"{i}:{j}",
                    expiry_index=i,
                    expiry_id=str(source.expiry_id),
                    contract=str(row[SliceColumn.CONTRACT]),
                    strike=float(row[SliceColumn.STRIKE]),
                    optiontype=str(row[SliceColumn.OPTION_TYPE]),
                    source_time=source_time,
                    exchange_time=row.get(SliceColumn.EXCHANGE_TIME),
                    price_scale=float(scales[j]),
                    contract_size=row.get(SliceColumn.CONTRACT_SIZE),
                    usd_multiplier=row.get(SliceColumn.USD_MULTIPLIER),
                    bid_price=float(bids[i][j]),
                    ask_price=float(asks[i][j]),
                )
            )
    result = replace(result, bid_prices=bids, ask_prices=asks)
    tables = {"quotes": pd.DataFrame(quote_rows), "expiries": pd.DataFrame(expiry_rows)}
    if quality_report is not None:
        tables = _attach_quality(
            tables,
            quality_report,
            chain,
            source_time,
            requested_time,
            price_unit,
            is_filtered,
            delta_bounds,
        )
        quality_metadata = quality_report.to_dict()
    report = DiagnosticReport.from_tables(
        tables,
        {
            "kind": "oca_provenance",
            "price_unit": price_unit,
            "price_conversion": price_conversion,
            "source_time": source_time,
            "requested_time": requested_time,
            "ttm_source": "OCA slice years, not recomputed",
            "year_basis": quality_report.metadata["config"]["year_basis"]
            if quality_report is not None
            else "not supplied by chain",
            "is_filtered": is_filtered,
            "delta_bounds": delta_bounds,
            "pre_reconstruction_counts": "provided"
            if quality_report is not None
            else "not_available",
            "quality_status": "joined"
            if quality_report is not None
            else ("provided" if quality_metadata is not None else "not_checked"),
            "quality_metadata": dict(quality_metadata or {}),
            "payoff_mapping": "not_verified_by_unit_conversion",
        },
    )
    return result, report


def _attach_quality(
    tables: dict[str, pd.DataFrame],
    quality: QuoteDiagnosticReport,
    chain: SlicesChain,
    source_time: pd.Timestamp,
    requested_time: pd.Timestamp,
    price_unit: str,
    is_filtered: bool,
    delta_bounds: tuple[float | None, float | None] | None,
) -> dict[str, pd.DataFrame]:
    """Validate and join OCA sidecar evidence without changing quote selection."""
    from option_chain_analytics.diagnostics import QuoteDiagnosticReport

    if not isinstance(quality, QuoteDiagnosticReport):
        raise TypeError("quality_report must be an OCA QuoteDiagnosticReport")
    metadata = quality.metadata
    if metadata.get("kind") != "reconstruction_diagnostics":
        raise ValueError("quality_report must include reconstruction provenance")
    for name, actual in (("source_time", source_time), ("requested_time", requested_time)):
        if pd.Timestamp(metadata.get(name)) != actual:
            raise ValueError(f"quality report {name} disagrees with conversion")
    if metadata["config"]["price_unit"] != price_unit:
        raise ValueError("quality report price units disagree with conversion")
    provenance = quality.table("provenance")
    evidence = quality.table("quotes").set_index("source_row_id", verify_integrity=True)
    retained = provenance.loc[provenance.retained.astype(bool)]
    keys = ["contract", "mat_id", "optiontype", "strike"]
    if retained.duplicated(keys).any():
        raise ValueError("ambiguous source-row mapping in quality report")
    lookup = retained.set_index(keys).source_row_id.to_dict()
    selected = tables["quotes"].copy()
    identities, reasons, failed_counts = [], [], []
    for row in selected.itertuples():
        key = (row.contract, row.expiry_id, row.optiontype, row.strike)
        if key not in lookup:
            raise ValueError("selected quote is absent from quality report provenance")
        identity = lookup[key]
        source = evidence.loc[identity]
        if not np.allclose(
            [row.bid_price, row.ask_price],
            np.asarray([source.bid_price, source.ask_price], dtype=float),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError("quality report quote prices disagree with conversion")
        identities.append(identity)
        reasons.append(source.reason_codes)
        failed_counts.append(source.n_failed_checks)
    selected["source_row_id"] = identities
    selected["quality_reason_codes"] = reasons
    selected["quality_failed_checks"] = failed_counts
    selected["quality_status"] = np.where(np.asarray(failed_counts) > 0, "fail", "pass")
    selected_ids = set(identities)
    expiry_ids = set(selected.expiry_id)
    clean_contracts = {}
    complete_sides = {}
    for expiry_id in expiry_ids:
        expiry = chain.expiry_slices[expiry_id]
        calls = expiry.get_call_slice(is_filtered=is_filtered)
        puts = expiry.get_put_slice(is_filtered=is_filtered)
        clean_contracts[expiry_id] = set(pd.concat([calls, puts]).contract)
        complete_sides[expiry_id] = not calls.empty and not puts.empty
    svm_stages, exclusions = [], []
    for row in provenance.itertuples():
        reason = row.exclusion_reason
        stage = row.stage
        if row.retained:
            if row.source_row_id in selected_ids:
                stage, reason = "svm_selected", ""
            elif str(row.mat_id) not in expiry_ids:
                stage, reason = "svm_maturity_selection", "maturity_not_selected"
            else:
                stage = "svm_quote_selection"
                expiry_id = str(row.mat_id)
                source = evidence.loc[row.source_row_id]
                forward = chain.expiry_slices[expiry_id].get_future_price()
                if row.contract not in clean_contracts[expiry_id]:
                    reason = "legacy_quote_filter"
                elif not complete_sides[expiry_id]:
                    reason = "empty_call_or_put_side"
                elif (row.optiontype == "P" and row.strike > forward) or (
                    row.optiontype == "C" and row.strike < forward
                ):
                    reason = "outside_otm_wing"
                elif delta_bounds is not None:
                    bound = delta_bounds[0 if row.optiontype == "P" else 1]
                    delta = float(source.delta) if source.delta is not None else np.nan
                    inside = bound is None or (
                        delta <= bound if row.optiontype == "P" else delta >= bound
                    )
                    reason = "delta_bound" if not inside else "legacy_joint_slice_selection"
                else:
                    reason = "legacy_joint_slice_selection"
        svm_stages.append(stage)
        exclusions.append(reason)
    provenance["svm_stage"] = svm_stages
    provenance["svm_exclusion_reason"] = exclusions
    provenance["svm_selected"] = provenance.source_row_id.isin(selected_ids)
    counts = quality.table("counts").copy()
    counts["n_svm_selected"] = len(selected)
    counts["n_svm_excluded"] = len(provenance) - len(selected)
    counts["svm_reconciled"] = len(provenance) == len(
        selected
    ) + provenance.svm_exclusion_reason.ne("").sum() and len(selected_ids) == len(selected)
    if not counts.svm_reconciled.all():
        raise ValueError("SVM selection counts do not reconcile with OCA source rows")
    return tables | {
        "quotes": selected,
        "source_selection": provenance,
        "counts": counts,
        "oca_checks": quality.table("checks"),
        "oca_parity": quality.table("parity"),
    }
