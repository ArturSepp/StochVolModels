"""Combine existing diagnostics; repository-only research, no accuracy certificate.

Run the core-only synthetic example, or replay a saved raw-SVI reliability report.
Serialized OCA sidecars are consumed without installing the optional OCA dependency.
"""

import argparse
from datetime import datetime, timezone
from enum import Enum
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from numba.typed import List
import pandas as pd

from stochvolmodels import OptionChain, local_path
from stochvolmodels.fitters.diagnostics import DiagnosticReport, diagnose_fit
from _wing_research_policy import assess_research_policy, load_policy


def _report(value, kind):
    """Restore a copied sidecar and verify its declared owner/type."""
    result = DiagnosticReport.from_dict(value.to_dict() if hasattr(value, "to_dict") else value)
    if result.metadata.get("kind") != kind:
        raise ValueError(f"Expected {kind} sidecar")
    return result


def _same_number(left, right):
    """Match nullable numeric observations without inventing missing values."""
    if pd.isna(left) or pd.isna(right):
        return bool(pd.isna(left) and pd.isna(right))
    return bool(np.isclose(float(left), float(right), rtol=1e-10, atol=1e-12))


def _time(metadata):
    """Require aware, ordered source and request times."""
    requested = pd.Timestamp(metadata.get("requested_time"))
    source = (pd.Timestamp(metadata["source_time"])
              if metadata.get("source_time") is not None else None)
    if (pd.isna(requested) or requested.tzinfo is None or source is not None
            and (pd.isna(source) or source.tzinfo is None or source > requested)):
        raise ValueError("Sidecars require aware source/request times without future selection")
    return requested, source


def _quality(chain, quotes, quality, parity, price_unit):
    """Join exact selected quote identities and keep unchecked evidence explicit."""
    facets, tables, metadata = [], {}, {}
    def unavailable(name, reason):
        return dict(facet=name, status="unavailable", reasons=[reason])
    if quality is None:
        return [unavailable("quote_quality", "quality_not_supplied"),
                unavailable("parity", "quality_not_supplied")], tables, metadata
    q = _report(quality, "quote_quality")
    config = q.metadata["config"]
    if config.get("price_unit") != price_unit:
        raise ValueError("OCA and model price units differ")
    times = _time(q.metadata)
    rows = q.table("quotes")
    if rows.source_row_id.duplicated().any():
        raise ValueError("OCA source row identities must be unique")
    indexed = rows.set_index("source_row_id")
    selected_ids = quotes.quote_id.tolist()
    if not set(selected_ids).issubset(indexed.index):
        raise ValueError("Selected quote IDs are absent from OCA evidence")
    selected = indexed.loc[selected_ids].reset_index()
    pairs = {"strike": "strike", "optiontype": "optiontype", "ttm": "ttm",
             "forward": "forward_price", "discount": "discount",
             "bid_price": "bid_price", "ask_price": "ask_price",
             "bid_iv": "bid_iv", "ask_iv": "ask_iv"}
    for left, right in zip(quotes.to_dict("records"), selected.to_dict("records")):
        for a, b in pairs.items():
            same = left[a] == right[b] if a == "optiontype" else _same_number(left[a], right[b])
            if not same:
                raise ValueError(f"Selected OCA quote differs in {a}: {left['quote_id']}")
    checks = q.table("checks")
    selected_checks = checks[checks.source_row_id.isin(selected_ids)].copy()
    required = {"strike", "forward_price", "discount", "ttm", "optiontype",
                "two_sided_price", "nonnegative_price", "uncrossed_price", "iv",
                "exchange_time", "expiry", "price_bounds", "quote_age"}
    if selected_checks.duplicated(["source_row_id", "check"]).any():
        raise ValueError("Ambiguous OCA check identities")
    failures = sorted(set(selected_checks.loc[selected_checks.status.eq("fail"), "reason"]))
    missing = []
    if times[1] is None:
        missing.append("unchecked:source_time")
    for identity in selected_ids:
        local = selected_checks[selected_checks.source_row_id.eq(identity)].set_index("check")
        for check in sorted(required):
            if check not in local.index or local.loc[check, "status"] not in ("pass", "fail"):
                missing.append(f"unchecked:{check}")
    summary = q.table("summary").iloc[0]
    if summary.snapshot_age_status == "fail":
        failures.append("snapshot_age")
    elif summary.snapshot_age_status != "pass":
        missing.append("unchecked:snapshot_age")
    status = "fail" if failures else ("unavailable" if missing else "pass")
    facets.append(dict(facet="quote_quality", status=status,
                       reasons=sorted(set(failures + missing))))
    tables.update(quality_selected=selected, quality_checks=selected_checks,
                  quality_source_summary=q.table("summary"))
    metadata["oca_quality"] = q.to_dict()
    if parity is None:
        facets.append(unavailable("parity", "parity_not_supplied"))
        return facets, tables, metadata
    p = _report(parity, "parity")
    if p.metadata["config"] != config or _time(p.metadata) != times:
        raise ValueError("OCA parity provenance differs from quote-quality evidence")
    pair_rows = p.table("pairs")
    selected_pairs = pair_rows[pair_rows.source_row_ids.map(
        lambda ids: bool(set(ids).intersection(selected_ids)))].copy()
    reasons = []
    seen = set()
    for pair in selected_pairs.to_dict("records"):
        ids = pair["source_row_ids"]
        if not ids or len(set(ids)) != len(ids) or not set(ids).issubset(indexed.index):
            raise ValueError("Parity pair has unmatched or ambiguous source identities")
        legs = indexed.loc[ids]
        if pair["status"] in ("pass", "fail") and (
                len(ids) != 2 or set(legs.optiontype) != {"C", "P"}):
            raise ValueError("Parity pair must identify one call and one put")
        for name in ("strike", "expiry", "exchange_time", "mat_id", "underlying_index"):
            if legs[name].nunique(dropna=False) != 1 or pair[name] != legs[name].iloc[0]:
                raise ValueError("Parity pair is not matched to the supplied source observations")
        matches = set(ids).intersection(selected_ids)
        if seen.intersection(matches):
            raise ValueError("Selected quotes have ambiguous parity pairs")
        seen.update(matches)
        if pair["status"] != "pass":
            reasons.append(pair["reason"] or f"parity:{pair['status']}")
    if seen != set(selected_ids):
        reasons.append("missing_selected_parity_pairs")
    failed = selected_pairs.status.eq("fail").any()
    status = "fail" if failed else ("unavailable" if reasons else "pass")
    facets.append(dict(facet="parity", status=status, reasons=sorted(set(reasons))))
    tables["parity_selected"] = selected_pairs
    metadata["oca_parity"] = p.to_dict()
    return facets, tables, metadata


def _snapshot_matches(chain, reliability):
    """Ensure sensitivity came from exactly the supplied training observations."""
    try:
        saved = reliability["training"]
        for name in ("ttms", "forwards", "discfactors", "ids", "strikes_ttms",
                     "optiontypes_ttms", "bid_ivs", "ask_ivs", "bid_prices", "ask_prices"):
            current = getattr(chain, name)
            original = saved[name]
            if current is None or original is None:
                if current is not None or original is not None:
                    return False
                continue
            if len(current) != len(original):
                return False
            for a, b in zip(current, original):
                a, b = np.asarray(a), np.asarray(b)
                if a.dtype.kind in "USO" or b.dtype.kind in "USO":
                    if not np.array_equal(a, b):
                        return False
                elif not np.array_equal(a, b, equal_nan=True):
                    return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def combine_research_diagnostics(
    chain, *, model_prices, model_ivols, evaluation_k, price_unit, spread_floor,
    reliability=None, policy=None, noise_assumption=None, quality=None, parity=None,
    quote_ids=None, price_tolerance=1e-10, iv_tolerance=1e-12,
):
    """Observe fit, provenance, support and sensitivity as separate report facets.

    Parameters
    ----------
    chain, model_prices, model_ivols, price_unit, spread_floor, quote_ids
        Training observations and final model outputs passed to ``diagnose_fit``.
        With OCA sidecars, quote IDs must be the original selected source-row IDs.
    evaluation_k : sequence of arrays
        One finite log(K/F) vector per expiry. These are evaluation locations,
        not additional observations or a declaration of model-domain validity.
    reliability, policy, noise_assumption : optional
        Existing sensitivity report, caller-frozen research artifact and explicit
        noise declaration. Missing/failed or mismatched evidence stays unavailable.
    quality, parity : optional
        Serialized OCA reports or objects with ``to_dict``. No OCA import is needed.
    price_tolerance, iv_tolerance : float
        Nonnegative roundoff allowances for observed-interval membership. Any
        checked model output outside an interval flags the residual facet.

    Returns
    -------
    DiagnosticReport
        Separate ``facets``, fit ``quotes``/``expiries``/``summary``, pointwise
        ``support`` and joined OCA tables. There is no overall accuracy verdict.
    """
    if any(not np.isfinite(v) or v < 0 for v in (price_tolerance, iv_tolerance)):
        raise ValueError("Interval tolerances must be finite and nonnegative")
    if len(evaluation_k) != len(chain.ttms):
        raise ValueError("Evaluation locations must align with expiries")
    fit = diagnose_fit(chain, model_prices=model_prices, model_ivols=model_ivols,
                       price_unit=price_unit, spread_floor=spread_floor, quote_ids=quote_ids)
    quotes = fit.table("quotes")
    n_valid = {}
    for kind, tolerance in (("price", price_tolerance), ("iv", iv_tolerance)):
        valid = quotes[f"{kind}_reason"].eq("valid")
        outside = valid & ((quotes[f"model_{kind}"] < quotes[f"bid_{kind}"] - tolerance)
                           | (quotes[f"model_{kind}"] > quotes[f"ask_{kind}"] + tolerance))
        quotes[f"{kind}_outside_interval"] = outside.where(valid, None)
        n_valid[kind] = int(valid.sum())
    complete = all(count == len(quotes) for count in n_valid.values())
    outside_count = int(quotes.price_outside_interval.eq(True).sum()
                        + quotes.iv_outside_interval.eq(True).sum())
    residual = dict(facet="observed_fit", status=("flag" if outside_count else "clear")
                    if complete else "unavailable", reasons=[])
    if outside_count:
        residual["reasons"].append("model_outside_observed_quote_intervals")
    if not complete:
        residual["reasons"].append("incomplete_price_or_iv_residual_evidence")
    facets, tables, provenance = _quality(chain, quotes, quality, parity, price_unit)
    facets.append(residual)
    support = []
    for i, points in enumerate(evaluation_k):
        points = np.asarray(points, dtype=float)
        strikes = np.asarray(chain.strikes_ttms[i], dtype=float)
        forward = float(chain.forwards[i])
        if (points.ndim != 1 or not len(points) or not np.all(np.isfinite(points))
                or not np.isfinite(forward) or forward <= 0
                or not len(strikes) or not np.all(np.isfinite(strikes) & (strikes > 0))):
            raise ValueError("Support requires finite log strikes and positive observed strikes/F")
        observed = np.log(strikes / forward)
        for k in points:
            region = ("left_extrapolation" if k < observed.min() - 1e-12 else
                      "right_extrapolation" if k > observed.max() + 1e-12 else
                      "inside_observed_range")
            support.append(dict(expiry_index=i, expiry_id=str(chain.ids[i]), k=float(k),
                                observed_k_min=float(observed.min()),
                                observed_k_max=float(observed.max()), region=region,
                                accuracy_status="unvalidated_extrapolation" if
                                region != "inside_observed_range" else "not_assessed"))
    support = pd.DataFrame(support)
    extrapolated = support.region.ne("inside_observed_range").any()
    facets.append(dict(facet="strike_support", status="extrapolated" if extrapolated
                       else "within_observed_range", reasons=["wing_accuracy_unvalidated"]
                       if extrapolated else ["accuracy_not_assessed"]))
    blockers = [f"{f['facet']}:{reason}" for f in facets[:2] if f["status"] != "pass"
                for reason in (f["reasons"] or [f["status"]])]
    if reliability is None or policy is None:
        sensitivity = dict(status="unavailable", score=None, cutoff=None,
                           reasons=["sensitivity_report_or_policy_not_supplied"])
    elif not _snapshot_matches(chain, reliability):
        sensitivity = dict(status="unavailable", score=None, cutoff=None,
                           reasons=["sensitivity_training_snapshot_mismatch"])
    else:
        try:
            _, expected_prices, expected_ivs = saved_model_outputs(reliability)
            model_matches = all(np.allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True)
                                for actual, expected in ((model_prices, expected_prices),
                                                         (model_ivols, expected_ivs))
                                for a, b in zip(actual, expected))
            grid_matches = (len(reliability["slices"]) == len(evaluation_k) and
                            all(np.array_equal(a, b["grid"]["k"])
                                for a, b in zip(evaluation_k, reliability["slices"])))
            if not model_matches:
                blockers.append("sensitivity_base_model_mismatch")
            if not grid_matches:
                blockers.append("sensitivity_evaluation_grid_mismatch")
            from _svi_benchmark import RawSvi, raw_svi
            for i, t in enumerate(chain.ttms):
                saved_grid = reliability["slices"][i]["grid"]
                params = RawSvi(**reliability["base"]["parameters"][i])
                expected_base = np.sqrt(raw_svi(np.asarray(saved_grid["k"]), params)[0] / t)
                if not np.allclose(saved_grid["base_iv"], expected_base,
                                   rtol=1e-10, atol=1e-12):
                    blockers.append("sensitivity_base_grid_mismatch")
                    break
            sensitivity = assess_research_policy(reliability, policy,
                                                 noise_assumption=noise_assumption,
                                                 quality_reasons=blockers)
        except (KeyError, TypeError, ValueError, IndexError):
            sensitivity = dict(status="unavailable", score=None, cutoff=None,
                               reasons=["malformed_or_missing_sensitivity_base"])
    facets.append(dict(facet="wing_sensitivity", status=sensitivity["status"],
                       reasons=sensitivity["reasons"] + sensitivity.get("quality_reasons", [])))
    summary = fit.table("summary")
    for kind in ("price", "iv"):
        count = int(quotes[f"{kind}_outside_interval"].eq(True).sum())
        summary[f"{kind}_interval_checked"] = n_valid[kind]
        summary[f"{kind}_interval_excluded"] = len(quotes) - n_valid[kind]
        summary[f"{kind}_interval_outside"] = count
        summary[f"{kind}_interval_inside_fraction"] = (
            (n_valid[kind] - count) / n_valid[kind] if n_valid[kind] else None)
    tables.update(facets=pd.DataFrame(facets), quotes=quotes, expiries=fit.table("expiries"),
                  summary=summary, support=support)
    return DiagnosticReport.from_tables(tables, dict(
        kind="combined_research", price_unit=price_unit, iv_unit="decimal", ttm_unit="years",
        price_tolerance=price_tolerance, iv_tolerance=iv_tolerance, spread_floor=spread_floor,
        residual_rule="any checked model output outside its observed interval",
        sensitivity=sensitivity, automatic_warning=False, accuracy_certificate=False,
        overall_accuracy_status=None, quote_quality_scope="selected rows and snapshot age",
        parity_scope="source pairs containing selected quote IDs", **provenance))


def saved_model_outputs(reliability):
    """Re-evaluate saved independent raw-SVI base parameters; never refit."""
    from _svi_benchmark import RawSvi, raw_svi
    from compare_smile_fitters import black_prices
    saved = reliability["training"]
    chain = OptionChain(**{k: np.asarray(v) if k in ("ttms", "forwards", "discfactors", "ids")
                          else List([np.asarray(row) for row in v]) if v is not None else None
                          for k, v in saved.items()})
    if reliability["metadata"]["model"] != "raw_svi_constrained":
        raise ValueError("Saved replay currently supports independent constrained raw SVI")
    ivs, prices = [], []
    for i, t in enumerate(chain.ttms):
        k = np.log(chain.strikes_ttms[i] / chain.forwards[i])
        p = RawSvi(**reliability["base"]["parameters"][i])
        iv = np.sqrt(raw_svi(k, p)[0] / t)
        ivs.append(iv)
        prices.append(black_prices(chain.forwards[i], chain.strikes_ttms[i], t, iv,
                                   chain.discfactors[i], chain.optiontypes_ttms[i]))
    return chain, prices, ivs


class Locals(Enum):
    """Core-only synthetic report and explicitly supplied saved-report replay."""

    SYNTHETIC = 1
    SAVED = 2


def run_local(local: Locals = Locals.SYNTHETIC):
    """Write a combined JSON report without calibrating or requesting provider data."""
    from compare_smile_fitters import _quote_chain, black_prices
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-reliability", type=Path)
    parser.add_argument("--research-policy", type=Path)
    parser.add_argument("--noise-assumption", choices=("known_iid_gaussian_iv",))
    parser.add_argument("--oca-sidecars", type=Path)
    parser.add_argument("--case-label")
    parser.add_argument("--price-unit",
                        help="Units of the saved prices; no conversion is performed")
    parser.add_argument("--spread-floor", type=float, help="Positive floor in declared price units")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.saved_reliability and (args.price_unit is None or args.spread_floor is None):
        parser.error("Saved replay requires --price-unit and --spread-floor")
    price_unit = args.price_unit if args.price_unit is not None else "USD per unit"
    spread_floor = args.spread_floor if args.spread_floor is not None else .01
    policy, sidecar, reliability = None, {}, None
    provenance = {}
    if args.research_policy:
        policy, digest = load_policy(args.research_policy)
        provenance["policy_sha256"] = digest
    if args.saved_reliability:
        local = Locals.SAVED
        raw = args.saved_reliability.read_bytes()
        data = json.loads(gzip.decompress(raw) if args.saved_reliability.suffix == ".gz" else raw)
        reliability = data.get("report", data.get("result", data))
        chain, prices, ivs = saved_model_outputs(reliability)
        points = [s["grid"]["k"] for s in reliability["slices"]]
        provenance["saved_reliability_sha256"] = sha256(raw).hexdigest()
    elif local == Locals.SYNTHETIC:
        chain = _quote_chain(np.array([.1]), [np.linspace(-.2, .2, 7)], [np.full(7, .25)])
        ivs = [np.full(7, .252)]
        prices = [black_prices(chain.forwards[0], chain.strikes_ttms[0], .1, ivs[0],
                               chain.discfactors[0], chain.optiontypes_ttms[0])]
        points = [np.linspace(-.4, .4, 17)]
    else:
        parser.error("Saved replay requires --saved-reliability")
    if args.oca_sidecars:
        raw = args.oca_sidecars.read_bytes()
        sidecar = json.loads(raw)
        if "cases" in sidecar:
            if args.case_label is None:
                parser.error("--case-label is required for a multi-case OCA sidecar")
            sidecar = sidecar["cases"][args.case_label]
        provenance["oca_sidecars_sha256"] = sha256(raw).hexdigest()
    ids = [[r["source_row_id"] for r in sidecar["selected"]]] if "selected" in sidecar else None
    report = combine_research_diagnostics(chain, model_prices=prices, model_ivols=ivs,
             evaluation_k=points, price_unit=price_unit, spread_floor=spread_floor,
             reliability=reliability, policy=policy, noise_assumption=args.noise_assumption,
             quality=sidecar.get("quality"), parity=sidecar.get("parity"), quote_ids=ids)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path(local_path.get_output_path()) / "combined_research" / stamp
    output.mkdir(parents=True, exist_ok=False)
    payload = dict(report=report.to_dict(), input_provenance=provenance,
                   consumer_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (output / "report.json").write_text(json.dumps(payload, indent=2, allow_nan=False),
                                       encoding="utf-8")
    print(report.table("facets").to_string(index=False), output / "report.json")


if __name__ == "__main__":
    run_local()
