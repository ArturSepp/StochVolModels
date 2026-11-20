"""Pair-aware conditional second moments, price comparisons and qualification decisions."""

import numpy as np

from validation.pde_projection_study import GRIDS, SEEDS, STEPS


def pair(values):
    """Keep seed and quote axes; the engine stores each partner half a batch away."""
    half = values.shape[1]//2
    if half < 2 or values.shape[1] != 2*half:
        raise ValueError("expected complete antithetic pairs within each seed")
    return .5*(values[:, :half] + values[:, half:])


def standard_error(values):
    """Pool independent pairs from the archived equal-size seeds."""
    pairs = values.reshape((-1, *values.shape[2:]))
    return pairs.std(axis=0, ddof=1)/np.sqrt(pairs.shape[0])


def bin_estimate(x_bin, sigma2, index):
    hit = x_bin == index
    den = pair(hit.astype(float))
    num = pair(np.where(hit, sigma2, 0.))
    p, n = den.mean(), num.mean()
    est = n/p if p else np.nan
    influence = (num-est*den)/p if p else np.full_like(den, np.nan)
    se = standard_error(influence) if np.count_nonzero(den) > 1 else np.nan
    seed_p = den.mean(axis=1)
    seed_n = num.mean(axis=1)
    seed_est = np.divide(seed_n, seed_p, out=np.full_like(seed_n, np.nan), where=seed_p > 0)
    return dict(prob=p, numerator=n, conditional=est, se=se, share=n/sigma2.mean(),
                paths_hit=hit.sum(), pairs_hit=np.count_nonzero(den), seed_estimates=seed_est,
                seed_counts=hit.sum(axis=1), seed_sd=seed_est.std(ddof=1),
                seed_prob=seed_p, seed_numerator=seed_n), influence


def load_states(raw_dir, design):
    states = {}
    for steps in STEPS:
        xs, sigmas = [], []
        for seed in SEEDS:
            with np.load(raw_dir/f"mc_{steps}_{seed}.npz", allow_pickle=False) as f:
                np.testing.assert_array_equal(f["times"], design["times"])
                x, sigma = f["x"], f["sigma"]
                if (f["steps"].item() != steps or f["seed"].item() != seed
                        or x.shape != (len(design["times"]), design["paths_per_seed"])
                        or sigma.shape != x.shape or not np.isfinite(x).all()
                        or not np.isfinite(sigma).all() or not (sigma > 0).all()):
                    raise ValueError("invalid saved price-volatility samples")
                xs.append(x)
                sigmas.append(sigma)
        states[steps] = (np.stack(xs), np.square(sigmas))
    return states


def price_checks(case, pde, states, archive, limits):
    forward, discount = case.chain.forwards[0], case.chain.discfactors[0]
    strikes = case.chain.strikes_ttms[0]
    unit = forward*discount/10000
    quotes = {name: np.asarray(pde[name]["prices"][0]) for name in GRIDS}
    results, previous = {}, {}
    for steps in STEPS:
        x = states[steps][0][:, -1, :]
        asset = forward*np.exp(x)
        raw = discount*np.maximum(asset[:, :, None]-strikes, 0.)
        estimators = dict(raw=raw, controlled=raw-discount*(asset-forward)[:, :, None])
        for key, individual in estimators.items():
            values = pair(individual)
            mean = values.mean(axis=(0, 1))
            se = standard_error(values)
            row = dict(mean=mean, se=se, allowance_bp=limits["agreement_se"]*se/unit
                       + limits["price_bp"])
            for name, prices in quotes.items():
                gap = (prices-mean)/unit
                row[name] = dict(gap_bp=gap, passed=np.abs(gap) <= row["allowance_bp"])
            if steps == STEPS[0]:
                for j, seed in enumerate(SEEDS):
                    archived = archive["batches"][j]
                    if archived["config"]["seed"] != seed:
                        raise ValueError("archived seed order changed")
                    np.testing.assert_allclose(values[j].mean(axis=0), archived[key]["mean"],
                                               rtol=0., atol=5e-14)
                    np.testing.assert_allclose(standard_error(values[j:j+1]),
                                               archived[key]["standard_error"], rtol=3e-13)
                np.testing.assert_allclose(mean, archive[key]["mean"], rtol=0., atol=5e-14)
                np.testing.assert_allclose(se, archive[key]["standard_error"], rtol=3e-13)
                previous[key] = values
            else:
                change = values-previous[key]
                row.update(mc_change_bp=change.mean(axis=(0, 1))/unit,
                           mc_change_se_bp=standard_error(change)/unit)
            results[f"{steps}_{key}"] = row
    grid_bp = (quotes[GRIDS[0]]-quotes[GRIDS[1]])/unit
    passed = all(np.all(row[k]["passed"]) for row in results.values() for k in GRIDS)
    return dict(strikes=strikes, pde=quotes, mc=results, grid_bp=grid_bp,
                passed=passed and np.all(np.abs(grid_bp) <= limits["price_bp"]))


def assess(case, design, pde, raw_dir, archive):
    limits, cuts = design["limits"], design["log_cuts"]
    states = load_states(raw_dir, design)
    prices = price_checks(case, pde, states, archive, limits)
    rows, outside = [], []
    for j, t in enumerate(design["times"]):
        estimates = {}
        for steps, (x, sigma2) in states.items():
            ids = np.searchsorted(cuts, x[:, j, :], side="right")
            estimates[steps] = [bin_estimate(ids, sigma2[:, j, :], i) for i in range(len(cuts)+1)]
            obs = [record for record, _ in estimates[steps]]
            np.testing.assert_allclose(sum(r["prob"] for r in obs), 1., atol=3e-15, rtol=0.)
            np.testing.assert_allclose(sum(r["numerator"] for r in obs), sigma2[:, j, :].mean(),
                                       rtol=3e-14)
        current = []
        for i in range(len(cuts)+1):
            grids = {k: {q: pde[k]["checkpoints"][j][q][i] for q in
                         ("prob", "numerator", "conditional", "share")} for k in GRIDS}
            mc = {str(k): estimates[k][i][0] for k in STEPS}
            coarse, fine = [estimates[k][i][0] for k in STEPS]
            fine_ratio = fine["conditional"]
            change = fine_ratio-coarse["conditional"]
            delta_se = standard_error(estimates[STEPS[1]][i][1] - estimates[STEPS[0]][i][1])
            grid_change = grids[GRIDS[0]]["conditional"]/grids[GRIDS[1]]["conditional"]-1.
            checks = dict(grid=abs(grid_change) <= limits["grid"],
                          mc_step=abs(change) <= limits["mc_step"]*fine_ratio)
            for k, r in mc.items():
                checks[f"precision_{k}"] = r["se"] <= limits["mc_relative_se"]*r["conditional"]
                for name, grid in grids.items():
                    checks[f"agreement_{name}_{k}"] = (
                        abs(grid["conditional"]-r["conditional"]) <= limits["agreement_se"]*r["se"]
                        + limits["agreement_relative"]*r["conditional"])
            row = dict(time=t, bin=i, pde=grids, mc=mc, grid_change=grid_change,
                       mc_change=change, mc_change_se=delta_se, checks=checks,
                       qualified=all(checks.values()))
            current.append(row)
        rows += current
        missed = [r for r in current if not r["qualified"]]
        shares = {}
        for key in (*GRIDS, *map(str, STEPS)):
            method = "pde" if key in GRIDS else "mc"
            shares[key] = {q: sum(r[method][key][q] for r in missed) for q in ("prob", "share")}
        outside.append(dict(time=t, qualified_bins=[r["bin"] for r in current if r["qualified"]],
                            outside=shares))
    verified = prices["passed"] and all(all(pde[k]["density_checks"].values()) for k in GRIDS)
    return dict(rows=rows, outside=outside, prices=prices, verified=verified,
                fully_qualified=verified and all(r["qualified"] for r in rows))


def render(result, design, pde):
    rows = result["rows"]
    qualified = sum(r["qualified"] for r in rows)
    strikes = design["strike_forward"]
    bins = ([f"<{strikes[0]:g}"] + [f"[{a:g}, {b:g})" for a, b in zip(strikes, strikes[1:])]
            + [f"≥{strikes[-1]:g}"])
    precision_misses = sum(any(not r["checks"][f"precision_{k}"] for k in STEPS) for r in rows)
    step_misses = sum(not r["checks"]["mc_step"] for r in rows)
    max_grid = max(abs(r["grid_change"]) for r in rows)
    lines = ["# Conditional volatility second moments on the frozen two-year case", "",
             f"**{qualified} of {len(rows)} bin–expiry comparisons qualified.** "
             f"The five vanilla strikes {'passed' if result['prices']['passed'] else 'failed'} "
             "the price checks. Bin labels show the asset divided by its forward. The calculation "
             "uses the logarithms of the five frozen strike ratios as edges, including both tails.",
             "", f"The largest change under PDE refinement was {100*max_grid:.3f}%. "
             f"{precision_misses} bin–expiry estimates exceeded the 1% Monte Carlo standard-error "
             f"limit and {step_misses} exceeded the 2% observed MC time-step limit.",
             "", "The qualified returning grid has 607 × 309 nodes and 512 steps. The refined grid "
             "has 727 × 446 nodes and 1,024 steps. Each row measures the conditional mean of "
             "annualized volatility squared. The probability and second-moment numerator are "
             "integrated separately, with a proportional split of any x cell crossing an edge. "
             "Each checkpoint reproduces the package projection and the Stage 3 joint second "
             "moment. All Monte Carlo observations use the same frozen model and four archived "
             "seeds, with 160,000 independent antithetic pairs per time resolution.", "",
             "## Conditional estimates", "",
             "| Years | S/F | Returning | Refined | MC 2880 | MC 5760 | Grid Δ % | "
             "MC step Δ % ± 1 SE | Fine MC SE % | Decision |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lo, hi = (r["mc"][str(k)] for k in STEPS)
        a, b = (r["pde"][k]["conditional"] for k in GRIDS)
        m = hi["conditional"]
        flags = r["checks"]
        bad = []
        if not flags["grid"]:
            bad.append("grid")
        if not flags["mc_step"]:
            bad.append("MC step")
        bad.extend(f"SE {k}" for k in STEPS if not flags[f"precision_{k}"])
        if any(not v for k, v in flags.items() if k.startswith("agreement")):
            bad.append("PDE/MC gap")
        lines.append(f"| {r['time']:g} | {bins[r['bin']]} | {a:.6g} | {b:.6g} | "
                     f"{lo['conditional']:.6g} | {m:.6g} | {100*r['grid_change']:+.3f} | "
                     f"{100*r['mc_change']/m:+.3f} ± {100*r['mc_change_se']/m:.3f} | "
                     f"{100*hi['se']/m:.3f} | {', '.join(bad) or 'qualified'} |")
    lines += ["", "Grid change uses the refined PDE as denominator. MC step change is fine minus "
              "coarse relative to fine. Its standard error uses same-seed, same-pair ratio "
              "influences across resolutions. The two grids of Brownian time increments share "
              "a random stream but are not explicitly coupled by summing fine increments. "
              "Each bin requires a grid change and an observed MC step change of at most 2%, "
              "relative MC standard errors of at most 1% at both resolutions, and agreement "
              "of both PDE estimates with both MC estimates within 4 SE plus 2%.", "",
              "## Probability and contribution to the unconditional moment", "",
              "| Years | S/F | Refined P | Refined numerator | Refined share % | Fine MC P | "
              "Fine MC numerator | Fine MC share % | Fine MC paths / pairs hit | "
              "Fine between-seed SD % |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        g, m = r["pde"][GRIDS[1]], r["mc"][str(STEPS[1])]
        lines.append(f"| {r['time']:g} | {bins[r['bin']]} | {g['prob']:.6g} | "
                     f"{g['numerator']:.6g} | {100*g['share']:.4f} | {m['prob']:.6g} | "
                     f"{m['numerator']:.6g} | {100*m['share']:.4f} | "
                     f"{m['paths_hit']} / {m['pairs_hit']} | "
                     f"{100*m['seed_sd']/m['conditional']:.3f} |")
    lines += ["", "The numerator is the bin's unconditional contribution to the second moment. "
              "The share divides it by the total second moment at that date. Between-seed SD "
              "uses the four individual ratio estimates. The numerical record includes these "
              "quantities on both grids and both simulation resolutions.", "",
              "## Outside the qualified bins", "",
              "| Years | Qualified S/F bins | Returning P / moment % | Refined P / moment % | "
              "MC 2880 P / moment % | MC 5760 P / moment % |",
              "|---:|---|---:|---:|---:|---:|"]
    for line in result["outside"]:
        shares = [line["outside"][k] for k in (*GRIDS, *map(str, STEPS))]
        valid = ", ".join(bins[i] for i in line["qualified_bins"]) or "none"
        fields = " | ".join(f"{100*s['prob']:.3f} / {100*s['share']:.3f}" for s in shares)
        lines.append(f"| {line['time']:g} | {valid} | {fields} |")
    prices = result["prices"]
    lines += ["", "## Two-year vanilla calls", "",
              "Prices are in units of the frozen unit forward. MC columns give price ± one "
              "pair-aware standard error. Both PDE grids were checked against all four estimates "
              "at each strike. The discounted-forward price tolerance is 4 SE plus 1 bp. "
              "The grid-refinement tolerance is 1 bp.", "",
              "| K/F | Returning | Refined | Grid Δ bp | 2880 raw | 2880 controlled | "
              "5760 raw | 5760 controlled | Worst allowance used % |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for i, k in enumerate(strikes):
        a, b = (prices["pde"][n][i] for n in GRIDS)
        obs = list(prices["mc"].values())
        ratios = [abs(v[n]["gap_bp"][i])/v["allowance_bp"][i] for v in obs for n in GRIDS]
        estimates = " | ".join(f"{v['mean'][i]:.6f} ± {v['se'][i]:.6f}" for v in obs)
        lines.append(f"| {k:g} | {a:.7f} | {b:.7f} | {prices['grid_bp'][i]:+.5f} | "
                     f"{estimates} | {100*max(ratios):.2f} |")
    checkpoints = [row for grid in pde.values() for row in grid["checkpoints"]]
    closure = max(abs(sum(r["numerator"])-r["direct_second"]) for r in checkpoints)
    gap = max(np.max(np.abs(np.subtract(r["package_numerator"], r["numerator"])))
              for r in checkpoints)
    lines += ["", "The 2,880-step replay reproduced the archived raw and controlled prices and "
              "standard errors for every seed and the pooled sample. PDE terminal prices "
              "reproduced the Stage 3 records on both grids. The numerical record contains "
              "per-strike signed gaps and changes under MC time refinement.", "",
              "The largest absolute difference between the summed bin numerators and the direct "
              f"joint second moment was {closure:.3g}. The largest numerator discrepancy with "
              f"the package projection was {gap:.3g}. "
              f"The returning and refined solves took {pde[GRIDS[0]]['elapsed_seconds']:.1f} s "
              f"and {pde[GRIDS[1]]['elapsed_seconds']:.1f} s. The eight Monte Carlo batches "
              f"took {sum(v['seconds'] for v in result['mc_receipts']):.1f} s in total.", "",
              "## Reproduction", "",
              "```powershell",
              "$py = 'C:/Python/StochVolModels312/Scripts/python.exe'",
              "$env:OPENBLAS_NUM_THREADS = '1'", "$env:OMP_NUM_THREADS = '1'",
              "$env:MKL_NUM_THREADS = '1'", "$env:MPLBACKEND = 'Agg'",
              "& $py -B -m validation.pde_volatility_moments --stage projection `",
              "  --frozen-report agents/pde_qualification/20261117_full/qualification.json",
              "```", "",
              "The runner saves joint observations for each seed and time grid. A matching "
              "output directory can be resumed with `--output`. It returns a nonzero status "
              "when any bin or vanilla safeguard misses its target. For bins limited by Monte "
              "Carlo precision, increase independent pairs with the same fixed bin edges. "
              "Time-step changes with large uncertainty warrant Brownian-coupled refinement.",
              "", "[Study design](projection_design.json) · [Numerical record](projection.json) "
              "· [Joint observations](observations/)", ""]
    return "\n".join(lines)
