"""Assemble the requested positive-skew/DVOL handoff from saved outputs.

Author: Artur Sepp's review replication, implemented by Codex, 29 October 2026.
No manuscript, numerical source or frozen bundle is modified.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import pandas as pd

BASE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(frame, digits=5):
    # Do not introduce a tabulate dependency for Markdown rendering.
    header = '| '+' | '.join(map(str,frame.columns))+' |'
    lines = [header, '| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False, name=None):
        cells = [f'{v:.{digits}f}' if isinstance(v,float) else str(v) for v in row]
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)


def main(commit):
    notes = BASE/'notes'
    notes.mkdir(exist_ok=True)
    provenance = read(BASE/'runtime_provenance.json')
    provenance['git_commit'] = commit
    provenance['git_error'] = None
    provenance['git_commit_read_method'] = 'Read-only git rev-parse HEAD outside the restricted sandbox.'
    provenance['final_script_sha256'] = {p.name:sha(p) for p in BASE.glob('*.py')}
    provenance['report_at_utc'] = datetime.now(timezone.utc).isoformat()
    (BASE/'runtime_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    selected = read(BASE/'selected_surfaces.json')['cases'][0]
    scan = read(BASE/'scan_summary.json')
    stem = selected['ticker'].lower()
    fit = read(BASE/f'{stem}_pde_fit.json')
    checks = read(BASE/f'{stem}_pde_checks.json')
    mc = read(BASE/f'{stem}_pde_mc.json')
    qmc = read(BASE/f'{stem}_logsv_fit.json')
    hfit = read(BASE/f'{stem}_heston_fit.json')
    level = mc['levels'][-1]
    p = fit['parameters']
    heston = checks['Heston']
    dvol = read(BASE.parent/'dvol_20261031/manifest.json')
    margin = p['kappa2']-p['beta']
    active_margin = 'margin' in fit['active_bounds']
    core = dict(ticker=selected['ticker'], date=selected['date'], quote_count=fit['quote_count'],
        logsv_beta=p['beta'], logsv_kappa2=p['kappa2'], stock_martingale_margin=margin,
        fitted_m=fit['x'][3], m_lower_bound=.05, m_lower_bound_active=active_margin,
        heston_rho=heston['parameters']['rho'], heston_iv_rmse_pp=heston['rmse_pp'],
        logsv_pde_iv_rmse_pp=fit['final']['rmse_pp'],
        heston_inside_spread_percent=heston['within_bidask_percent'],
        logsv_inside_spread_percent=fit['final']['within_bidask_percent'])
    (BASE/'claim_numbers.json').write_text(json.dumps(core, indent=2), encoding='utf-8')
    gate_rows = [dict(refinement=name, max_iv_change_pp=c['max_IV_change_pp'],
        max_price_change_bpF=c['max_price_change_bpF'],
        max_fraction_premium_spread=c['max_price_change_fraction_bidask'],
        passed=c['max_IV_change_pp'] < .005 and c['max_price_change_fraction_bidask'] < .02)
        for name,c in checks['checks'].items()]
    checks_ok = (all(row['passed'] for row in gate_rows) and mc['status']=='complete' and
        not level['warnings'] and heston['max_price_gap_bpF']<1e-6 and
        all(r['success'] for r in fit['runs']+qmc['runs']+hfit['runs']) and
        fit['quote_sha256']==sha(BASE/selected['quote_file']))
    audit = read(BASE/'audit/protected_file_hashes.json')
    changed = [name for name,digest in audit.items() if not Path(name).exists() or sha(Path(name))!=digest]
    if changed:
        raise RuntimeError(f'Protected inputs changed: {changed}')
    acceptance = dict(all_acceptance_checks_pass=checks_ok, deterministic=gate_rows,
        heston_independent_max_gap_bpF=heston['max_price_gap_bpF'],
        mc_status=mc['status'], mc_warnings=level['warnings'],
        all_optimizer_stages_successful=all(r['success'] for r in fit['runs']+qmc['runs']+hfit['runs']),
        quote_hash_verified=True, unchanged_protected_files=len(audit))
    (BASE/'acceptance.json').write_text(json.dumps(acceptance, indent=2), encoding='utf-8')
    common = (f'Commit `{commit}`; OCA {provenance["versions"]["option-chain-analytics"]}; '
              f'StochVolModels {provenance["versions"]["stochvolmodels"]}; exact source hashes in '
              '`runtime_provenance.json` and `audit/scripts.diff`.')
    attribution = ('*Paper authors: [Artur Sepp](https://github.com/ArturSepp) and Parviz Rakhmonov. '
        'Execution and report by Codex, 29 October 2026. First recorded in Git: pending.*\n\n'
        '[StochVolModels](https://github.com/ArturSepp/StochVolModels) · '
        '[CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).')
    text = ['# Positive-skew surface and DVOL: results', attribution,
        '## Contradictions, qualifications and status',
        ('C1 and C2 sign predictions are supported.' if p['beta']>0 and heston['parameters']['rho']>0 else
         '**C1/C2 sign prediction contradicted: see the fitted coefficients below.**')+
        f' The selected surface is TLT, not GLD. The martingale-margin lower bound is {"active" if active_margin else "not active"}. '
        'A constrained fit cannot empirically prove the martingality theorem. '
        'This example illustrates positive-beta calibration but does not show pressure near the martingale boundary.',
        'GLD has no positive 30-day skew date in its 22-date July 2026 cache. The pilot name “400d” refers to '
        'the maximum option maturity in its manifest, not 400 observation dates. TLT supplies the positive-skew example. '
        'The reported skew shares concern unequal available cache periods and cannot be equated to the volatility-index beta shares.',
        '**DVOL timestamp correction to the brief:** a native midnight timestamp labels the start of the daily candle. '
        'Its close is at the end of that UTC day. Same-date midnight spot would therefore be one day earlier. '
        'Hourly aggregates verify this for both assets. The CSV preserves the native date; align close timestamps explicitly downstream.',
        f'Numerical acceptance: **{checks_ok}**. No manuscript or 20261029/20261030 generator was edited. '
        'The brief expressly reserves manuscript integration and DVOL descriptive analysis for a later step. '
        'The requested 20261031 folder labels are retained; actual execution is dated in the provenance files.',
        f'Monte Carlo qualification: {level["outside_2se"]} of {fit["quote_count"]} PDE prices lie outside two '
        f'estimated scramble standard errors; the largest standardized gap is {level["max_abs_z"]:.3f}. '
        f'The largest absolute price gap is {level["max_gap_bpF"]:.6f} basis points of forward. '
        'The prescribed coupled-time criterion passes and no flooring is reported. '
        'These four-scramble diagnostics do not establish simultaneous confidence coverage.',
        '## A1. Scan and preregistered selection',
        table(pd.DataFrame(scan['coverage'])),
        'The denominator is the number of dates with a finite OCA 25-delta skew on the eligible slice nearest 30 days. '
        'All 827 snapshots were processed without errors. Four expiries are chosen with the original rule before ranking. '
        'No GLD date passes either the four-positive or relaxed two-shortest-positive rule. TLT passes the strict four-positive rule. '
        'The selected date maximises the eligible nearest-30-day skew within that rule. No relaxation was used for TLT.',
        'Five highest-ranked candidate dates per ticker (GLD rows are ineligible and retained to make the fallback auditable):',
        table(pd.read_csv(BASE/'selection_candidates.csv')),
        '![Skew scan](../skew_scan.png)',
        f'Provenance: `select_positive_skew_surface.py --scan`; {common} No random seed. '
        'All options/spot partition and cache-manifest hashes are in `scan_summary.json`. '
        'The plot-only follow-up changes date-axis formatting, not scan values.',
        '## A2. Selected quotes', table(pd.DataFrame(selected['slices'])),
        f'Quote file `{selected["quote_file"]}`, SHA-256 `{selected["quote_sha256"]}`. '
        'The four expiries contain 99 retained quotes. The unchanged expiry rule selects approximately '
        '12, 28, 49 and 59 days from the available TLT cache (whose maximum DTE is 60), not a manufactured 90-day slice.',
        'The new selector reproduced both frozen 15 July GLD/TLT input tables exactly, including every CSV value and column. '
        'OCA skew uses its own 25-delta contract selection on the source slice; it is not recomputed from the downsampled strike grid.',
        f'Provenance: `select_positive_skew_surface.py --ticker TLT --date 2024-08-02`; {common} '
        'No random seed. Cache/partition hashes and OCA metadata are in `selected_surfaces.json`.',
        '## A3. Calibration and numerical checks',
        table(pd.read_csv(BASE/'comparison_table.csv')),
        'The comparison CSV preserves the earlier schema: one row per model, hence two rows for one new surface. '
        'A literal single row would change that schema and was not used.',
        table(pd.DataFrame([core])),
        'LogSV parameters: '+', '.join(f'`{k}={v:.8g}`' for k,v in p.items())+'. '
        'Active bounds: '+', '.join(fit['active_bounds'])+'.',
        'Heston parameters: '+', '.join(f'`{k}={v:.8g}`' for k,v in heston['parameters'].items())+'.',
        table(pd.read_csv(BASE/'maturity_comparison.csv')),
        table(pd.DataFrame(gate_rows)),
        f'Heston independent repricing differs from the refined canonical transform by '
        f'{heston["max_price_gap_bpF"]:.9g} basis points of forward (required < 1e-6).',
        table(pd.DataFrame([dict(steps=l['steps'], coarse_steps=l['coarse_steps'],
            max_pde_minus_mc_bpF=l['max_gap_bpF'], max_abs_z=l['max_abs_z'],
            outside_two_SE=l['outside_2se'], time_resolved=l['time_resolved'], warnings=len(l['warnings']))
            for l in mc['levels']])),
        'Monte Carlo uses four fresh scrambles with 32,768 paths each, coupled time grids and no forward recentering. '
        'No volatility-floor warning was reported. Four-scramble standard errors are noisy diagnostics, not simultaneous confidence tests.',
        table(pd.DataFrame([dict(model=model, stage=r['stage'], success=r['success'], nfev=r['nfev'],
            seconds=r['seconds'], rmse_pp=r['rmse_pp']) for model, record in
            [('Heston',hfit),('LogSV start',qmc),('LogSV PDE',fit)] for r in record['runs']])),
        'Fit times exclude compilation/warm-up and final validation; the PDE time excludes the simulation warm-start generation, '
        'whose separate timing is listed above. This matches the prior reporting convention.',
        '![Surface fits](../tlt_fits.png)',
        f'Provenance: `run_calibrations.py`, `run_pde.py --mode fit/check/mc`, `make_results.py`; {common} '
        f'Quote SHA-256 `{fit["quote_sha256"]}`; simulation-fit seed 291026; validation scrambles '
        '291201, 291202, 291203, 291204. Same bounds, objective, grids and gates as the original bundle. '
        'The copied transform adapter is used only for Heston; its legacy LogSV expansion diagnostic is not used.',
        '## B. Deribit DVOL coverage',
        table(pd.DataFrame([dict(currency=c, rows=r['row_count'], first=r['first_date'], last=r['last_date'],
            gaps=len(r['gaps']), duplicates=r['repeated_boundary_rows'], sha256=r['sha256']) for c,r in dvol['currencies'].items()])),
        'The API returned all annual pages without continuation; the fetcher nevertheless implements the documented '
        'continuation-as-next-end-timestamp rule. No dates were filled and no vendor was substituted. '
        'The final daily OHLC exactly matches same-day hourly aggregation for BTC and ETH. '
        'Both series contain annualised 30-day implied volatility in index percentage points.',
        f'Provenance: `../dvol_20261031/fetch_dvol.py` and `verify_timestamps.py`; {common} No random seed. '
        'Request URLs, UTC fetch times, raw-response and CSV hashes, and the official methodology link are in the DVOL manifest. '
        'No descriptive model fitting was run on DVOL.',
        '## Standing caveats',
        'one date is an illustration, not a representative sample; GLD/TLT are American ETF quotes '
        'expressed as OCA European-equivalent Black IVs; timings compare implementations, not '
        'model families; a fitted `beta` is a pricing-measure parameter of one snapshot.',
        f'All {len(audit)} protected manuscript/source/frozen-bundle files retain their before hashes. '
        'Only the two new data bundles are delivered. No package source was changed.']
    (notes/'results.md').write_text('\n\n'.join(text)+'\n', encoding='utf-8')
    if not checks_ok:
        raise RuntimeError('Acceptance checks failed; consult saved results without changing the requested gates.')
    print(json.dumps(core, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    main(parser.parse_args().commit)
