"""Finalize SPX: retain Heston fit, discard unvalidated LogSV fit, preserve attempts."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import run_equity_benchmark as e

OUT = Path(__file__).resolve().parent

def run():
    archive = OUT/'equity_discarded_1025_grid_results.json'
    if not archive.exists():
        archive.write_bytes((OUT/'equity_benchmark_results.json').read_bytes())
    r = json.loads(archive.read_text(encoding='utf-8'))
    chain, _ = e.read_snapshot()
    trial = r['calibration']['LogSV']
    p = e.base.LogSvParams(**trial['parameters'])
    meshes = {}
    for n in [4097,8193]:
        raw = e.logsv_prices(chain,p,n,.0056)
        meshes[str(n)] = {'prices_forward_units':[np.asarray(x) for x in raw],
            'ivols':[np.asarray(x) for x in chain.compute_model_ivols_from_chain_data(raw)]}
    diff = {
        'max_abs_price_diff':float(np.max(abs(e.base.flatten(meshes['4097']['prices_forward_units'])-e.base.flatten(meshes['8193']['prices_forward_units'])))),
        'max_abs_iv_diff_pp':float(100*np.max(abs(e.base.flatten(meshes['4097']['ivols'])-e.base.flatten(meshes['8193']['ivols']))))}
    v = r['validation'][1]
    c = e.base.call_chain(v['T'],np.array(v['strikes_over_forward']))
    spot = e.logsv_prices(c,p,4097,.0056)[0]
    extended = e.logsv_prices(c,p,8193,.0028)[0]
    r['LogSV_refined_process_validation'] = {
        'status':'FAILED: discrepancies remain after mesh and cutoff refinement',
        'parameters':trial['parameters'],'T':v['T'],'strikes_over_forward':v['strikes_over_forward'],
        'refined_transform_prices':spot,'extended_cutoff_transform_prices':extended,
        'cutoff1000_to2000_price_difference':extended-spot,
        'refined_transform_minus_512_step_mc':spot-np.asarray(v['levels'][-1]['prices']),
        'mc_standard_errors':v['levels'][-1]['standard_errors'],
        'last_mc_time_grid_change':v['grid_changes'][-1]['price_change'],
        'mc_steps':[x['steps'] for x in v['levels']], 'antithetic_pairs':v['antithetic_pairs'],
        'interpretation':'The right call differs by about 13 sampling SEs; the measured last time-grid change is much smaller. Grid refinement does not certify the affine expansion.'}
    fine_path = OUT/'equity_heston_mc_2048.json'
    if fine_path.exists():
        fine = json.loads(fine_path.read_text(encoding='utf-8'))
    else:
        hp = e.base.HestonParams(**r['calibration']['Heston']['parameters'])
        fine = e.base.independent_mc('Heston',hp,v['T'],np.array(v['strikes_over_forward']),100000,2048)
        q = e.heston_prices(c,hp,4097,.0028)[0]
        fine['refined_black_control_transform_prices'] = q
        fine['refined_transform_minus_fine_mc'] = q-fine['levels'][-1]['prices']
        fine_path.write_text(json.dumps(fine,indent=2,default=e.base.serial),encoding='utf-8')
    r['heston_independent_mc_2048'] = fine
    r['heston_independent_mc_2048_source_sha256'] = e.base.sha(fine_path)
    r['independent_heston_quadrature_audit'] = {
        'method':'Separate classical Heston characteristic function and Lewis quadrature over [0,infinity), all 78 quotes; read-only independent audit.',
        'max_transform_difference_sampled_frequencies':1.12e-16,
        'max_abs_price_difference_forward_units':1.389e-9,
        'max_abs_iv_difference_percentage_points':2.675e-5,
        'independent_rmse_iv_percentage_points':.5400941134311,
        'source_bid_ask_premium_black_roundtrip_max_difference_forward_units':6.531e-8}
    r['study'] = 'Archived SPXW: bounded Heston illustration and failed LogSV numerical validation'
    r['final_status'] = {'Heston':'Representative prices independently validated; bounded in-sample fit.',
        'LogSV':'Not validated. Coarse-grid fit discarded; model-fit RMSE withheld. Refined fixed-parameter prices are a diagnostic.'}
    r['discarded_LogSV_calibration'] = trial
    r['calibration']['LogSV'] = {'status':'discarded; do not quote fit RMSE or rank models',
        'rmse_iv_percentage_points':None,'parameters':trial['parameters'],
        'reason':'Fitted-parameter mesh and independent process validation failed.'}
    r['preliminary_validation_do_not_use_for_final_interpretation'] = r.pop('validation')
    r['LogSV_fixed_trial_parameters_mesh_diagnostic'] = {
        'parameters':trial['parameters'],'fit_node_count':1025,'refit_performed':False,
        'frequency_cutoff':1000.,'meshes':meshes,'4097_to8193':diff,
        'purpose':'Fixed-parameter repricing diagnoses numerical integration; refined curves are not refitted estimates.'}
    r['fourier_control_variate'] = {
        'method':'Canonical model-grid premium minus canonical Black-grid premium plus analytic Black premium; identical grid/forward/discount/option type.',
        'black_volatility':.15,'scope':'Cancels common payoff-pole quadrature error, not affine-expansion error.',
        'initial_point_pass_did_not_certify_fit':'1025-to2049 difference 6.4e-11 initially, up to 1.09e-4 after fitting.'}
    r['finalization_script_sha256'] = e.base.sha(__file__)
    r['discarded_trial_archive_sha256'] = e.base.sha(archive)
    r['cautions'] += [
        'Trial 1e-12 forward-unit price floor used on 36 of 275 LogSV calls; no endpoint premium used it. This does not validate the affine expansion.',
        'Heston noise reaches upper bound 2. LogSV trial noise reaches upper bound 3 and kappa2 reaches lower bound .05.',
        'No further optimization after failed LogSV process validation; no LogSV model RMSE or cross-model ranking is reported.']
    repo = Path(e.base.stochvolmodels.__file__).resolve().parents[2]
    paths = ['src/stochvolmodels/pricers/heston_pricer.py','src/stochvolmodels/pricers/logsv_pricer.py',
        'src/stochvolmodels/pricers/logsv/affine_expansion.py','src/stochvolmodels/utils/mgf_pricer.py']
    r['canonical_source_files_sha256'] = {str(repo/path):e.base.sha(repo/path) for path in paths}
    e.dump(r)
    draw(chain,r)
    write_text(r)
    print('Finalized: bounded Heston illustration; LogSV validation failed, RMSE withheld.',flush=True)

def draw(chain,r):
    fig,axs = plt.subplots(2,3,figsize=(12,7.5),layout='constrained')
    old = r['discarded_LogSV_calibration']['model_ivols']
    fine = r['LogSV_fixed_trial_parameters_mesh_diagnostic']['meshes']['8193']['ivols']
    for i in range(3):
        k = np.log(chain.strikes_ttms[i])
        for j in range(2):
            ax=axs[j,i]
            ax.fill_between(k,100*chain.bid_ivs[i],100*chain.ask_ivs[i],color='#d1d5db',label='Archived bid/ask')
            ax.plot(k,100*chain.get_mid_vols()[i],'o',ms=3,color='black',label='IV midpoint')
            ax.set(xlabel='log(strike / forward)',ylabel='Implied volatility (%)')
            ax.grid(alpha=.2)
        axs[0,i].plot(k,100*np.asarray(r['calibration']['Heston']['model_ivols'][i]),color='#d97706',label='Heston bounded fit')
        axs[0,i].set_title(f'Heston: {chain.ttms[i]*365:.1f} calendar days')
        axs[1,i].plot(k,100*np.asarray(old[i]),'--',color='#9ca3af',label='Discarded 1025-node fit')
        axs[1,i].plot(k,100*np.asarray(fine[i]),color='#7c3aed',label='8193 nodes; same parameters')
        axs[1,i].set_title('LogSV diagnostic: not validated',color='#7c3aed')
    axs[0,0].legend(fontsize=8);axs[1,0].legend(fontsize=8)
    fig.suptitle('SPXW, 21 October 2021: bounded Heston fit and LogSV numerical failure',fontsize=12)
    fig.savefig(OUT/'equity_benchmark.png',dpi=180)
    plt.close(fig)

def write_text(r):
    h=r['calibration']['Heston'];lv=r['LogSV_refined_process_validation'];hv=r['heston_independent_mc_2048']
    tex=r"""% Generated by finalize_equity_benchmark.py; no LogSV fit ranking.
\paragraph{An equity snapshot and a failed numerical validation.}
The local Cboe-sourced SPXW cache is stamped 21 October 2021, 16:00 New York.
We retain 78 OTM quotes at approximately 15, 43 and 71 calendar days, using
5 November, 3 December and 31 December expiries. The 26 November early-close
expiry is excluded before fitting because it conflicts with the cache's fixed
16:00 expiry convention. Common filters require $|\log(K/F)|\leq0.12$,
positive ordered premiums and bid--ask IV spreads at most three volatility
percentage points. Each eligible slice is thinned to 25 equally spaced strike
indices, plus its nearest ATM strike. Recorded forwards and discounts are
constant within each slice. Re-inverting normalized bid and ask premiums agrees
with the cache IVs within $3.8\times10^{-7}$ in decimal volatility.

Both models use two starting points and the same midpoint-IV objective, with
equal maturity weights and uniform retained-strike weights. Heston's bounded
in-sample IV RMSE is HRMSE volatility percentage points; its noise parameter
reaches the specified upper bound of two. Both Heston runs converge by optimizer
tolerances. Independent classical-transform quadrature checks all 78 prices,
with maximum difference $1.4\times10^{-9}$ in forward units. At the 43-day maturity,
independent conditional Monte Carlo with 100000 antithetic pairs and coupled
512, 1024 and 2048-step grids agrees with three refined prices within 1.5 sampling
standard errors; the maximum final time-grid change is 0.133 basis points of forward.
This is a bounded single-date illustration, not an unrestricted optimum or an
out-of-sample model ranking.

The LogSV attempt does not pass numerical validation, so its model-fit RMSE is
withheld. An analytic Black control variate cancels common Fourier-payoff
quadrature error, but a grid that passed at the starting parameters fails at the
fitted high-noise boundary: 1025-to-2049 refinement changes IVs by up to 0.458
percentage points. At fixed trial parameters, 4097-to-8193 refinement reduces
the maximum difference to $2.7\times10^{-5}$ IV percentage points. Nevertheless,
refined 43-day call prices at $K/F=0.95,1,1.05$ differ from independent process
simulation by 1.608, 1.007 and 0.988 basis points of forward. The corresponding
sampling standard errors are 0.531, 0.362 and 0.076, and measured final time-grid
changes have absolute values below 0.089. Doubling the Fourier cutoff does not
change these representative prices. These LogSV checks use 100000 antithetic
pairs and coupled 128, 256 and 512-step grids. The right-call discrepancy is about thirteen
sampling standard errors; quadrature convergence does not certify the affine
expansion. No further optimization is attempted after this process-validation
failure. Trial parameters, all bounds, a budget-terminated second start and
failed-grid outputs remain in the companion archive. A trial-point price floor
was used on 36 of 275 LogSV evaluations; no endpoint premium uses it.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/equity_benchmark.png}
\caption{SPXW on 21 October 2021. The top row shows the bounded Heston fit.
The lower row shows the discarded LogSV grid and a finer-grid repricing at the
same trial parameters; it is not a validated or refitted model comparison.}
\end{figure}
"""
    (OUT/'equity_results_snippet.tex').write_text(tex.replace('HRMSE',f"{h['rmse_iv_percentage_points']:.3f}"),encoding='utf-8')
    lines=['# SPXW numerical audit','','**Outcome:** bounded Heston illustration validated at disclosed tolerances; LogSV validation failed. Its model-fit RMSE is withheld.','',
        f"Heston IV RMSE: **{h['rmse_iv_percentage_points']:.6f} percentage points**. Both starts converge; volatility of variance reaches upper bound 2. The common sample is 78 OTM quotes at 15.000, 43.042 and 71.042 calendar days on 21 October 2021, 16:00 New York. The 26 November early-close expiry is excluded before fitting.",'',
        '| Model | Strike/F | Refined transform minus MC (bp forward) | MC SE (bp forward) | Last time-grid change (bp forward) |',
        '|---|---:|---:|---:|---:|']
    for model,gaps,se,changes in [
        ('Heston (2048 steps)',hv['refined_transform_minus_fine_mc'],hv['levels'][-1]['standard_errors'],hv['grid_changes'][-1]['price_change']),
        ('LogSV (512 steps)',lv['refined_transform_minus_512_step_mc'],lv['mc_standard_errors'],lv['last_mc_time_grid_change'])]:
        for i,k in enumerate([.95,1,1.05]):
            lines.append(f'| {model} | {k:.2f} | {1e4*gaps[i]:.4f} | {1e4*se[i]:.4f} | {1e4*changes[i]:.4f} |')
    lines += ['',
        'Heston additionally agrees with independent classical characteristic-function / Lewis quadrature over the infinite frequency interval on all 78 quotes (maximum normalized-price difference 1.389e-9). Default public-wrapper grids are not accuracy references.','',
        'The first LogSV start converges at 20 evaluations; the second reaches its 20-evaluation budget. The noise parameter reaches 3 and kappa2 reaches .05, the imposed bounds. A 1e-12 forward-unit floor supports trial IV inversion on 36 of 275 calls; all endpoint raw prices are positive. These facts do not validate the affine expansion.','',
        'The fitted 1025-node grid fails 2049-node refinement: maximum IV change 0.458 percentage points. At fixed trial parameters, 4097-to-8193 refinement changes prices by 6.91e-9 forward units and IVs by 2.68e-5 percentage points. The refined right-call process discrepancy remains about 13 sampling SEs, while doubling the cutoff changes representative prices by less than 2e-16. The failed fit is discarded.','',
        'The lower figure row holds discarded trial parameters fixed. It is a numerical diagnostic, not a refined calibration. No model ranking or speed claim follows.','',
        'Reproduction: use C:\\Python\\StochVolModels312\\Scripts\\python.exe. Run run_equity_benchmark.py --max-nfev 20, then finalize_equity_benchmark.py. The finalizer preserves the initial output, performs fitted-parameter mesh/cutoff checks, embeds or regenerates finer Heston MC, suppresses the failed LogSV model RMSE, and writes the figure and final snippet.','']
    (OUT/'equity_benchmark_audit.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':
    run()
