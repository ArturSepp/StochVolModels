"""Frozen Bloomberg panel: exploratory diagnostics, explicit cohorts and source audit."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_descriptive as d

BASE = Path(__file__).resolve().parent
OLD = BASE.parent/'empirical_20261029'
RAW = pd.read_csv(BASE/'inputs'/'bloomberg_volatility.csv',index_col=0,parse_dates=True)
REF = pd.read_csv(BASE/'inputs'/'bloomberg_reference_split_adjusted.csv',index_col=0,parse_dates=True)
CASH = pd.read_csv(BASE/'inputs'/'bloomberg_reference_cash_adjusted.csv',index_col=0,parse_dates=True)
REQ = json.loads((BASE/'bloomberg_request.json').read_text())
PAIRS = REQ['pairs']
SEED = 2026102911
d.RNG = np.random.default_rng(SEED)


def get_vol(name):
    x=RAW[name+' Index'].dropna()
    return d.clean(x if name=='MOVE' else x/100,name)[0]


def increments(name, vol=None, ref=None, common=None):
    vol=get_vol(name) if vol is None else vol
    ref=REF[PAIRS[name]['reference_ticker']].dropna() if ref is None else ref
    if name=='MOVE':
        ref=ref*100
    levels=pd.concat([vol.rename('vol'),ref.rename('ref')],axis=1,sort=True).dropna()
    levels=levels[(levels>0).all(axis=1)]
    if common is not None:
        levels=levels.reindex(common).dropna()
    gaps=levels.index.to_series().diff().dt.days
    dx=levels.ref.diff() if name=='MOVE' else np.log(levels.ref).diff()
    out=pd.concat([dx.rename('x'),levels.vol.diff().rename('y')],axis=1)
    return out.loc[gaps.between(1,4)].dropna()


def beta_summary(inc,span, save=None):
    if len(inc)<span:
        return {'n':0,'positive_fraction':None,'interval':None}
    a=d.local_beta(inc.x.to_numpy(),inc.y.to_numpy(),span)
    valid=np.isfinite(a[:,0]); a=a[valid]
    pos=(a[:,0]>0).astype(float)
    eligible=len(pos)>=5*span and pos.min()!=pos.max()
    res={'n':len(a),'positive_fraction':float(pos.mean()),
         'median_beta':float(np.median(a[:,0])),
         'interval':d.ci(pos,span) if eligible else None,
         'interval_double_block':d.ci(pos,2*span) if len(pos)>=10*span and pos.min()!=pos.max() else None,
         'pointwise_positive_fraction':float((a[:,1]>0).mean()),
         'interval_note':'conditional fitted-indicator block interval' if eligible else 'omitted: too few blocks or constant indicator'}
    if save:
        pd.DataFrame(a,index=inc.index[valid],columns=['beta','low','high']).to_csv(BASE/'tables'/save)
    return res


def audit_sources():
    old={
        'VIX':pd.read_csv(OLD/'inputs'/'VIX_History.csv',index_col=0,parse_dates=True)['CLOSE'],
        'OVX':pd.read_csv(OLD/'inputs'/'OVX_History.csv',index_col=0,parse_dates=True).iloc[:,0],
        'MOVE':pd.read_csv(OLD/'inputs'/'move.csv',index_col=0,parse_dates=True).iloc[:,0]*100}
    results={}
    rows=[]
    for name,series in old.items():
        series=series.loc['2000-01-01':'2026-08-21']
        pair=pd.concat([series.rename('old'),RAW[name+' Index'].rename('new')],axis=1,sort=True).dropna()
        err=(pair.new-pair.old).abs()
        rec={'n_overlap':len(pair),'start':str(pair.index.min().date()),'end':str(pair.index.max().date()),
             'median_abs_index_points':float(err.median()),'max_abs_index_points':float(err.max()),
             'mean_abs_index_points':float(err.mean()),'correlation':float(pair.corr().iloc[0,1]),
             'largest_differences':[{ 'date':str(t.date()),'old':float(pair.loc[t,'old']), 'bloomberg':float(pair.loc[t,'new']), 'absolute_difference':float(err.loc[t])} for t in err.nlargest(5).index]}
        # Source-only marginal comparison holds dates and train/test split fixed.
        source_scores={}
        for provider in ['old','new']:
            values=pair[provider].to_numpy(); split=int(.7*len(values))
            fit=d.fit_distributions(values[:split])
            diff=d.logdens(values[split:],fit,'gig')-d.logdens(values[split:],fit,'sqrt_gamma')
            source_scores[provider]={'gig_kind':fit['gig_kind'],'score_difference':float(diff.mean()),'ci65':d.ci(diff,65)}
        rec['same_dates_marginal_scores']=source_scores
        # Source-only observed beta, fixed Bloomberg reference and aligned dates.
        rec['same_dates_beta']={}
        for provider in ['old','new']:
            vol=pair[provider] if name=='MOVE' else pair[provider]/100
            inc=increments(name,vol=vol)
            rec['same_dates_beta'][provider]={str(span):beta_summary(inc,span) for span in [65,260]}
        threshold=.005 if name=='VIX' else .1
        flagged=pair.index[err>threshold]
        vol=get_vol(name)
        levels=pd.concat([vol.rename('vol'),REF[PAIRS[name]['reference_ticker']].rename('ref')],axis=1,sort=True).dropna()
        contaminated=levels.index.isin(flagged)
        affected=levels.index[contaminated | np.r_[False,contaminated[:-1]]]
        inc=increments(name)
        retained=inc.loc[~inc.index.isin(affected)]
        rec['disputed_date_sensitivity']={'threshold_index_points':threshold,'flagged_dates':[str(t.date()) for t in flagged],
            'removed_increments':len(inc)-len(retained),
            'beta_excluding_affected_increments':{str(span):beta_summary(retained,span) for span in [65,260]},
            'note':'Post-audit sensitivity only; primary Bloomberg series is not edited. Both increments touching a disputed level are excluded.'}
        results[name]=rec
        rows.append({'series':name,**{k:v for k,v in rec.items() if not isinstance(v,(list,dict))}})
        pair.to_csv(BASE/'audit'/f'{name}_provider_overlap.csv')
    pd.DataFrame(rows).to_csv(BASE/'tables'/'provider_overlap_summary.csv',index=False)
    return results


def main():
    established=[name for name,p in PAIRS.items() if p['cohort']=='established' and len(get_vol(name))>=1500]
    # Common sessions include underlying availability, avoiding a hidden pre-2014 GOOG splice.
    common=None
    for name in established:
        idx=get_vol(name).index.intersection(REF[PAIRS[name]['reference_ticker']].dropna().index)
        common=idx if common is None else common.intersection(idx)
    common=common.sort_values()
    # Latest documented launch in established panel is VXTLT, 22 April 2015.
    common=common[common>=pd.Timestamp('2015-04-22')]
    result={'seed':SEED,'cutoff':'2026-10-28','established_panel':established,
            'common_sessions':{'n':len(common),'start':str(common[0].date()),'end':str(common[-1].date())},
            'source_comparison':audit_sources(),'series':{},'candidate_coverage':[]}
    rows=[]
    for name,p in PAIRS.items():
        raw=RAW[p['volatility_ticker']].dropna().loc['2000-01-01':'2026-10-28']
        s=get_vol(name)
        active=not s.empty and s.index[-1]>=pd.Timestamp('2026-10-21')
        status='established panel' if name in established else p['cohort']
        if s.empty: status='no Bloomberg history'
        elif not active: status='stale history; excluded'
        cov={'series':name,'reference':p['reference_ticker'],'group':p['group'],'cohort':p['cohort'],'status':status,
             'n':len(s),'start':str(s.index.min().date()) if len(s) else None,'end':str(s.index.max().date()) if len(s) else None,
             'nonpositive_removed':int((raw<=0).sum()),'repeat_fraction':float(s.diff().eq(0).iloc[1:].mean()) if len(s)>1 else None,
             'gaps_gt4':int((s.index.to_series().diff().dt.days>4).sum()) if len(s) else 0}
        result['candidate_coverage'].append(cov)
        if name not in established and name!='MOVE':
            # New/relaunched histories are not pooled into established diagnostics.
            launch={'VXSLV':'2025-06-16','VXHYG':'2025-06-16','VXFXI':'2025-11-03','VXGDX':'2025-11-03',
                    'VXSMH':'2025-11-03','BITVX':'2026-03-23','VXUNG':'2025-06-16'}.get(name)
            if launch and active:
                live=s.loc[launch:]
                cov.update({'post_launch_start':launch,'post_launch_n':len(live),'post_launch_mean_percent':float(100*live.mean())})
            continue
        inc=increments(name)
        inc_common=increments(name,common=common)
        rec={'coverage':cov,'mean':float(s.mean()),'median':float(s.median()),'beta':{},'common_beta':{},'cash_adjusted_beta':{},'post2025_beta':{}}
        for span in [65,260]:
            rec['beta'][str(span)]=beta_summary(inc,span,f'{name}_panel_beta_{span}.csv')
            rec['common_beta'][str(span)]=beta_summary(inc_common,span)
            rec['cash_adjusted_beta'][str(span)]=beta_summary(increments(name,ref=CASH[p['reference_ticker']].dropna()),span)
            rec['post2025_beta'][str(span)]=beta_summary(inc.loc['2025-02-10':],span)
        rec['paired_increment_dates']={'n':len(inc),'start':str(inc.index[0].date()),'end':str(inc.index[-1].date())}
        rec['acf']={str(lag):{'estimate':float(d.lag_correlations(s.to_numpy(),[lag])[0]),
                                 'ci250':d.acf_ci(s.to_numpy(),lag,250),'ci500':d.acf_ci(s.to_numpy(),lag,500)} for lag in [21,63,120]}
        ntrain=int(.7*len(s)); train,test=s.to_numpy()[:ntrain],s.to_numpy()[ntrain:]
        fit=d.fit_distributions(train)
        diff=d.logdens(test,fit,'gig')-d.logdens(test,fit,'sqrt_gamma')
        fit.update({'n_train':len(train),'n_test':len(test),'train_end':str(s.index[ntrain-1].date()),'test_start':str(s.index[ntrain].date()),
                    'score_difference':float(diff.mean()),'score_ci65':d.ci(diff,65),'score_ci130':d.ci(diff,130)})
        rec['density']=fit
        matched=s.reindex(common).dropna().to_numpy()
        mtrain=int(.7*len(matched)); mf=d.fit_distributions(matched[:mtrain])
        mdiff=d.logdens(matched[mtrain:],mf,'gig')-d.logdens(matched[mtrain:],mf,'sqrt_gamma')
        rec['common_density']={'n':len(matched),'n_train':mtrain,'n_test':len(matched)-mtrain,
            'gig_kind':mf['gig_kind'],'fit':mf,'score_difference':float(mdiff.mean()),'ci65':d.ci(mdiff,65),'ci130':d.ci(mdiff,130)}
        rec['common_acf63']=float(d.lag_correlations(matched,[63])[0])
        result['series'][name]=rec
        rows.append({'series':name,'group':p['group'],'n':len(s),'start':cov['start'],'end':cov['end'],
                     'paired_increment_start':rec['paired_increment_dates']['start'],
                     'positive_beta65':rec['beta']['65']['positive_fraction'], 'positive_beta260':rec['beta']['260']['positive_fraction'],
                     'common_positive_beta65':rec['common_beta']['65']['positive_fraction'],
                     'common_positive_beta260':rec['common_beta']['260']['positive_fraction'],
                     'cash_positive_beta65':rec['cash_adjusted_beta']['65']['positive_fraction'],
                     'post2025_positive_beta65':rec['post2025_beta']['65']['positive_fraction'],
                     'acf63':rec['acf']['63']['estimate'],'acf120':rec['acf']['120']['estimate'],
                     'gig_kind':fit['gig_kind'],'gig_minus_sqrtGamma':fit['score_difference'],
                     'common_gig_minus_sqrtGamma':float(mdiff.mean()),'common_acf63':rec['common_acf63'],
                     'score_ci65_low':fit['score_ci65'][0],'score_ci65_high':fit['score_ci65'][1]})
        print(name, 'n',len(s),'beta65+',round(rec['beta']['65']['positive_fraction'],4),'score',round(fit['score_difference'],4),flush=True)
    result['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (BASE/'panel_results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame(rows).to_csv(BASE/'tables'/'expanded_panel.csv',index=False)
    pd.DataFrame(result['candidate_coverage']).to_csv(BASE/'tables'/'candidate_coverage.csv',index=False)
    # MOVE is displayed separately in the manuscript, not mixed into price-beta charts.
    names=established; yy=np.arange(len(names))
    fig,axes=plt.subplots(1,2,figsize=(12,8),sharey=True,layout='constrained')
    for ax,key,title in zip(axes,['beta','common_beta'],['Native histories','Same common sessions and reference coverage']):
        for i,n in enumerate(names):
            r=result['series'][n][key]['65']; interval=r['interval']
            if interval: ax.hlines(i,100*interval[0],100*interval[1],color='#285e8e',lw=1.4)
        ax.scatter([100*result['series'][n][key]['65']['positive_fraction'] for n in names],yy,color='#285e8e',label='span 65',s=28)
        ax.scatter([100*result['series'][n][key]['260']['positive_fraction'] for n in names],yy,marker='x',color='#bb6730',label='span 260',s=32)
        ax.set(title=title,xlabel='Positive observed-beta estimates (%)',xlim=(-3,103),yticks=yy,yticklabels=names)
        ax.legend(loc='lower right',fontsize=9)
    axes[0].invert_yaxis()
    fig.savefig(BASE/'figures'/'expanded_beta.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,8),sharey=True,layout='constrained')
    for i,n in enumerate(names):
        rec=result['series'][n]
        a=rec['acf']['63']; score=rec['density']
        axes[0].hlines(i,*a['ci250'],color='#285e8e');axes[0].scatter(a['estimate'],i,color='#285e8e',s=24)
        axes[1].hlines(i,*score['score_ci65'],color='#285e8e');axes[1].scatter(score['score_difference'],i,color='#285e8e',s=24)
    for ax in axes:
        ax.axvline(0,color='grey',lw=.6);ax.set(yticks=yy,yticklabels=names)
    axes[0].set(title='Persistence of observed levels',xlabel='Lag-63 correlation; 250-observation block interval',xlim=(-.1,1))
    axes[1].set(title='Chronological marginal-density transfer',xlabel='GIG minus sqrt-Gamma held-out log score')
    axes[0].invert_yaxis()
    axes[0].set_yticks(yy, labels=names)
    axes[0].tick_params(axis='y', labelleft=True)
    fig.savefig(BASE/'figures'/'expanded_persistence_density.png',dpi=180);plt.close(fig)
    print('Expanded panel complete',result['common_sessions'],flush=True)


if __name__=='__main__':
    main()
