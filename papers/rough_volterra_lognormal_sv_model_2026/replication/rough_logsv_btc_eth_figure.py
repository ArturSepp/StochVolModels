"""Figure 1: independently qualified BTC and ETH fits at fixed H=.30 and .50."""
from pathlib import Path
from enum import Enum
import argparse,hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.special import ndtr
from scipy.optimize import brentq

if __package__:
    from .rough_logsv_saved_fits import replay_fit
else:
    from rough_logsv_saved_fits import replay_fit

class Locals(Enum):
    FIGURE="figure"

def black(k,v,call):
    d=-np.log(k)/v+v/2
    value=ndtr(d)-k*ndtr(d-v)
    return value if call else value-1+k

def run_local(local,inputs,btc_classical,btc_rough,eth,output,evidence_root=None):
    panel=pd.read_csv(inputs)
    assert len(panel)==84 and panel.contract.is_unique
    files=[inputs,Path(__file__)];tables={};fits={};summary=[];replays=[]
    sources=[("BTC",.5,btc_classical/"classical_full_fit.json",btc_classical/"classical_full_fit_quotes.csv"),
             ("BTC",.3,btc_rough/"rough_h030_fit.json",btc_rough/"rough_h030_fit_quotes.csv"),
             ("ETH",.5,eth/"classical_fit.json",eth/"classical_fit_quotes.csv"),
             ("ETH",.3,eth/"rough_h030_fit.json",eth/"rough_h030_fit_quotes.csv")]
    for asset,h,fitfile,qfile in sources:
        fit,q,replay=replay_fit(asset,panel,fitfile,qfile,evidence_root)
        replays.append(replay)
        assert fit["calibrated"] and fit["parameters"]["H"]==h
        if h==.3:assert fit["qualification"]["calibration_ready"] and fit["independent_confirmation"]
        expected=panel.loc[panel.asset.eq(asset)]
        assert len(q)==len(expected) and q.contract.is_unique and set(q.contract)==set(expected.contract)
        for name in ["common_forward","strike","bid_price","ask_price","ttm_act365"]:
            assert np.array_equal(q.set_index("contract").loc[expected.contract,name],expected[name])
        q["model_iv"]=[brentq(lambda v:black(r.strike/r.common_forward,v*np.sqrt(r.ttm_act365),r.optiontype=="C")-r.price,1e-6,8) for r in q.itertuples()]
        q["H"]=h;tables[(asset,h)]=q;fits[f"{asset}_H{h}"]=fit["parameters"];files.extend([fitfile,qfile])
        for day,g in q.groupby("days_to_expiry"):
            summary.append(dict(asset=asset,H=h,days=int(day),quotes=len(g),inside_bid_ask=int(g.inside_bid_ask.sum()),
                rms_midpoint_error_spreads=float(np.sqrt(np.mean(g.error_spreads**2))),
                max_abs_midpoint_error_spreads=float(g.error_spreads.abs().max()),
                rms_iv_error_percentage_points=float(100*np.sqrt(np.mean((g.model_iv-.5*(g.common_bid_iv+g.common_ask_iv))**2)))))
    plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"mathtext.fontset":"cm",
        "font.size":8,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,
        "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    fig,axes=plt.subplots(2,3,figsize=(160/25.4,108/25.4),layout="constrained",sharex=True)
    for row,asset in enumerate(["BTC","ETH"]):
        g=panel[panel.asset.eq(asset)]
        for col,day in enumerate(sorted(g.days_to_expiry.unique())):
            z=g[g.days_to_expiry.eq(day)].sort_values("strike");ax=axes[row,col]
            observed="#2166ac" if asset=="BTC" else "#1b7837"
            ax.fill_between(z.x,100*z.common_bid_iv,100*z.common_ask_iv,color=observed,alpha=.18)
            ax.plot(z.x,50*(z.common_bid_iv+z.common_ask_iv),"o",color=observed,markersize=2.4,zorder=4)
            # Draw the dashed baseline last so both curves remain visible where they overlap.
            for h,color,style in [(.3,"#542788","-"),(.5,"#a52127","--")]:
                zz=tables[(asset,h)].loc[tables[(asset,h)].days_to_expiry.eq(day)].sort_values("strike")
                ax.plot(zz.x,100*zz.model_iv,color=color,linestyle=style,linewidth=1.15)
            ax.axvline(0,color=".6",linestyle=":",linewidth=.6)
            ax.set_title(f"{asset}, {int(day)} days",fontsize=8)
            ax.set_xlim(-.26,.26);ax.set_xticks([-.2,0,.2]);ax.grid(axis="y",alpha=.2)
            if row==1:ax.set_xlabel(r"$\log(K/F)$")
            if col==0:ax.set_ylabel("Implied volatility (%)")
    fig.legend(handles=[
        Line2D([],[],marker="o",linestyle="none",color=".35",markersize=3,label="Observed midpoint"),
        Patch(facecolor=".5",alpha=.18,label="Bid-ask band"),
        Line2D([],[],color="#542788",linewidth=1.15,label=r"Fit: $H=0.30$"),
        Line2D([],[],color="#a52127",linestyle="--",linewidth=1.15,label=r"Fit: $H=0.50$")],
        loc="outside upper center",ncol=4,frameon=False,columnspacing=1.2,handlelength=2)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,bbox_inches="tight",pad_inches=.035)
    fig.savefig(output.with_suffix(".png"),bbox_inches="tight",pad_inches=.035,dpi=180)
    plt.close(fig)
    pd.concat(tables.values(),ignore_index=True).to_csv(output.with_suffix(".csv"),index=False)
    metadata=dict(standalone_replays=replays,wider_calibration_gated=True,calibration_ready=False,summary=summary,parameters=fits,observed_quotes=84,btc_quotes_per_fit=45,eth_quotes_per_fit=39,
        added_asset="ETH",H02_calibration_gated=True,fixed_H_sensitivity_only=True,core_figures_added=0,
        lines="Connect qualified model values at the observed strikes; no new wing extrapolation",
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    output.with_suffix(".json").write_text(json.dumps(metadata,indent=2))
    print(json.dumps(summary),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ["inputs","btc-classical","btc-rough","eth","output"]:p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--evidence-root",type=Path,default=None)
    a=p.parse_args()
    run_local(Locals.FIGURE,a.inputs,a.btc_classical,a.btc_rough,a.eth,a.output,a.evidence_root)

