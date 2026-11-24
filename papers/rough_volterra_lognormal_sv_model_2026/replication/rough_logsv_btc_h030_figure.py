"""Figure 1 with independently qualified BTC fits at fixed H=.30 and .50."""
from pathlib import Path
from enum import Enum
import argparse
import hashlib
import json
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
    price=ndtr(d)-k*ndtr(d-v)
    return price if call else price-1+k

def run_local(local,inputs,classical_fit,classical_quotes,rough_fit,rough_quotes,output,evidence_root=None):
    panel=pd.read_csv(inputs)
    assert len(panel)==84 and panel.contract.is_unique
    fits={};tables={};summary=[];replays=[]
    for h,fitfile,qfile in [(.3,rough_fit,rough_quotes),(.5,classical_fit,classical_quotes)]:
        fit,q,replay=replay_fit("BTC",panel,fitfile,qfile,evidence_root)
        replays.append(replay)
        assert fit["calibrated"] and fit["parameters"]["H"]==h
        if h==.3:
            assert fit["qualification"]["calibration_ready"] and fit["independent_confirmation"]
        assert len(q)==45 and q.contract.is_unique
        assert set(q.contract)==set(panel.loc[panel.asset.eq("BTC"),"contract"])
        q["model_iv"]=[brentq(lambda s:black(r.strike/r.common_forward,s*np.sqrt(r.ttm_act365),r.optiontype=="C")-r.price,1e-6,8) for r in q.itertuples()]
        q["H"]=h;tables[h]=q;fits[h]=fit
        for day,g in q.groupby("days_to_expiry"):
            summary.append(dict(H=h,days=int(day),quotes=len(g),inside_bid_ask=int(g.inside_bid_ask.sum()),
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
            color="#2166ac" if asset=="BTC" else "#1b7837"
            ax.fill_between(z.x,100*z.common_bid_iv,100*z.common_ask_iv,color=color,alpha=.18)
            ax.plot(z.x,50*(z.common_bid_iv+z.common_ask_iv),"o",color=color,markersize=2.4)
            if asset=="BTC":
                for h,color,style in [(.5,"#a52127","--"),(.3,"#542788","-")]:
                    zz=tables[h].loc[tables[h].days_to_expiry.eq(day)].sort_values("strike")
                    ax.plot(zz.x,100*zz.model_iv,color=color,linestyle=style,linewidth=1.15)
            ax.axvline(0,color=".6",linestyle=":",linewidth=.6)
            ax.set_title(f"{asset}, {int(day)} days",fontsize=8)
            ax.set_xlim(-.26,.26);ax.set_xticks([-.2,0,.2]);ax.grid(axis="y",alpha=.2)
            if row==1:ax.set_xlabel(r"$\log(K/F)$")
            if col==0:ax.set_ylabel("Implied volatility (%)")
    fig.legend(handles=[
        Line2D([],[],marker="o",linestyle="none",color=".35",markersize=3,label="Observed midpoint"),
        Patch(facecolor=".5",alpha=.18,label="Bid-ask band"),
        Line2D([],[],color="#542788",linewidth=1.15,label=r"BTC fit: $H=0.30$"),
        Line2D([],[],color="#a52127",linestyle="--",linewidth=1.15,label=r"BTC fit: $H=0.50$")],
        loc="outside upper center",ncol=4,frameon=False,columnspacing=1.2,handlelength=2)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,bbox_inches="tight",pad_inches=.035)
    fig.savefig(output.with_suffix(".png"),bbox_inches="tight",pad_inches=.035,dpi=180)
    plt.close(fig)
    pd.concat(tables.values(),ignore_index=True).to_csv(output.with_suffix(".csv"),index=False)
    metadata=dict(standalone_replays=replays,wider_calibration_gated=True,calibration_ready=False,summary=summary,parameters={str(h):f["parameters"] for h,f in fits.items()},
        observed_quotes=84,btc_quotes_per_fit=45,rough_H_added=.3,H02_calibration_gated=True,
        fixed_H_sensitivity_only=True,core_figures_added=0,
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [inputs,classical_fit,classical_quotes,rough_fit,rough_quotes,Path(__file__)]})
    output.with_suffix(".json").write_text(json.dumps(metadata,indent=2))
    print(json.dumps(summary),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ["inputs","classical-fit","classical-quotes","rough-fit","rough-quotes","output"]:
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--evidence-root",type=Path,default=None)
    a=p.parse_args()
    run_local(Locals.FIGURE,a.inputs,a.classical_fit,a.classical_quotes,a.rough_fit,a.rough_quotes,a.output,a.evidence_root)

