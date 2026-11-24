"""Produce the focused BTC calibration overlay from qualified, frozen quote prices."""
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
    FIGURE = "figure"

def black(k,v,call):
    d1=-np.log(k)/v+v/2
    c=ndtr(d1)-k*ndtr(d1-v)
    return c if call else c-1+k

def run_local(local,inputs,fit,quotes,output,evidence_root=None):
    panel=pd.read_csv(inputs)
    metadata,q,replay=replay_fit("BTC",panel,fit,quotes,evidence_root)
    assert metadata["calibrated"] and metadata["parameters"]["H"]==.5
    assert len(panel)==84 and len(q)==45 and q.contract.is_unique
    assert set(q.contract)==set(panel.loc[panel.asset.eq("BTC"),"contract"])
    iv=[]
    for row in q.itertuples():
        k=row.strike/row.common_forward
        iv.append(brentq(lambda s:black(k,s*np.sqrt(row.ttm_act365),row.optiontype=="C")-row.price,1e-6,8))
    q["model_iv"]=iv
    output.parent.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],
                         "mathtext.fontset":"cm","font.size":8,"axes.labelsize":8,
                         "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,
                         "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    fig,axes=plt.subplots(2,3,figsize=(160/25.4,108/25.4),layout="constrained",sharex=True)
    for row,asset in enumerate(["BTC","ETH"]):
        g=panel[panel.asset.eq(asset)]
        for col,day in enumerate(sorted(g.days_to_expiry.unique())):
            z=g[g.days_to_expiry.eq(day)].sort_values("strike")
            ax=axes[row,col]
            color="#2166ac" if asset=="BTC" else "#1b7837"
            ax.fill_between(z.x,100*z.common_bid_iv,100*z.common_ask_iv,color=color,alpha=.18)
            ax.plot(z.x,50*(z.common_bid_iv+z.common_ask_iv),"o",color=color,markersize=2.4)
            if asset=="BTC":
                zz=q[q.days_to_expiry.eq(day)].sort_values("strike")
                ax.plot(zz.x,100*zz.model_iv,color="#a52127",linewidth=1.15)
            ax.axvline(0,color=".6",linestyle=":",linewidth=.6)
            ax.set_title(f"{asset}, {int(day)} days",fontsize=8)
            ax.set_xlim(-.26,.26);ax.set_xticks([-.2,0,.2]);ax.grid(axis="y",alpha=.2)
            if row==1:ax.set_xlabel(r"$\log(K/F)$")
            if col==0:ax.set_ylabel("Implied volatility (%)")
    fig.legend(handles=[Line2D([],[],marker="o",linestyle="none",color=".35",markersize=3,label="Observed midpoint"),
                        Patch(facecolor=".5",alpha=.18,label="Bid–ask band"),
                        Line2D([],[],color="#a52127",linewidth=1.15,label=r"BTC fit: $H=0.50$")],
               loc="outside upper center",ncol=3,frameon=False)
    fig.savefig(output,bbox_inches="tight",pad_inches=.035)
    fig.savefig(output.with_suffix(".png"),bbox_inches="tight",pad_inches=.035,dpi=180)
    plt.close(fig)
    q.to_csv(output.with_suffix(".csv"),index=False)
    summary=[]
    for day,z in q.groupby("days_to_expiry"):
        summary.append(dict(days=int(day),quotes=len(z),inside_bid_ask=int(z.inside_bid_ask.sum()),
                            rms_midpoint_error_spreads=float(np.sqrt(np.mean(z.error_spreads**2))),
                            max_abs_midpoint_error_spreads=float(z.error_spreads.abs().max()),
                            rms_iv_error_percentage_points=float(100*np.sqrt(np.mean((z.model_iv-.5*(z.common_bid_iv+z.common_ask_iv))**2)))))
    output.with_suffix(".json").write_text(json.dumps(dict(
        standalone_replays=[replay],wider_calibration_gated=True,calibration_ready=False,
        parameters=metadata["parameters"],qualification=metadata["qualification"],summary=summary,
        rough_overlays_added=False,rough_H=[.2,.3],rough_status="Pricing and raw-forward precision gates remain open",
        inputs={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in [inputs,fit,quotes,Path(__file__)]}),indent=2))
    print(json.dumps(summary),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--inputs",type=Path,required=True);p.add_argument("--fit",type=Path,required=True)
    p.add_argument("--quotes",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--evidence-root",type=Path,default=None)
    a=p.parse_args()
    run_local(Locals.FIGURE,a.inputs,a.fit,a.quotes,a.output,a.evidence_root)

