"""Render the two panels of Figure 4 from the archived Table 4 and the W3 result JSON."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np


YEARS = (1, 2, 5)
COLORS = {0.: "#28292c", .3: "#166695", .6: "#ba5921", .9: "#318063", 1.2: "#865191"}
METRIC = "time_averaged_second_moment_ratio"


def render(left_source, right_source, output):
    left = json.loads(left_source.read_text(encoding="utf-8"))
    right = json.loads(right_source.read_text(encoding="utf-8"))
    exact = right["benchmarks"]["linear_exact"]
    bounds = {(c["kappa2"],c["T"]): c["bound_over_theta2"] for c in right["benchmarks"]["three_factor_caps"]}
    kappas = [row["kappa2"] for row in left["results"]]
    if kappas != list(COLORS) or [r["kappa2"] for r in right["results"]] != list(COLORS):
        raise ValueError("The figure requires the five W3 coefficient rows in order.")
    style = {"font.family":"DejaVu Sans","mathtext.fontset":"dejavusans","font.size":9,
             "axes.labelsize":9,"axes.titlesize":10,"axes.spines.top":False,"axes.spines.right":False,
             "axes.edgecolor":"#686e74","xtick.color":"#343a40","ytick.color":"#343a40","pdf.fonttype":42}
    with plt.rc_context(style):
        fig,(a,b) = plt.subplots(1,2,figsize=(7.55,4.0),gridspec_kw={"width_ratios":(.9,1.12)})
        fig.subplots_adjust(left=.091,right=.98,top=.905,bottom=.287,wspace=.38)
        for ax in (a,b):
            ax.grid(axis="y",which="major",color="#d8dde0",lw=.55)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=8)
        a.set_title("(a)  Raw conditional forward",loc="left")
        a.axhline(1.,color="#777",lw=.85,ls="--")
        a.axvline(.6,color="#777",lw=.85,ls=":")
        for g,color,marker in ((360,"#326b9e","o"),(720,"#d48137","s"),(1440,"#3a8565","^")):
            rows = [next(c for c in r["left"] if c["steps_per_year"]==g)["conditional_forward"] for r in left["results"]]
            a.errorbar(kappas,[v["mean"] for v in rows],yerr=[v["standard_error"] for v in rows],
                       color=color,marker=marker,ms=3.5,lw=1.05,capsize=2,label=f"{g:,} steps/year")
        a.set(xlabel=r"Quadratic coefficient $\kappa_2$",ylabel=r"Raw conditional mean of $S_1$")
        a.set_xticks(kappas)
        a.legend(frameon=False,loc="lower right",fontsize=7.3,handlelength=1.9)
        b.set_title("(b)  Time-averaged second moment",loc="left")
        b.set_yscale("log")
        b.set_ylim(1.3,4300)
        b.set_yticks([2,5,10,25,100,500,2500],["2","5","10","25","100","500","2,500"])
        b.yaxis.set_minor_locator(NullLocator())
        b.set_xlim(.7,5.3)
        b.set_xticks(YEARS)
        b.set(xlabel="Horizon (years)",ylabel=r"$\mathrm{E}[I_T]/(T\theta^2)$")
        b.plot(YEARS,[v[METRIC] for v in exact],"-D",color=COLORS[0.],lw=1.5,ms=3.6,zorder=5)
        for group in right["results"]:
            k = group["kappa2"]
            color = COLORS[k]
            cells = {(c["T"],c["steps_per_year"]):c[METRIC] for c in group["right"]}
            if k:
                b.plot(YEARS,[bounds[k,t] for t in YEARS],ls="--",lw=1,color=color)
            for g, shift, marker in [(360,-.065,"o"),(720,.065,"s")]:
                xs = np.array(YEARS)+shift
                vals = [cells[t,g] for t in YEARS]
                b.errorbar(xs,[v["mean"] for v in vals],yerr=[v["batch_standard_error"] for v in vals],
                           color=color,lw=1,ls=":" if k==0 else "-",fmt=marker,
                           mfc="white" if g==360 else color,mew=.85,ms=3.35,capsize=2.2,
                           elinewidth=.9,capthick=.8,zorder=4 if k==0 else 3)
        types = [Line2D([],[],color=COLORS[0.],ls="-",marker="D",ms=3.5,label="Linear exact"),
                 Line2D([],[],color=COLORS[0.],ls=":",marker="s",ms=3.5,label="Linear MC control"),
                 Line2D([],[],color="#555",ls="--",label="Proved bound"),
                 Line2D([],[],color="#555",marker="o",mfc="white",ls="none",ms=4,label="360 steps/year"),
                 Line2D([],[],color="#555",marker="s",ls="none",ms=4,label="720 steps/year")]
        fig.legend(handles=types,loc="lower center",bbox_to_anchor=(.53,.096),ncol=5,frameon=False,
                   fontsize=7.6,handlelength=1.9,columnspacing=1.5,handletextpad=.45)
        lines = [Line2D([],[],color=color,lw=2,label=rf"$\kappa_2={k:g}$")
                 for k,color in COLORS.items() if k]
        fig.legend(handles=lines,loc="lower center",bbox_to_anchor=(.53,.024),ncol=4,frameon=False,
                   fontsize=8,handlelength=1.8,columnspacing=1.7,handletextpad=.5)
        output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output,metadata={"Title":"Raw conditional forwards and time-averaged second moments",
                    "Author":"Artur Sepp and Parviz Rakhmonov","CreationDate":None,"ModDate":None})
        plt.close(fig)


if __name__ == "__main__":
    rep = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left",type=Path,default=rep/"results/F4_quadratic_drift.json")
    parser.add_argument("--right",type=Path,default=rep/"results/F4_right_panel_time_averaged.json")
    parser.add_argument("--output",type=Path,default=rep.parent/"paper/figures/F4_time_averaged.pdf")
    args = parser.parse_args()
    render(args.left,args.right,args.output)

