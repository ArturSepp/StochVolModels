"""One predeclared enlarged-bound sensitivity after run_pricing_illustration.py.

Keeps quotes, objective, 513-point grid, tolerances and parameterization unchanged.
One warm start is the best primary optimizer vector; iteration cap 60.
This is a bounded sensitivity fit, not a new market sample or independent replication.
"""
import json
import sys
from pathlib import Path
import run_pricing_illustration as study


def main():
    previous=json.loads((study.OUT/"pricing_results.json").read_text(encoding="utf-8"))
    chain, provenance=study.read_snapshot(study.SOURCE)
    result={"purpose":"Expand Heston volatility-of-variance upper bound 4 to 6 and LogSV "
                       "residual volatility-of-volatility upper bound 2 to 3; all other controls fixed.",
            "source_sha256":provenance["sha256"],"quote_time_utc":provenance["quote_time_utc"],
            "script_sha256":study.sha(__file__),"helper_sha256":study.sha(study.__file__),
            "fits":{},"validation":[],
            "limitation":"One warm start per model, 60 function-evaluation cap. Single historical "
                         "snapshot; primary and expanded fits are in-sample with unequal model "
                         "parameter counts. No unconstrained optimum or model dominance claim."}
    validate_only = "--validate-only" in sys.argv
    if validate_only:
        result=json.loads((study.OUT/"bounds_sensitivity.json").read_text(encoding="utf-8"))
        result.setdefault("initial_512_step_checks",result["validation"])
        result["validation"]=[]
        result["script_sha256"]=study.sha(__file__)
        result["helper_sha256"]=study.sha(study.__file__)
    for model in ["Heston","LogSV"]:
        base=previous["calibration"][model]
        best=min(base["runs"],key=lambda r:r["rmse_iv_percentage_points"])
        if validate_only:
            p=(study.HestonParams if model=="Heston" else study.LogSvParams)(
                **result["fits"][model]["parameters"])
        else:
            fitted,p=study.fit(chain,model,60,expanded_bounds=True,
                              starts_override=[best["optimizer_x"]])
            result["fits"][model]=fitted
        (study.OUT/"bounds_sensitivity.json").write_text(
            json.dumps(result,indent=2,default=study.serial,allow_nan=False),encoding="utf-8")
        check=study.independent_mc(model,p,float(chain.ttms[1]),study.np.array([.9,1.,1.1]),
                                   pairs=200000,finest=1024,seed=study.SEED+3)
        c=study.call_chain(float(chain.ttms[1]),study.np.array([.9,1.,1.1]))
        if model=="LogSV":
            prices=study.LogSVPricer().price_chain(c,p)[0]
        else:
            prices=study.HestonPricer().price_chain(c,p)[0]
        check["public_transform_prices"]=study.np.asarray(prices)
        check["transform_minus_fine_mc"]=study.np.asarray(prices)-check["levels"][-1]["prices"]
        result["validation"].append(check)
        (study.OUT/"bounds_sensitivity.json").write_text(
            json.dumps(result,indent=2,default=study.serial,allow_nan=False),encoding="utf-8")
    study.tex(previous,result)
    print("Bounds sensitivity complete",flush=True)


if __name__=="__main__":
    main()
