# Jump risk premia in the presence of clustered jumps

Francis Liu, Natalie Packham and Artur Sepp. Accepted for publication in the *SIAM Journal on Financial Mathematics* (SIFIN).

[Read the author manuscript](paper/hawkes.pdf) · [LaTeX, figures and build instructions](paper/README.md) · [arXiv preprint](https://arxiv.org/abs/2510.21297) · [SSRN 4735365](https://ssrn.com/abstract=4735365)

The paper models positive and negative jumps with two Hawkes intensities. Each jump can excite both intensities. The authors estimate the historical dynamics from Bitcoin returns, then fit positive and negative jump risk premia to option prices. Their empirical study relates those premia to the BTC futures cost of carry and delta-hedged option returns.

## What can be reproduced here

| Part of the article | Evidence and code in StochVolModels |
|---|---|
| Model and pricing (Section 2) | The [Hawkes pricer](../../src/stochvolmodels/pricers/hawkes_jd_pricer.py) computes option prices. The [executable model guide](../../examples/docs/hawkes_jump_diffusion.py) checks stationarity, the martingale condition and the Poisson limit on generated inputs. Its smile and simulation cases illustrate the pricing method. |
| Historical estimation and option fitting (Section 3, Figures 2–4, Tables 1–2) | The [estimator](replication/hawkes_estimator.py), [chain loader](replication/chain_data.py), [single-chain calibration](replication/calibrate_chain.py) and [risk-premium calibration](replication/risk_premia_calibration.py) implement parts of the two-stage workflow. The dated parameter estimates and smiles require the original observations, sampling times and fit settings. |
| Risk-premium sensitivities (Section 4, Figures 5–6) | The [pricing and simulation experiments](replication/risk_premia_mc.py) explore implied volatilities and forward curves. The accepted figure files accompany the LaTeX source. |
| Market panels and regressions (Figure 1 and Section 5, Figures 7–9, Tables 3–5) | The article includes the reported results. Some co-author code is outside SVM. The manuscript points to [QuantLet/HJP](https://github.com/QuantLet/HJP/tree/main/codes) for exhibits including goodness of fit, the calibration diagnostic, the premium time series and cost-of-carry analysis. Rebuilding the PDF uses the supplied exhibits. |

## Build and run

Build the author manuscript with the [LaTeX and BibTeX commands](paper/README.md).
The supplied exhibits are sufficient for that build.

From the repository root, use your external Python environment.

```console
python -m pip install -e ".[research]" pyarrow
python examples/docs/hawkes_jump_diffusion.py
```

The example uses generated inputs. Historical estimation and calibration use
the original Tardis data. Set `RESOURCE_PATH` in the
[settings template](../../src/stochvolmodels/settings.yaml.example) and supply
`tardis/BTC_freq_H.feather` and `tardis/BTC_perp_freq_H.feather` there.
For ETH analysis provide the corresponding `ETH_` files. The hourly loader
uses the original intraday observations. The end-of-day chain loader requires
the corresponding OCA exact-08:00 UTC snapshot files at the same resource
location.

```console
python -m papers.jump_risk_premia_clustered_jumps_sifin_2027.replication.hawkes_estimator
python -m papers.jump_risk_premia_clustered_jumps_sifin_2027.replication.risk_premia_mc
```

Select the desired case through each module's `Locals` enum. For the dated
time-series regressions and article-specific calibration diagnostics, use the
original fit settings and [co-authors' QuantLet workflows](https://github.com/QuantLet/HJP/tree/main/codes).
