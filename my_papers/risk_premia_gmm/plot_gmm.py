"""
illustrations of gmm vols
"""
import numpy as np
import pandas as pd
import qis as qis
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
from stochvolmodels import GmmParams, OptionChain, GmmPricer


def plot_gmm_pdfs(params: GmmParams,
                  option_chain0: OptionChain,
                  nstdev: float = 10.0,
                  titles: List[str] = None,
                  axs: List[plt.Subplot] = None
                  ) -> plt.Figure:
    """
    plot gmm pdf and model fit
    """
    stdev = nstdev * params.get_get_avg_vol() * np.sqrt(params.ttm)
    x = np.linspace(-stdev, stdev, 3000)
    state_pdfs, agg_pdf = params.compute_state_pdfs(x=x)

    columns = []
    for idx in range(len(params.gmm_weights)):
        columns.append(
            f"state-{idx + 1}: mean={params.gmm_mus[idx]:0.2f}, vol={params.gmm_vols[idx]:0.2f}, weight={params.gmm_weights[idx]:0.2f}")

    state_pdfs = pd.DataFrame(state_pdfs, index=x, columns=columns)
    agg_pdf = pd.Series(agg_pdf, index=x, name='Aggregate PDF')
    df = pd.concat([agg_pdf, state_pdfs], axis=1)

    kwargs = dict(fontsize=14, framealpha=0.80)

    if axs is None:
        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(1, 2, figsize=(16, 4.5))
    else:
        fig = None

    qis.plot_line(df=df,
                  linestyles=['--'] + ['-'] * len(params.gmm_weights),
                  y_limits=(0.0, None),
                  xvar_format='{:,.2f}',
                  xlabel='log-price',
                  first_color_fixed=True,
                  ax=axs[0],
                  **kwargs)
    axs[0].get_lines()[0].set_linewidth(4.0)
    axs[0].get_legend().get_lines()[0].set_linewidth(4.0)
    qis.set_title(ax=axs[0], title='(A) State PDF and Aggregate Risk-Neutral PDF', **kwargs)

    gmm_pricer = GmmPricer()
    gmm_pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain0, params=params,
                                           is_log_strike_xaxis=True,
                                           axs=[axs[1]],
                                           **kwargs)
    return fig
