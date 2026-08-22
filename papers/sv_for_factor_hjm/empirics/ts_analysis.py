
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qis
from typing import Dict
from enum import Enum
from stochvolmodels import local_path as lp

from . import ust_rates_data as loader
from .fit_nelson_siegel import fit_timeseries_ns_factors, plot_ns_rates


def plot_factors(xs: pd.DataFrame,  # factors xs at weekly freq
                 span: int = 52
                 ) -> plt.Figure:

    delta_xs = xs.diff(1).dropna()
    ewma_corr = qis.compute_ewm_corr_df(df=delta_xs, span=span)
    print(ewma_corr)

    figs = {}
    # ts of factors
    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        qis.set_suptitle(fig, title='Time series of factors')
        figs['time_series'] = fig
        kwargs = dict(x_date_freq='YE',
                      legend_stats=qis.LegendStats.AVG_STD_LAST,
                      var_format='{:,.2%}',
                      framealpha=0.75)
        qis.plot_time_series(df=xs,
                             ax=axs[0],
                             **kwargs)
        qis.plot_time_series(df=ewma_corr,
                             ax=axs[1],
                             **kwargs)
    return fig


def plot_with_move(ys: pd.DataFrame,
                   resid: pd.DataFrame,
                   move: pd.Series,
                   ewm_span: int = 52,
                   ) -> Dict[str, plt.Figure]:
    from statsmodels.formula.api import ols

    delta_ys = ys.diff(1).dropna()
    # with move scatter
    move_ys = pd.concat([move, ys], axis=1).dropna()
    move_ys_chages = move_ys.diff(1).dropna()

    figs = {}
    # ts of factors
    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        qis.set_suptitle(fig, title='Time series of factors')
        figs['time_series'] = fig
        kwargs = dict(x_date_freq='YE',
                      legend_stats=qis.LegendStats.AVG_STD_LAST,
                      var_format='{:,.2%}',
                      framealpha=0.75)
        qis.plot_time_series(df=ys,
                             ax=axs[0],
                             **kwargs)
        qis.plot_time_series(df=np.sqrt(resid),
                             ax=axs[1],
                             **kwargs)

    # correlation of factors
    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        qis.set_suptitle(fig, title='Time series of correlations of factors')
        figs['time_series_corr'] = fig
        kwargs = dict(x_date_freq='YE',
                      legend_stats=qis.LegendStats.AVG_STD_LAST,
                      var_format='{:,.2%}',
                      span=ewm_span,
                      framealpha=0.75)
        qis.plot_returns_corr_matrix_time_series(prices=ys,
                                                 return_type=qis.ReturnTypes.LEVEL,
                                                 title='Level',
                                                 ax=axs[0],
                                                 **kwargs)
        qis.plot_returns_corr_matrix_time_series(prices=ys,
                                                 return_type=qis.ReturnTypes.DIFFERENCE,
                                                 title='Difference',
                                                 ax=axs[1],
                                                 **kwargs)

    # volatility
    vol_ewm = qis.compute_ewm_vol(data=delta_ys, span=ewm_span, annualize=True)
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        qis.set_suptitle(fig, title='Time series of vol')
        figs['vol_ts'] = fig
        kwargs = dict(x_date_freq='YE',
                      legend_stats=qis.LegendStats.AVG_STD_LAST,
                      var_format='{:,.2%}',
                      framealpha=0.75)
        qis.plot_time_series(df=vol_ewm,
                             ax=ax,
                             **kwargs)

    # betas to move
    ewm_linear_model = qis.EwmLinearModel(x=move_ys_chages[ys.columns],
                                          y=move_ys_chages[[move.name]])
    ewm_linear_model.fit(span=ewm_span, is_x_correlated=True)

    move_betas = ewm_linear_model.get_asset_factor_betas().dropna()
    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        qis.set_suptitle(fig, title='Time series of move betas')
        figs['move_betas'] = fig
        kwargs = dict(x_date_freq='YE',
                      legend_stats=qis.LegendStats.AVG_STD_LAST,
                      var_format='{:,.2f}',
                      framealpha=0.75)
        qis.plot_time_series(df=move_betas,
                             title='EMWA',
                             ax=axs[0],
                             **kwargs)
        qis.plot_time_series(df=move_betas.rolling(ewm_span).mean(),
                             title='EMWA rolling',
                             ax=axs[1],
                             **kwargs)
    # attribution
    attribution = ewm_linear_model.get_asset_factor_attribution().dropna()
    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        qis.set_suptitle(fig, title='Time series of attribution')
        figs['move_attribution'] = fig
        kwargs = dict(x_date_freq='YE',
                      legend_stats=qis.LegendStats.AVG_STD_LAST,
                      var_format='{:,.2f}',
                      framealpha=0.75)
        qis.plot_time_series(df=attribution.rolling(12).sum(),
                             title='3m',
                             ax=axs[0],
                             **kwargs)
        qis.plot_time_series(df=attribution.rolling(52).sum(),
                             title='1y',
                             ax=axs[1],
                             **kwargs)


    # pdf of factors and changes
    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(1, 2, figsize=(10, 8))
        qis.set_suptitle(fig, title='PDF of factors and changes')
        figs['pdf'] = fig
        kwargs = dict(xvar_format='{:,.2%}',
                      desc_table_type=qis.DescTableType.WITH_NORMAL_PVAL)
        qis.plot_histogram(df=ys,
                           title='Factors',
                           ax=axs[0],
                           **kwargs)
        qis.plot_histogram(df=delta_ys,
                           title='Factor Changes',
                           ax=axs[1],
                           **kwargs)

    # hist plot 2d
    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(2, 3, figsize=(10, 8), tight_layout=True)
        qis.set_suptitle(fig, title='Histplot of factors and changes')
        figs['hist2d'] = fig
        kwargs = dict(fontsize=6, linewidth=0.5, weight='normal',
                      xvar_format='{:,.2%}', yvar_format='{:,.2%}')
        qis.plot_histplot2d(df=ys[['y1', 'y2']], ax=axs[0, 0], **kwargs)
        qis.plot_histplot2d(df=ys[['y1', 'y3']], ax=axs[0, 1], **kwargs)
        qis.plot_histplot2d(df=ys[['y2', 'y3']], ax=axs[0, 2], **kwargs)

        qis.plot_histplot2d(df=delta_ys[['y1', 'y2']], ax=axs[1, 0], **kwargs)
        qis.plot_histplot2d(df=delta_ys[['y1', 'y3']], ax=axs[1, 1], **kwargs)
        qis.plot_histplot2d(df=delta_ys[['y2', 'y3']], ax=axs[1, 2], **kwargs)

    # with move scatter

    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(1, 3, figsize=(10, 7))
        figs['scatter'] = fig
        for idx, factor in enumerate(ys.columns):
            df_chages1 = move_ys_chages[[factor, move.name]]
            qis.plot_scatter(df=df_chages1,
                             x=factor,
                             full_sample_order=1,
                             title=f"Move vs {factor}",
                             ax=axs[idx])

    model = ols(f"x ~ y1 + y2 + y3", move_ys_chages.rename({move.name: 'x'}, axis=1)).fit()
    print(model.summary())

    return figs


def plot_fit(rates: pd.DataFrame, lambdat: float, freq: str = 'ME'):
    from tqdm import tqdm

    ys, resid = fit_timeseries_ns_factors(rates=rates, lambdat=lambdat, freq=freq)
    figs = []
    for date in tqdm(ys.index):
        if date in rates.index:
            rates_t = rates.loc[date, :]
            ys_t = ys.loc[date, :].to_numpy()
            fig = plot_ns_rates(rates=rates_t, y1=ys_t[0], y2=ys_t[1], y3=ys_t[2], lambdat=lambdat)
            qis.set_suptitle(fig, title=date.strftime('%d-%b-%y'))
            figs.append(fig)
    qis.save_figs_to_pdf(
        figs,
        file_name=f"NelsonSiegel fit {freq}",
        local_path=lp.get_output_path(),
    )
    plt.close('all')


class UnitTests(Enum):
    PLOT_FACTORS = 1
    MOVE_FACTOR_CORR = 2
    PLOT_FIT = 3


def run_unit_test(unit_test: UnitTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    rates = loader.load_ust_rates()

    if unit_test == UnitTests.PLOT_FACTORS:
        ys, resid = fit_timeseries_ns_factors(rates=rates, freq='W-WED', lambdat=12.0*0.0609)
        plot_factors(xs=ys)

    elif unit_test == UnitTests.MOVE_FACTOR_CORR:
        ys, resid = fit_timeseries_ns_factors(rates=rates, freq='W-WED', lambdat=12.0*0.0609)
        move = loader.fetch_move(is_update=False)

        figs = plot_with_move(ys=ys, resid=resid, move=move)
        plt.close('all')
        qis.save_figs_to_pdf(
            figs,
            file_name='nelson_siegel_analysis',
            local_path=lp.get_output_path(),
        )

    elif unit_test == UnitTests.PLOT_FIT:
        plot_fit(rates=rates, lambdat=12.0*0.0609)

    plt.show()


if __name__ == '__main__':
    run_unit_test(unit_test=UnitTests.PLOT_FACTORS)
