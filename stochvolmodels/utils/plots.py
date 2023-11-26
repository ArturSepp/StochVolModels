"""
utility plotting functions
"""

import datetime as dt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
from os.path import join
from typing import Union, Dict, Tuple, List, Optional, Literal


DATE_TIME_FORMAT = '%Y%m%d_%H%M'
DATE_FORMAT = '%d%b%Y'
DATE_FORMAT_TIME = '%d%b%Y %H:%M '

FIGSIZE = (18, 10)  # global display size


# talk
def set_fig_props(size: int = 14):
    sns.set_context("talk", rc={'font.size': size, 'axes.titlesize': size, 'axes.labelsize': size, 'legend.fontsize': size})

    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def create_dummy_line(**kwargs):
    return Line2D([], [], **kwargs)


def get_n_sns_colors(n: int) -> List[str]:
    return sns.color_palette(None, n)


def fig_to_pdf(fig: plt.Figure,
               file_name: str,
               local_path: str,
               orientation: Literal['portrait', 'landscape'] = 'portrait'
               ) -> str:
    file_path = join(local_path, f"{file_name}.pdf")
    with PdfPages(file_path) as pdf:
        pdf.savefig(fig, orientation=orientation)
    print(f"created PDF: {file_path}")
    return file_path


def fig_list_to_pdf(figs: List[plt.Figure],
                    file_name: str,
                    local_path: str,
                    is_add_current_date: bool = False,
                    orientation: Literal['portrait', 'landscape'] = 'portrait'
                    ) -> str:
    """
    create PDF of list of plf figures
    """
    if is_add_current_date:
        file_name = f"{file_name}_{dt.datetime.now().strftime(DATE_TIME_FORMAT)}"
    file_path = join(local_path, f"{file_name}.pdf")

    with PdfPages(file_path) as pdf:
        for fig in figs:
            pdf.savefig(fig, orientation=orientation)
    print(f"created PDF doc: {file_path}")
    return file_path


def save_fig(fig: plt.Figure,
             file_name: str,
             local_path: Optional[str] = None,
             dpi: int = 300,
             extension: str = 'PNG',
             **kwargs
             ) -> str:
    """
    save matplotlib figure
    """
    file_path = join(local_path or '..//', f"{file_name}.{extension}")
    fig.savefig(file_path, dpi=dpi)  # , bbox_inches=bbox_inches
    return file_path


def save_figs(figs: Dict[str, plt.Figure],
              local_path: Optional[str] = None,
              dpi: int = 300,
              extension: str = 'PNG',
              **kwargs
              ) -> None:
    """
    save matplotlib figures dict
    """
    for key, fig in figs.items():
        file_path = save_fig(fig=fig,
                             file_name=key,
                             local_path=local_path,
                             dpi=dpi,
                             extension=extension,
                             **kwargs)
        print(file_path)


def vol_slice_fit(bid_vol: pd.Series,
                  ask_vol: pd.Series,
                  model_vols: Union[pd.Series, pd.DataFrame],
                  title: str = None,
                  strike_name: str = 'strike',
                  bid_name: str = 'bid',
                  ask_name: str = 'ask',
                  mid_name: str = 'mid',
                  model_color: str = 'black',
                  bid_color: str = 'red',
                  ask_color: str = 'green',
                  mid_color: str = 'lightslategrey',
                  is_add_mids: bool = False,
                  atm_points: Dict[str, Tuple[float, float]] = None,
                  yvar_format: str = '{:.0%}',
                  xvar_format: Optional[str] = '{:0,.0f}',
                  fontsize: int = 12,
                  ylabel: str = 'Implied vols',
                  x_rotation: int = 0,
                  ax: plt.Subplot = None,
                  **kwargs
                  ) -> plt.Figure:

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = None

    if isinstance(model_vols, pd.Series):  # optimise for frame
        model_vols = model_vols.to_frame()

    # for legend
    lines = []

    # add fitted vols line
    if len(model_vols.columns) == 1:
        palette = [model_color]
    else:
        palette = sns.husl_palette(len(model_vols.columns), h=.5)
    sns.lineplot(data=model_vols, palette=palette, dashes=False, ax=ax)
    for legend, color in zip(model_vols.columns, palette):
        lines.append((legend, {'color': color}))

    # add mids with error bars
    if is_add_mids:
        aligned_vols = pd.concat([bid_vol, ask_vol], axis=1)
        mid = np.mean(aligned_vols.to_numpy(), axis=1)
        lower_error = aligned_vols.iloc[:, 0].to_numpy()
        upper_error = aligned_vols.iloc[:, 1].to_numpy()
        error = sns.utils.ci_to_errsize((lower_error, upper_error), mid)
        ax.errorbar(x=aligned_vols.index.to_numpy(), y=mid, yerr=error, fmt='o', color=mid_color)
        lines.append((mid_name, {'color': mid_color, 'linestyle': '', 'marker': 'o'}))

    # add bid ask scatter
    legends_data = {bid_name: bid_color, ask_name: ask_color}
    for vol, legend in zip([bid_vol, ask_vol], legends_data.keys()):
        vol.index.name = strike_name
        vol.name = legend
        data = vol.to_frame().reset_index()
        sns.scatterplot(x=strike_name, y=legend, data=data, color=legends_data[legend],
                        edgecolor=None,
                        facecolor=None,
                        s=40,
                        linewidth=3,
                        marker='_',
                        ax=ax)
        lines.append((legend, {'color': legends_data[legend], 'linestyle': '', 'marker': '_'}))

    # atm points
    if atm_points is not None:
        for key, (x, y) in atm_points.items():
            ax.scatter(x, y, marker='*', color='navy', s=40, linewidth=5)
            lines.append(('ATM', {'color': 'navy', 'linestyle': '', 'marker': '*'}))

    ax.legend([create_dummy_line(**l[1]) for l in lines],  # Line handles
              [l[0] for l in lines],  # Line titles
              loc='upper center',
              framealpha=0,
              fontsize=fontsize)
    get_legend_colors(ax)

    if x_rotation != 0:
        [tick.set_rotation(x_rotation) for tick in ax.get_xticklabels()]

    # ticks
    if xvar_format is not None:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda z, _: xvar_format.format(z)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda z, _: yvar_format.format(z)))

    ax.set_ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize, color='darkblue')

    return fig


def plot_model_risk_var(risk_var: Union[pd.Series, pd.DataFrame],
                        xvar_format: str = '{:.2f}',
                        yvar_format: str = '{:.2f}',
                        x_rotation: int = 0,
                        xlabel: str = 'log-return',
                        ylabel: str = 'probability',
                        title: str = None,
                        ax: plt.Subplot = None
                        ) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = None

    if isinstance(risk_var, pd.Series):  # optimise for frame
        risk_var = risk_var.to_frame()

    if len(risk_var.columns) == 1:
        palette = ['black']
    else:
        palette = None

    sns.lineplot(data=risk_var, palette=palette, dashes=False, ax=ax)

    if len(risk_var.columns) == 1:
        ax.legend().set_visible(False)
        # ax.axes.get_yaxis().set_visible(False)
    else:
        ax.legend(loc='upper left', framealpha=0)
        get_legend_colors(ax)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda z, _: xvar_format.format(z)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda z, _: yvar_format.format(z)))

    if x_rotation != 0:
        [tick.set_rotation(x_rotation) for tick in ax.get_xticklabels()]

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    return fig


def model_vols_ts(model_vols: Union[pd.Series, pd.DataFrame],
                  is_delta_space: bool = False,
                  xvar_format: str = '{:0,.0f}',
                  yvar_format: str = '{:.0%}',
                  x_rotation: int = 0,
                  xlabel: str = 'strike',
                  n_tickwindow: int = None,
                  marker: str = None,
                  title: str = None,
                  legend_loc: str = 'upper center',
                  ax: plt.Subplot = None,
                  **kwargs
                  ) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = None

    if is_delta_space:
        model_vols.index = map_deltas_to_str(bsm_deltas=model_vols.index.to_numpy())

    sns.lineplot(data=model_vols, dashes=False, marker=marker, ax=ax)

    if is_delta_space:
        yvar_format = '{:,.2f}'

    ax.legend(loc=legend_loc, framealpha=0)
    get_legend_colors(ax)

    if not isinstance(model_vols.index.dtype, object):  # do not apply for str
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda z, _: xvar_format.format(z)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda z, _: yvar_format.format(z)))

    if n_tickwindow is not None:
        if n_tickwindow == 5:
            ax.set_xticks(ax.get_xticks()[::n_tickwindow])
        else:
            this = ax.get_xticks()
            ax.set_xticks([this[0]] + this[1:-1][::n_tickwindow]+[this[-1]])

    if x_rotation != 0:
        [tick.set_rotation(x_rotation) for tick in ax.get_xticklabels()]

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(xlabel)

    return fig


def model_param_ts(param_ts: Union[pd.Series, pd.DataFrame],
                   yvar_format: str = '{:.2f}',
                   x_rotation: int = 0,
                   title: str = None,
                   markers: bool = True,
                   legend_loc: str = 'upper center',
                   ax: plt.Subplot = None
                   ) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = None

    sns.lineplot(data=param_ts, dashes=True, markers=markers, ax=ax)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda z, _: yvar_format.format(z)))

    ax.legend(loc=legend_loc, framealpha=0)
    get_legend_colors(ax)

    if x_rotation != 0:
        [tick.set_rotation(x_rotation) for tick in ax.get_xticklabels()]
    if isinstance(param_ts, pd.Series):
        ax.set_title(param_ts.name, color='blue')
    elif title is not None:
        ax.set_title(title, color='blue')
    return fig


def get_legend_colors(ax: plt.Subplot,
                      text_weight: str = None,
                      colors: List[str] = None,
                      fontsize: int = 12,
                      **kwargs
                      ) -> None:

    leg = ax.get_legend()
    if colors is None:
        colors = [line.get_color() for line in leg.get_lines()]

    for text, color in zip(leg.get_texts(), colors):
        text.set_color(color)
        text.set_size(fontsize)
        if text_weight is not None:
            text.set_weight(text_weight)


def set_y_limits(ax: plt.Subplot,
                 y_limits: Tuple[Union[float, None], Union[float, None]]
                 ) -> None:

    ymin, ymax = ax.get_ylim()
    if y_limits[0] is not None:
        ymin = y_limits[0]
    if y_limits[1] is not None:
        ymax = y_limits[1]
    ax.set_ylim([ymin, ymax])


def map_deltas_to_str(bsm_deltas: np.ndarray) -> List[str]:
    slice_index = []
    index_str = [f"{x:0.2f}" for x in bsm_deltas]
    for idx, x in enumerate(bsm_deltas):
        x_str = index_str[idx]
        if idx > 0:
            if x_str == index_str[idx - 1]:
                if x < 0.0:  # decrease previous delta
                    slice_index[idx - 1] = f"{bsm_deltas[idx - 1]:0.3f}"
                else:
                    x_str = f"{x:0.3f}"
        slice_index.append(x_str)
    return slice_index


def set_subplot_border(fig: plt.Figure,
                       n_ax_col: int = 1,
                       n_ax_rows: int = 1
                       ) -> None:

    n_ax1 = n_ax_rows
    n_ax2 = n_ax_col

    rects = []
    height = 1.0 / n_ax1
    for r in range(n_ax1):
        rects.append(plt.Rectangle((0.0, r*height), 1.0, height,  # (lower-left corner), width, height
                                   fill=False,
                                   color='#00284A',
                                   lw=1,
                                   zorder=1000,
                                   transform=fig.transFigure, figure=fig))
    width = 1.0 / n_ax2
    for r in range(n_ax2):
        rects.append(plt.Rectangle((r*width, 0), width, 1.0,  # (lower-left corner), width, height
                                   fill=False,
                                   color='#00284A',
                                   lw=1,
                                   zorder=1000,
                                   transform=fig.transFigure, figure=fig))
    fig.patches.extend(rects)


def align_x_limits_axs(axs: List[plt.Subplot],
                       is_invisible_xs: bool = False
                       ) -> None:
    xmins = []
    xmaxs = []
    for ax in axs:
        xmin, xmax = ax.get_xlim()
        xmins.append(xmin)
        xmaxs.append(xmax)
    xmin = np.min(xmins)
    xmax = np.max(xmaxs)
    for ax in axs:
        ax.set_xlim([xmin, xmax])

    if is_invisible_xs:
        for idx, ax in enumerate(axs):
            if idx > 0:
                ax.axes.get_xaxis().set_visible(False)


def align_y_limits_axs(axs: List[plt.Subplot],
                       is_invisible_ys: bool = False
                       ) -> None:
    ymins = []
    ymaxs = []
    for ax in axs:
        ymin, ymax = ax.get_ylim()
        ymins.append(ymin)
        ymaxs.append(ymax)
    ymin = np.min(ymins)
    ymax = np.max(ymaxs)
    for ax in axs:
        ax.set_ylim([ymin, ymax])

    if is_invisible_ys:
        for idx, ax in enumerate(axs):
            if idx > 0:
                ax.axes.get_yaxis().set_visible(False)