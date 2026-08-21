"""Manual distribution and plotting checks for the analytic Student-t implementation.

Run this module explicitly with the research dependencies installed. These scenarios are not
part of the automated pytest suite.
"""

from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qis
from qis.plots.utils import get_n_colors

from stochvolmodels.fitters.tdist import cdf_tdist, cum_mean_tdist, pdf_tdist


class LocalTests(Enum):
    """Available manual analytic Student-t checks."""

    PLOT_PDF = 1
    PLOT_CDF = 2
    PLOT_CUM_X = 3
    PLOT_H = 4


def run_local_test(local_test: LocalTests) -> None:
    """Run the selected manual analytic Student-t check.

    Parameters
    ----------
    local_test : LocalTests
        Scenario to run.
    """
    x = np.linspace(-5.0, 5.0, 20000)
    dx = x[1] - x[0]
    ttm = 1.0
    mu_vols = {
        "mu=0.0, vol=0.2": (0.0, 0.2),
        "mu=0.2, vol=0.2": (0.2, 0.2),
        "mu=0.2, vol=0.4": (0.2, 0.4),
    }

    if local_test == LocalTests.PLOT_PDF:
        pdfs = {}
        for key, mu_vol in mu_vols.items():
            pdf = dx * pdf_tdist(x=x, mu=mu_vol[0], vol=mu_vol[1], nu=3.0, ttm=ttm)
            pdfs[key] = pd.Series(pdf, index=x)
            variance = np.sum(np.square(x) * pdf) - np.square(np.sum(x * pdf))
            print(
                f"{key}: sum={np.sum(pdf)}, mean={np.sum(x * pdf)}, "
                f"std={np.sqrt(variance)}"
            )
        qis.plot_line(df=pd.DataFrame.from_dict(pdfs, orient="columns"))

    elif local_test == LocalTests.PLOT_CDF:
        pdfs = {}
        cpdfs = {}
        for key, mu_vol in mu_vols.items():
            pdf = dx * pdf_tdist(x=x, mu=mu_vol[0], vol=mu_vol[1], nu=3.0, ttm=ttm)
            cpdf = cdf_tdist(x=x, mu=mu_vol[0], vol=mu_vol[1], nu=3.0, ttm=ttm)
            pdfs[f"{key}_pdf_sum"] = pd.Series(np.cumsum(pdf), index=x)
            cpdfs[f"{key}_cdf"] = pd.Series(cpdf, index=x)
        frame = pd.concat(
            [
                pd.DataFrame.from_dict(pdfs, orient="columns"),
                pd.DataFrame.from_dict(cpdfs, orient="columns"),
            ],
            axis=1,
        )
        colors = get_n_colors(n=len(mu_vols))
        qis.plot_line(df=frame, colors=2 * colors)

    elif local_test == LocalTests.PLOT_CUM_X:
        pdfs = {}
        cpdfs = {}
        for key, mu_vol in mu_vols.items():
            pdf = dx * pdf_tdist(x=x, mu=mu_vol[0], vol=mu_vol[1], nu=3.0, ttm=ttm)
            cpdf = cum_mean_tdist(x=x, mu=mu_vol[0], vol=mu_vol[1], nu=3.0, ttm=ttm)
            pdfs[f"{key}_h_pdf_sum"] = pd.Series(np.cumsum(x * pdf), index=x)
            cpdfs[f"{key}_t_h"] = pd.Series(cpdf, index=x)
        frame = pd.concat(
            [
                pd.DataFrame.from_dict(pdfs, orient="columns"),
                pd.DataFrame.from_dict(cpdfs, orient="columns"),
            ],
            axis=1,
        )
        colors = get_n_colors(n=len(mu_vols))
        qis.plot_line(df=frame, colors=2 * colors)

    elif local_test == LocalTests.PLOT_H:
        x = np.linspace(-10.0, 10.0, 2000)
        h = pd.Series(
            cum_mean_tdist(x=x, mu=0.5, vol=1.0, nu=3.0, ttm=1.0),
            index=x,
            name="h",
        )
        qis.plot_line(df=h, xlabel="x")

    plt.show()


if __name__ == "__main__":
    run_local_test(local_test=LocalTests.PLOT_CUM_X)
