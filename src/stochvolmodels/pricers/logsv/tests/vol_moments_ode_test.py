"""Manual analytic-versus-Monte-Carlo checks for LogSV volatility moments.

Run this module explicitly with the visualization dependencies installed. Its large simulations
are intentionally excluded from the automated pytest suite.
"""

from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv.vol_moments_ode import (
    compute_expected_vol_t,
    compute_sqrt_qvar_t,
    compute_vol_moments_t,
    fit_model_vol_backbone_to_varswaps,
)
from stochvolmodels.pricers.logsv_pricer import LogSVPricer
from stochvolmodels.utils.funcs import set_seed


class LocalTests(Enum):
    """Available manual LogSV moment checks."""

    VOL_MOMENTS = 1
    EXPECTED_VOL = 2
    EXPECTED_QVAR = 3
    VOL_BACKBONE = 4


def run_local_test(local_test: LocalTests) -> None:
    """Run the selected manual LogSV moment check.

    Parameters
    ----------
    local_test : LocalTests
        Scenario to run.
    """
    logsv_pricer = LogSVPricer()
    n_terms = 4
    nb_path = 200000
    ttm = 1.0
    params = LogSvParams(
        sigma0=1.0,
        theta=1.0,
        kappa1=4.0,
        kappa2=4.0,
        beta=0.0,
        volvol=1.75,
    )
    params.assert_vol_moments_stability(n_terms=n_terms)
    set_seed(8)
    sigma_t, grid_t = logsv_pricer.simulate_vol_paths(
        ttm=ttm,
        params=params,
        nb_path=nb_path,
    )

    if local_test == LocalTests.VOL_MOMENTS:
        mcs = []
        for moment in np.arange(n_terms):
            if moment > 0:
                moment_values = np.power(sigma_t - params.theta, moment + 1)
            else:
                moment_values = sigma_t - params.theta
            mc_mean = np.mean(moment_values, axis=1)
            mcs.append(pd.Series(mc_mean, index=grid_t, name=f"MC m{moment + 1}"))

        analytic = compute_vol_moments_t(params=params, ttm=grid_t, n_terms=n_terms)
        analytic = pd.DataFrame(
            analytic,
            index=grid_t,
            columns=[f"m{moment + 1}" for moment in range(n_terms)],
        )
        frame = pd.concat([analytic, pd.concat(mcs, axis=1)], axis=1)
        print(frame)
        frame.plot()

    elif local_test == LocalTests.EXPECTED_VOL:
        mc_mean = np.mean(sigma_t, axis=1)
        mc_std = np.std(sigma_t, axis=1) / np.sqrt(nb_path)
        mc = pd.Series(mc_mean, index=grid_t, name="MC")
        mc_lower = pd.Series(mc_mean - 1.96 * mc_std, index=grid_t, name="MC-cd")
        mc_upper = pd.Series(mc_mean + 1.96 * mc_std, index=grid_t, name="MC+cd")
        analytic = pd.Series(
            compute_expected_vol_t(params=params, t=grid_t, n_terms=n_terms),
            index=grid_t,
            name="Analytic",
        )
        frame = pd.concat([analytic, mc, mc_lower, mc_upper], axis=1)
        print(frame)
        frame.plot()

    elif local_test == LocalTests.EXPECTED_QVAR:
        q_var = pd.DataFrame(np.square(sigma_t)).expanding(axis=0).mean().to_numpy()
        mc_mean = np.sqrt(np.mean(q_var, axis=1))
        mc_std = np.std(q_var, axis=1) / np.sqrt(nb_path)
        mc = pd.Series(mc_mean, index=grid_t, name="MC")
        mc_lower = pd.Series(mc_mean - 1.96 * mc_std, index=grid_t, name="MC-cd")
        mc_upper = pd.Series(mc_mean + 1.96 * mc_std, index=grid_t, name="MC+cd")
        analytic = pd.Series(
            compute_sqrt_qvar_t(params=params, t=grid_t, n_terms=n_terms),
            index=grid_t,
            name="Analytic",
        )
        frame = pd.concat([analytic, mc, mc_lower, mc_upper], axis=1)
        with sns.axes_style("darkgrid"):
            _, ax = plt.subplots(1, 1, figsize=(18, 10), tight_layout=True)
            sns.lineplot(data=analytic, dashes=False, ax=ax)
            ax.errorbar(
                x=frame.index[::5],
                y=mc_mean[::5],
                yerr=mc_std[::5],
                fmt="o",
                color="green",
                capsize=8,
            )

    elif local_test == LocalTests.VOL_BACKBONE:
        fit_model_vol_backbone_to_varswaps(
            log_sv_params=params,
            varswap_strikes=pd.Series([1.0, 1.0], index=[1.0 / 12.0, 2.0 / 12.0]),
            verbose=True,
        )

    plt.show()


if __name__ == "__main__":
    run_local_test(local_test=LocalTests.VOL_BACKBONE)
