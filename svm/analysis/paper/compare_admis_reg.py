"""
figures 2 and 3 in the paper
plot admissible regions for model parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

# stoch_vol_models
import svm.utils.plots as plot


def lognormal_combined(vartheta_min=0.5,
                       vartheta_max=3.0,
                       beta_min=-2.5,
                       beta_max=2.5,
                       kappa2s=[3.0, 0.0]):
    hatch1 = '\\\\\\\\'
    hatch2 = '////'
    vartheta = np.linspace(vartheta_min, vartheta_max, 100)
    nb_kappas = len(kappa2s)
    moment = 2
    num = 97

    fig, ax = plt.subplots(1, nb_kappas, figsize=(4 * nb_kappas, 3), tight_layout=True)
    for idx, kappa2 in enumerate(kappa2s):
        # plot for bounds ensuring martingale property under spot and inverse spot
        beta_spot_meas = np.maximum(kappa2, beta_min)
        beta_spot_meas = np.ones_like(vartheta) * beta_spot_meas
        beta_inv_meas = np.maximum(0.5 * kappa2, beta_min)
        beta_inv_meas = np.ones_like(vartheta) * beta_inv_meas
        outline, = ax[idx].plot(vartheta, beta_spot_meas, color='black', linewidth=0.8)
        ax[idx].fill_between(vartheta, beta_min, beta_spot_meas,
                                edgecolor='black', hatch=hatch1, label='MMA', facecolor='none')
        outline, = ax[idx].plot(vartheta, beta_inv_meas, color='black', linewidth=0.8)
        ax[idx].fill_between(vartheta, beta_min, beta_inv_meas,
                                edgecolor='grey', hatch=hatch2, label='Inverse', facecolor='none')
        ax[idx].set_ylim(beta_min, beta_max, auto=True)
        ax[idx].set_title(f"({chr(num).upper()}): $\kappa_2={kappa2}$")
        num = num + 1
        # # Right plot for bounds ensuring existence of second moment under spot and inverse spot
        # m = moment
        # beta_spot_meas = np.maximum((kappa2 - vartheta * np.sqrt(m ** 2 - m)) / m, beta_min)
        # beta_inv_meas = np.maximum((kappa2 - vartheta * np.sqrt(m ** 2 - m)) / (m + 1.0), beta_min)
        # outline, = ax[idx, 1].plot(vartheta, beta_spot_meas, color='black', linewidth=0.8)
        # ax[idx, 1].fill_between(vartheta, beta_min, beta_spot_meas,
        #                         edgecolor='black', hatch=hatch1, label='Spot', facecolor='none')
        # outline, = ax[idx, 1].plot(vartheta, beta_inv_meas, color='black', linewidth=0.8)
        # ax[idx, 1].fill_between(vartheta, beta_min, beta_inv_meas,
        #                         edgecolor='grey', hatch=hatch2, label='Inverse', facecolor='none')
        # ax[idx, 1].set_ylim(beta_min, beta_max, auto=True)
        # ax[idx, 1].set_title(f"({chr(num).upper()}): $\kappa_2={kappa2}$")
        # num = num + 1
    for i in range(nb_kappas):
        # for j in range(2):
        ax[i].legend()
        ax[i].set(xlabel=r"$\vartheta$", ylabel=r"$\beta$")

    plot.save_fig(fig=fig, local_path='../../../draft/figures//',
                  file_name='logsv_regions')


def heston_exp_ou_combined(vartheta_min=0.5,
                           vartheta_max=3.0,
                           rho_min=-1.0,
                           rho_max=1.0,
                           kappa=1,
                           theta=1):
    hatch1 = '\\\\\\\\'
    hatch2 = '////'
    vartheta = np.linspace(vartheta_min, vartheta_max, 100)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3), tight_layout=True)
    # Left plot for bounds ensuring existence of second moment under spot and inverse spot
    kappa_adj_inv = np.maximum(kappa / vartheta, rho_min)
    outline, = ax[0].plot(vartheta, kappa_adj_inv, color='black', linewidth=0.8)
    ax[0].fill_between(vartheta, rho_min, kappa_adj_inv,
                       edgecolor='black', hatch=hatch1, label=r"$\kappa>\rho\vartheta$",
                       facecolor='none')
    critical_val = np.sqrt(2.0 * kappa * theta)
    ax[0].axvspan(vartheta_min, critical_val, edgecolor='black', hatch=hatch2, label='Feller',
                  facecolor='none')
    ax[0].set_ylim(rho_min, rho_max, auto=True)
    ax[0].legend()
    ax[0].set(xlabel=r"$\vartheta$", ylabel=r"$\rho$")
    ax[0].set_title(f"(A) Heston model")
    # Right plot for bounds ensuring existence of second moment under spot and inverse spot
    kappa2 = 0
    rho_spot_meas = np.maximum(kappa2 / vartheta, rho_min)
    rho_inv_meas = np.maximum(0.5 * kappa2 / vartheta, rho_min)
    outline, = ax[1].plot(vartheta, rho_spot_meas, color='black', linewidth=0.8)
    ax[1].fill_between(vartheta, rho_min, rho_spot_meas,
                       edgecolor='black', hatch=hatch1, label='MMA',
                       facecolor='none')
    outline, = ax[1].plot(vartheta, rho_inv_meas, color='black', linewidth=0.8)
    ax[1].fill_between(vartheta, rho_min, rho_inv_meas,
                       edgecolor='grey', hatch=hatch2, label='Inverse',
                       facecolor='none')
    ax[1].set_ylim(rho_min, rho_max, auto=True)
    ax[1].legend()
    ax[1].set(xlabel=r"$\vartheta$", ylabel=r"$\rho$")
    ax[1].set_title(f"(B) Exp-OU model")

    plot.save_fig(fig=fig, local_path='../../../draft/figures//',
                  file_name='heston_exp_ou_combined')


class UnitTests(Enum):
    LOGNORMAL_SV_COMBINED = 1
    HESTON_EXP_OU_COMBINED = 2


def run_unit_test(unit_test: UnitTests):
    if unit_test == UnitTests.LOGNORMAL_SV_COMBINED:
        lognormal_combined(vartheta_min=0.5,
                           vartheta_max=3.0,
                           kappa2s=[0.0, 1.0])

    elif unit_test == UnitTests.HESTON_EXP_OU_COMBINED:
        heston_exp_ou_combined(vartheta_min=0.5,
                               vartheta_max=3.0)

    plt.show()


if __name__ == '__main__':
    unit_test = UnitTests.LOGNORMAL_SV_COMBINED
    run_unit_test(unit_test=unit_test)
