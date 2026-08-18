"""Illustrate zero-DTE option decay under flat and time-varying volatility skews."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import stochvolmodels as sv


def repeat_by_columns(values: np.ndarray, n: int) -> np.ndarray:
    """Repeat a vector into columns."""
    return values.repeat(n).reshape((-1, n))


def repeat_by_rows(values: np.ndarray, n: int) -> np.ndarray:
    """Repeat a vector into rows."""
    return values.repeat(n).reshape((-1, n)).T


def compute_zero_dte_panels() -> tuple[pd.DataFrame, ...]:
    """Compute premium and one-minute decay panels for a 6.5-hour session."""
    annualization_factor = 60 * 24 * 260
    minutes = np.linspace(0, 6.5 * 60, int(6.5 * 60) + 1)[::-1]
    ttms = minutes / annualization_factor

    forward = 100.0
    strikes = np.array([forward * 0.999, forward * 1.001])
    optiontypes = np.array(["P", "C"])

    flat_vols = 0.2 * np.ones((len(ttms), len(strikes)))
    skew = -30.0
    # At expiry volatility no longer affects intrinsic value. Use one minute in the
    # power law to avoid evaluating an otherwise irrelevant infinite expiry skew.
    effective_ttms = np.maximum(ttms, 1.0 / annualization_factor)
    skew_power = np.power(effective_ttms / ttms[0], -0.30)
    log_moneyness = np.log(strikes / forward)
    skew_vols = (
        flat_vols
        + skew
        * repeat_by_rows(log_moneyness, n=len(ttms))
        * repeat_by_columns(skew_power, n=len(strikes))
    )

    flat_premiums = np.zeros((len(ttms), len(strikes)))
    skew_premiums = np.zeros((len(ttms), len(strikes)))
    for idx, ttm in enumerate(ttms):
        flat_premiums[idx, :] = sv.compute_bsm_vanilla_slice_prices(
            ttm=ttm,
            forward=forward,
            strikes=strikes,
            vols=flat_vols[idx, :],
            optiontypes=optiontypes,
        )
        skew_premiums[idx, :] = sv.compute_bsm_vanilla_slice_prices(
            ttm=ttm,
            forward=forward,
            strikes=strikes,
            vols=skew_vols[idx, :],
            optiontypes=optiontypes,
        )

    flat_labels = [f"flat vol, strike={strike:0.2f}" for strike in strikes]
    skew_labels = [f"skew vol, strike={strike:0.2f}" for strike in strikes]
    flat_premium_frame = pd.DataFrame(
        flat_premiums[:-1], columns=flat_labels, index=minutes[:-1]
    )
    skew_premium_frame = pd.DataFrame(
        skew_premiums[:-1], columns=skew_labels, index=minutes[:-1]
    )
    flat_theta_frame = pd.DataFrame(
        flat_premiums[1:, :] - flat_premiums[:-1, :],
        columns=flat_labels,
        index=minutes[:-1],
    )
    skew_theta_frame = pd.DataFrame(
        skew_premiums[1:, :] - skew_premiums[:-1, :],
        columns=skew_labels,
        index=minutes[:-1],
    )

    out_premiums = pd.concat(
        [flat_premium_frame.iloc[:, 0], skew_premium_frame.iloc[:, 0]], axis=1
    )
    in_premiums = pd.concat(
        [flat_premium_frame.iloc[:, -1], skew_premium_frame.iloc[:, -1]], axis=1
    )
    out_thetas = pd.concat(
        [flat_theta_frame.iloc[:, 0], skew_theta_frame.iloc[:, 0]], axis=1
    )
    in_thetas = pd.concat(
        [flat_theta_frame.iloc[:, -1], skew_theta_frame.iloc[:, -1]], axis=1
    )
    return out_premiums, in_premiums, out_thetas, in_thetas


def plot_zero_dte_theta() -> tuple[plt.Figure, plt.Figure]:
    """Plot zero-DTE premiums and one-minute option decay."""
    out_premiums, in_premiums, out_thetas, in_thetas = compute_zero_dte_panels()

    with sns.axes_style("darkgrid"):
        premium_figure, axes = plt.subplots(2, 1, figsize=(8, 8), tight_layout=True)
        for axis, data in zip(axes, (out_premiums, in_premiums), strict=True):
            sns.lineplot(data=data, dashes=False, ax=axis)
            axis.invert_xaxis()
            axis.set_xlabel("minutes to end of trading session")

        theta_figure, axes = plt.subplots(2, 1, figsize=(8, 8), tight_layout=True)
        for axis, data in zip(axes, (out_thetas, in_thetas), strict=True):
            sns.lineplot(data=data, dashes=False, ax=axis)
            axis.invert_xaxis()
            axis.set_xlabel("minutes to end of trading session")

    return premium_figure, theta_figure


if __name__ == "__main__":
    plot_zero_dte_theta()
    plt.show()
