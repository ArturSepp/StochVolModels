"""Fixed IJTAF cases shared by the historical panels and their PDE overlays."""

from stochvolmodels import LogSvParams


FIGURE_SEED = 37
FIGURE3_TTM = 2.0
FIGURE6_TTM = 1.0 / 12.0


def figure3_cases():
    """Return fresh parameters in the paper's seeded simulation and legend order."""
    return {
        rf"$(\kappa_{{1}}=4, \kappa_{{2}}={kappa2}), \sigma_{{0}}={sigma0}$": LogSvParams(
            sigma0=sigma0, theta=1.0, kappa1=4.0, kappa2=kappa2, beta=0.0, volvol=1.5
        )
        for sigma0 in (1.5, 0.5)
        for kappa2 in (0, 4, 8)
    }


def figure6_params():
    """The one-month BTC illustration uses 60% of the fitted volatility of volatility."""
    return LogSvParams(
        sigma0=0.4083, theta=0.3789, kappa1=2.21, kappa2=2.18, beta=0.5010, volvol=0.6 * 3.0633
    )
