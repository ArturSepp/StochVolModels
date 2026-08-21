"""Fit the approximate LogSV smile to a bundled OCA-generated option chain.

This example is credential-free and needs no market-data cache::

    python examples/calibration/run_logsv_smile_fitter.py --maturity 1m

The ready chain has the same :class:`stochvolmodels.OptionChain` shape returned by the OCA and
ThetaData adapters. Replace ``get_oca_simulated_chain_data`` with one of those loaders to fit a
provider-backed observation without changing the fitting code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stochvolmodels.data.sample_option_chains import get_oca_simulated_chain_data
from stochvolmodels.fitters import calc_logsv_ivols, fit_logsv_ivols


def fit_and_plot_smile(maturity: str) -> tuple[dict[str, float], float, plt.Figure]:
    """Fit and plot one maturity from the bundled OCA-generated chain.

    Parameters
    ----------
    maturity : str
        Maturity label prefix, either ``"1w"`` or ``"1m"``.

    Returns
    -------
    tuple[dict[str, float], float, matplotlib.figure.Figure]
        Fitted parameters, implied-volatility RMSE, and smile figure.
    """
    option_chain = get_oca_simulated_chain_data()
    matches = [
        idx for idx, slice_id in enumerate(option_chain.ids) if slice_id.startswith(maturity)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one maturity matching {maturity!r}, found {len(matches)}")
    idx = matches[0]

    forward = option_chain.forwards[idx]
    log_strikes = np.log(option_chain.strikes_ttms[idx] / forward)
    mid_vols = 0.5 * (option_chain.bid_ivs[idx] + option_chain.ask_ivs[idx])
    fit_params = fit_logsv_ivols(
        log_strikes=log_strikes,
        mid_vols=mid_vols,
        ttm=option_chain.ttms[idx],
    )
    fitted_vols = calc_logsv_ivols(log_strikes, **fit_params)
    rmse = float(np.sqrt(np.mean(np.square(fitted_vols - mid_vols))))

    grid = np.linspace(log_strikes.min(), log_strikes.max(), 201)
    figure, axis = plt.subplots(figsize=(9, 5), tight_layout=True)
    axis.scatter(log_strikes, 100.0 * mid_vols, label="OCA-generated mid IV")
    axis.plot(grid, 100.0 * calc_logsv_ivols(grid, **fit_params), label="Approximate LogSV")
    axis.set_title(f"Approximate LogSV smile — {option_chain.ids[idx]}")
    axis.set_xlabel("log(strike / forward)")
    axis.set_ylabel("Implied volatility (%)")
    axis.grid(alpha=0.3)
    axis.legend()
    return fit_params, rmse, figure


def main() -> None:
    """Parse CLI arguments and run the bundled-chain smile illustration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maturity", choices=("1w", "1m"), default="1m")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    fit_params, rmse, figure = fit_and_plot_smile(maturity=args.maturity)
    print(f"fitted_logsv={fit_params}")
    print(f"implied_vol_rmse={rmse:.8f}")

    if args.output_dir is None:
        plt.show()
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / f"oca_generated_logsv_smile_{args.maturity}.png"
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
        print(f"plot={output_path.resolve()}")


if __name__ == "__main__":
    main()
