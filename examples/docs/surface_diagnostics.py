"""Offline canonical examples for fit and surface diagnostics."""

from enum import Enum

import numpy as np
from numba.typed import List
import vanilla_option_pricers as bsm

import stochvolmodels as svm
from stochvolmodels import fitters


def synthetic_chain() -> svm.OptionChain:
    """Standard European options with a known approximate LogSV smile."""
    ttms = np.array([0.25, 0.5])
    forwards = 100 * np.exp(0.02 * ttms)
    discounts = np.exp(-0.01 * ttms)
    k = np.linspace(-0.2, 0.2, 9)
    mid = fitters.calc_logsv_ivols(k, sigma0=0.3, beta=-0.12, volvol=0.2)
    strikes = List([f * np.exp(k) for f in forwards])
    types = List([np.where(k < 0, "P", "C") for _ in ttms])
    bid_iv, ask_iv = List([mid - 0.005 for _ in ttms]), List([mid + 0.005 for _ in ttms])
    prices = []
    for vols in (bid_iv, ask_iv):
        prices.append(
            List(
                [
                    np.array(
                        [
                            bsm.compute_bsm_vanilla_price(
                                forward=f,
                                strike=strike,
                                ttm=t,
                                vol=vol,
                                optiontype=cp,
                                discfactor=d,
                            )
                            for strike, cp, vol in zip(ks, cps, vs)
                        ]
                    )
                    for t, f, d, ks, cps, vs in zip(ttms, forwards, discounts, strikes, types, vols)
                ]
            )
        )
    return svm.OptionChain(
        ttms=ttms,
        forwards=forwards,
        discfactors=discounts,
        strikes_ttms=strikes,
        optiontypes_ttms=types,
        ids=np.array(["near", "far"]),
        bid_ivs=bid_iv,
        ask_ivs=ask_iv,
        bid_prices=prices[0],
        ask_prices=prices[1],
    )


def fitted_report():
    """Fit existing LogSV smiles and assess their final prices and IVs."""
    chain = synthetic_chain()
    model_ivols, model_prices = [], []
    for i, ttm in enumerate(chain.ttms):
        k = np.log(chain.strikes_ttms[i] / chain.forwards[i])
        params = fitters.fit_logsv_ivols(k, 0.5 * (chain.bid_ivs[i] + chain.ask_ivs[i]), ttm)
        vols = fitters.calc_logsv_ivols(k, **params)
        model_ivols.append(vols)
        model_prices.append(
            np.array(
                [
                    bsm.compute_bsm_vanilla_price(
                        forward=chain.forwards[i],
                        strike=strike,
                        ttm=ttm,
                        vol=vol,
                        optiontype=cp,
                        discfactor=chain.discfactors[i],
                    )
                    for strike, cp, vol in zip(
                        chain.strikes_ttms[i], chain.optiontypes_ttms[i], vols
                    )
                ]
            )
        )
    report = fitters.diagnose_fit(
        chain,
        model_ivols=model_ivols,
        model_prices=model_prices,
        price_unit="USD per unit",
        spread_floor=0.01,
    )
    return chain, report


def flat_surface(vol=0.3) -> fitters.SurfaceEvaluator:
    """Synthetic analytic evaluator with explicit common-underlying assumptions."""
    return fitters.SurfaceEvaluator(
        iv=lambda strikes, t: np.full_like(strikes, vol),
        forward=lambda t: 100 * np.exp(0.02 * t),
        discount=lambda t: np.exp(-0.01 * t),
        ttm_domain=(0.1, 1.0),
        k_domain=(-0.4, 0.4),
        underlying_id="synthetic equity",
        interpolation="constant volatility",
        carry_model="proportional",
    )


def plot_report(report):
    """Plot IV and price residuals from an already computed fit report."""
    import matplotlib.pyplot as plt

    quotes = report.table("quotes")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for expiry, group in quotes.groupby("expiry_id"):
        k = np.log(group.strike / group.forward)
        axes[0].plot(k, 100 * group.iv_residual, marker="o", label=expiry)
        axes[1].plot(k, group.spread_scaled_error, marker="o", label=expiry)
        half = 0.5 * (group.ask_price - group.bid_price)
        band = half / np.maximum(half, report.metadata["spread_floor"])
        axes[1].fill_between(k, -band, band, alpha=0.08)
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Forward log-moneyness")
        ax.legend()
    axes[0].set_ylabel("IV residual, vol points")
    axes[1].set_ylabel("Price residual / half-spread floor")
    return figure


def plot_surface_report(report):
    """Plot density and adjacent total-variance gaps from a stored surface report."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ttm, group in report.table("density").groupby("ttm"):
        axes[0].plot(group.strike, group.density, label=f"T={ttm:g}")
    grid = report.table("grid").pivot(index="log_moneyness", columns="ttm", values="total_variance")
    for previous, current in zip(grid.columns[:-1], grid.columns[1:]):
        axes[1].plot(
            grid.index, grid[current] - grid[previous], label=f"{previous:g} to {current:g}"
        )
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend()
    axes[0].set(xlabel="Strike", ylabel="Density per strike unit")
    axes[1].set(xlabel="Forward log-moneyness", ylabel="Total variance gap")
    return figure


class Locals(Enum):
    """Fast, deterministic examples with executable assertions."""

    FIT = 1
    SURFACE = 2
    COMPARISON = 3


def run_local(local: Locals) -> None:
    """Run a synthetic case without network or optional research packages."""
    if local == Locals.FIT:
        _, report = fitted_report()
        summary = report.table("summary").iloc[0]
        assert summary.iv_rmse < 1e-6
        assert summary.price_inside_bid_ask_fraction == 1
        print(report.table("summary").to_string(index=False))
    elif local == Locals.SURFACE:
        report = fitters.diagnose_surface(
            flat_surface(),
            [0.25, 0.5],
            np.linspace(-0.3, 0.3, 61),
            extra_ttms=[0.375],
            price_unit="USD per unit",
        )
        assert report.table("checks").status.eq("pass").all()
        print(report.table("checks").to_string(index=False))
    elif local == Locals.COMPARISON:
        old = fitters.diagnose_surface(
            flat_surface(0.3), [0.25, 0.5], [-0.2, 0, 0.2], price_unit="USD"
        )
        new = fitters.diagnose_surface(
            flat_surface(0.32), [0.25, 0.5], [-0.2, 0, 0.2], price_unit="USD"
        )
        change = fitters.compare_surface_reports(old, new)
        np.testing.assert_allclose(change.table("expiries").atm_change, 0.02)
        np.testing.assert_allclose(change.table("expiries").skew_change, 0, atol=1e-14)
        print(change.table("expiries").to_string(index=False))
    else:
        raise ValueError(local)


if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
