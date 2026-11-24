"""Standalone rough valuation and legacy fixed-random Monte Carlo comparisons."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as mticker
from enum import Enum
import stochvolmodels as sv
from stochvolmodels.utils import plots as plot
from stochvolmodels import LogSvParams


from scipy.special import ndtr
from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.pricers.rough_logsv import RoughLogSVPricer, RoughLogSvParams


def standalone() -> None:
    """Check the constant-volatility boundary of the rough pricer against a Black call."""
    params = RoughLogSvParams(H=0.3, sigma0=0.2, theta=0.2, beta=0, epsilon=0)
    ttm = 28 / 365
    chain = OptionChain.get_uniform_chain(np.array([ttm]), np.array(["28d"]), strikes=np.array([1.0]))
    result = RoughLogSVPricer().price_chain_result(chain, params)
    black = 2 * ndtr(params.sigma0 * np.sqrt(ttm) / 2) - 1
    np.testing.assert_allclose(result.prices[0][0], black, atol=1e-12, rtol=0)
    print(f"28-day ATM call {result.prices[0][0]:.6f} per unit forward, Black {black:.6f}")


class Locals(Enum):
    STANDALONE = 1
    ROUGH_MC_WITH_FIXED_RANDOMS = 9
    BENCHM_ROUGH_PRICER = 10
    CALIBRATE_MODEL_TO_BTC_OPTIONS_WITH_MC = 8


def run_local(local: Locals):
    if local == Locals.STANDALONE:
        standalone()
    elif local == Locals.ROUGH_MC_WITH_FIXED_RANDOMS:
        btc_option_chain = sv.get_btc_test_chain_data()
        Z0, Z1, grid_ttms = sv.get_randoms_for_rough_vol_chain_valuation(ttms=btc_option_chain.ttms,
                                                                             nb_path=10000,
                                                                             nb_steps_per_year=360,
                                                                             seed=10)
        params0 = LogSvParams(sigma0=0.377, theta=0.347, kappa1=1.29, kappa2=1.93, beta=2.45, volvol=1.81)
        params0.H = 0.1
        params0.approximate_kernel(T=btc_option_chain.ttms[-1])

        option_prices_ttm, option_std_ttm = sv.rough_logsv_mc_chain_pricer_fixed_randoms(ttms=btc_option_chain.ttms,
                                                                                         forwards=btc_option_chain.forwards,
                                                                                         discfactors=btc_option_chain.discfactors,
                                                                                         strikes_ttms=btc_option_chain.strikes_ttms,
                                                                                         optiontypes_ttms=btc_option_chain.optiontypes_ttms,
                                                                                         Z0=Z0,
                                                                                         Z1=Z1,
                                                                                         sigma0=params0.sigma0,
                                                                                         theta=params0.theta,
                                                                                         kappa1=params0.kappa1,
                                                                                         kappa2=params0.kappa2,
                                                                                         beta=params0.beta,
                                                                                         orthog_vol=params0.volvol,
                                                                                         weights=params0.weights,
                                                                                         nodes=params0.nodes,
                                                                                         timegrids=grid_ttms)
        print(option_prices_ttm)

    elif local == Locals.BENCHM_ROUGH_PRICER:
        btc_option_chain = sv.get_btc_test_chain_data()
        # params0 = LogSvParams(sigma0=0.8, theta=1.0, kappa1=2.21, kappa2=1.0, beta=0.15, volvol=1.0)
        params0 = LogSvParams(sigma0=0.377, theta=0.347, kappa1=1.29, kappa2=1.93, beta=2.45, volvol=1.81)
        nb_path = 10000
        H = 0.1
        seed = 1

        def rough_vol():
            params1 = LogSvParams.copy(params0)
            params1.H = H
            params1.approximate_kernel(T=btc_option_chain.ttms[-1])

            Z0, Z1, grid_ttms = sv.get_randoms_for_rough_vol_chain_valuation(ttms=btc_option_chain.ttms,
                                                                             nb_path=nb_path,
                                                                             nb_steps_per_year=360,
                                                                             seed=seed)


            option_prices_ttm, option_std_ttm = sv.rough_logsv_mc_chain_pricer_fixed_randoms(ttms=btc_option_chain.ttms,
                                                                                             forwards=btc_option_chain.forwards,
                                                                                             discfactors=btc_option_chain.discfactors,
                                                                                             strikes_ttms=btc_option_chain.strikes_ttms,
                                                                                             optiontypes_ttms=btc_option_chain.optiontypes_ttms,
                                                                                             Z0=Z0,
                                                                                             Z1=Z1,
                                                                                             sigma0=params0.sigma0,
                                                                                             theta=params0.theta,
                                                                                             kappa1=params0.kappa1,
                                                                                             kappa2=params0.kappa2,
                                                                                             beta=params0.beta,
                                                                                             orthog_vol=params0.volvol,
                                                                                             weights=params1.weights,
                                                                                             nodes=params1.nodes,
                                                                                             timegrids=grid_ttms)
            model_ivols_ttms = btc_option_chain.compute_model_ivols_from_chain_data(option_prices_ttm)
            return model_ivols_ttms

        def regular_vol():
            W0s, W1s, dts = sv.get_randoms_for_chain_valuation(ttms=btc_option_chain.ttms,
                                                               nb_path=nb_path,
                                                               nb_steps_per_year=360,
                                                               seed=10)
            vol_backbone_etas = params0.get_vol_backbone_etas(ttms=btc_option_chain.ttms)
            args = dict(ttms=btc_option_chain.ttms,
                        forwards=btc_option_chain.forwards,
                        discfactors=btc_option_chain.discfactors,
                        strikes_ttms=btc_option_chain.strikes_ttms,
                        optiontypes_ttms=btc_option_chain.optiontypes_ttms,
                        W0s=W0s,
                        W1s=W1s,
                        dts=dts,
                        v0=params0.sigma0,
                        theta=params0.theta,
                        kappa1=params0.kappa1,
                        kappa2=params0.kappa2,
                        beta=params0.beta,
                        volvol=params0.volvol,
                        vol_backbone_etas=vol_backbone_etas)
            option_prices_ttm, option_std_ttm = sv.logsv_mc_chain_pricer_fixed_randoms(**args)
            model_ivols_ttms = btc_option_chain.compute_model_ivols_from_chain_data(option_prices_ttm)

            return model_ivols_ttms

        ivols_rough_logsv = rough_vol()
        ivols_logsv = regular_vol()

        nb_slices = btc_option_chain.ttms.size
        assert nb_slices == 4

        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(2, 2, figsize=(15, 9), tight_layout=True)
        axs = plot.to_flat_list(axs)

        for i in range(nb_slices):
            data = pd.DataFrame({"Rough Log-SV": ivols_rough_logsv[i],
                                 "Log-SV": ivols_logsv[i]},
                                index=np.log(
                                    btc_option_chain.strikes_ttms[i] / btc_option_chain.forwards[i]))
            ax = axs[i]
            sns.lineplot(data, ax=ax, markers=["o"])
            ax.set_title("Expiry: " + btc_option_chain.ids[i])
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda z, _: '{:.0%}'.format(z)))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda z, _: '{:.2f}'.format(z)))
            ax.legend()
        fig.suptitle(f"Conventional LogSV model vs Rough LogSV, H={H:.2f} via Markovian approximation\n"
                     f"{params0.to_str()}",
                     color = "darkblue", fontsize = 14)


    elif local == Locals.CALIBRATE_MODEL_TO_BTC_OPTIONS_WITH_MC:
        raise NotImplementedError(
            "Rough calibration is gated. Use the standalone RoughLogSVPricer and "
            "validation/rough_logsv/benchmark.py for the selected BTC/ETH cases.")

    plt.show()


if __name__ == "__main__":
    run_local(Locals.STANDALONE)
