"""Deterministic offline first result for the stable LogSV API."""

from time import perf_counter

import numpy as np
from scipy.special import ndtr

import stochvolmodels as svm


def main() -> None:
    """Price one vanilla and a small synthetic chain, then print compact evidence."""
    started = perf_counter()
    params = svm.LogSvParams(
        sigma0=1.0,
        theta=1.0,
        kappa1=5.0,
        kappa2=5.0,
        beta=0.2,
        volvol=2.0,
    )
    pricer = svm.LogSVPricer()

    vanilla_price, vanilla_ivol = pricer.price_vanilla(
        params=params,
        ttm=0.25,
        forward=1.0,
        strike=1.0,
        optiontype="C",
    )

    chain = svm.OptionChain.get_uniform_chain(
        ttms=np.array([0.25, 0.5]),
        ids=np.array(["3m", "6m"]),
        forwards=np.array([1.0, 1.0]),
        strikes=np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
    )
    chain_prices, chain_ivols = pricer.compute_chain_prices_with_vols(
        option_chain=chain,
        params=params,
    )

    assert all(np.all(np.isfinite(values)) for values in (*chain_prices, *chain_ivols))
    np.testing.assert_allclose(vanilla_price, 0.197698, rtol=5.0e-6, atol=1.0e-8)
    np.testing.assert_allclose(vanilla_ivol, 1.001478, rtol=5.0e-6, atol=1.0e-8)
    np.testing.assert_allclose(chain_prices[1][2], 0.275623, rtol=5.0e-6, atol=1.0e-8)
    np.testing.assert_allclose(chain_ivols[1][2], 0.997347, rtol=5.0e-6, atol=1.0e-8)
    # Independent price/IV consistency checks; these do not bound expansion error.
    np.testing.assert_allclose(
        vanilla_price, 2.0 * ndtr(vanilla_ivol * np.sqrt(0.25) / 2.0) - 1.0,
        rtol=0.0, atol=2.0e-12,
    )
    for ttm, forward, discount, strikes, types, prices, vols in zip(
        chain.ttms, chain.forwards, chain.discfactors, chain.strikes_ttms,
        chain.optiontypes_ttms, chain_prices, chain_ivols,
    ):
        sd = vols * np.sqrt(ttm)
        d1 = np.log(forward / strikes) / sd + 0.5 * sd
        calls = forward * ndtr(d1) - strikes * ndtr(d1 - sd)
        expected = discount * (calls - np.where(types == "P", forward - strikes, 0.0))
        np.testing.assert_allclose(prices, expected, rtol=0.0, atol=2.0e-12)
    print(f"stochvolmodels={svm.__version__}")
    print(f"maturities={chain.ttms.tolist()}")
    print(f"price_shapes={[values.shape for values in chain_prices]}")
    print(f"ivol_shapes={[values.shape for values in chain_ivols]}")
    print(f"vanilla_price={vanilla_price:.6f}")
    print(f"vanilla_ivol={vanilla_ivol:.6f}")
    print(f"six_month_atm_price={chain_prices[1][2]:.6f}")
    print(f"six_month_atm_ivol={chain_ivols[1][2]:.6f}")
    print(f"elapsed_seconds={perf_counter() - started:.2f}")


if __name__ == "__main__":
    main()
