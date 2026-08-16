"""Deterministic offline first result for the stable LogSV API."""

from time import perf_counter

import numpy as np

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
