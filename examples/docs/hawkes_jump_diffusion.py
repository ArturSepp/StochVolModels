"""Canonical script of docs/hawkes_jump_diffusion.md.

Every Python block on that page is a verbatim excerpt of this file, and every number the page
quotes is asserted here. A diffusion with positive and negative jumps whose intensities are
self-exciting Hawkes processes is priced from its affine moment generating function, for three
strengths of clustering at the same average jump intensity, and checked by the package's Monte
Carlo simulation. The Monte Carlo case takes minutes and is listed in ``SLOW_CASES``. Select a case
in ``Locals`` or run the file to execute every case.
"""
from dataclasses import replace
from enum import Enum

import numpy as np

import stochvolmodels as svm

SLOW_CASES = ("MONTE_CARLO",)
BASE = svm.HawkesJDParams()  # defaults: annualised parameters close to Bitcoin's
TTMS = np.array([1.0 / 12.0, 0.5])
Z = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])  # strikes in standard deviations at 60%
BRANCHING = (0.0, 0.35, 0.7)


def clustered(n: float, base: svm.HawkesJDParams = BASE) -> svm.HawkesJDParams:
    """Self-excitation with branching ratio n for both jump signs, no cross-excitation.

    Each jump raises its own intensity by n kappa / E[J] per unit of jump size, so it triggers n
    further jumps on average; theta = lambda0 (1 - n) keeps the mean intensity at lambda0.
    """
    mean_jump_p, mean_jump_m = base.shift_p + base.mean_p, base.shift_m + base.mean_m
    return replace(base, beta1_p=n * base.kappa_p / mean_jump_p, beta2_p=0.0,
                   beta1_m=0.0, beta2_m=n * base.kappa_m / mean_jump_m,
                   theta_p=base.lambda_p * (1.0 - n), theta_m=base.lambda_m * (1.0 - n))


def chain() -> svm.OptionChain:
    """Out-of-the-money options at one and six months, forward 1."""
    strikes = tuple(np.exp(Z * 0.6 * np.sqrt(ttm)) for ttm in TTMS)
    return svm.OptionChain(ttms=TTMS, forwards=np.ones(2), discfactors=np.ones(2),
                           strikes_ttms=strikes, ids=np.array(["1m", "6m"]),
                           optiontypes_ttms=tuple(np.where(k < 1.0, "P", "C") for k in strikes))


def smiles(params: svm.HawkesJDParams) -> list:
    """Implied volatilities from the Fourier inversion of the affine MGF."""
    return svm.HawkesJDPricer().compute_model_ivols_for_chain(option_chain=chain(), params=params)


def poisson_log_mgf(params: svm.HawkesJDParams, ttm: float, phi: np.ndarray) -> np.ndarray:
    """log E[exp(-phi X_T)] without excitation: constant intensities, shifted exponential jumps."""
    def jump_term(intensity, shift, mean, compensator):
        return intensity * (np.exp(-shift * phi) / (1.0 + mean * phi) - 1.0 + compensator * phi)
    return ttm * (0.5 * params.sigma ** 2 * phi * (phi + 1.0)
                  + jump_term(params.lambda_p, params.shift_p, params.mean_p, params.compensator_p)
                  + jump_term(params.lambda_m, params.shift_m, params.mean_m, params.compensator_m))

def monte_carlo(params: svm.HawkesJDParams, nb_batch: int = 20, nb_path: int = 10000,
                seed: int = 2026) -> tuple:
    """Monte Carlo prices and standard errors pooled over batches that fit in memory.

    The simulator draws from NumPy's global generator, which is seeded once here.
    """
    np.random.seed(seed)
    batches = []
    for _ in range(nb_batch):
        batches.append(svm.HawkesJDPricer().model_mc_price_chain(option_chain=chain(),
                                                                 params=params, nb_path=nb_path))
    prices = [np.mean([p[i] for p, _ in batches], axis=0) for i in range(len(TTMS))]
    errors = [np.sqrt(np.sum([s[i] ** 2 for _, s in batches], axis=0)) / nb_batch
              for i in range(len(TTMS))]
    return prices, errors


class Locals(Enum):
    """Cases of this script; each asserts the numbers the page quotes."""
    STATIONARITY = 1
    MARTINGALE = 2
    POISSON_LIMIT = 3
    SMILES = 4
    MONTE_CARLO = 5


def run_local(local: Locals) -> None:
    """Run one case and assert its documented results."""
    if local == Locals.STATIONARITY:
        conditions = [(clustered(n).jump1_cond, clustered(n).jump2_cond) for n in BRANCHING]
        # kappa minus the expected excitation per jump stays positive: kappa (1 - n)
        np.testing.assert_allclose(conditions, [(22.29, 29.0), (14.4885, 18.85), (6.687, 8.70)],
                                   atol=1e-3)
        np.testing.assert_allclose([clustered(0.7).beta1_p, clustered(0.7).beta2_m],
                                   [173.37, -225.56], atol=0.01)

    elif local == Locals.MARTINGALE:
        one = svm.OptionChain.slice_to_chain(ttm=0.5, forward=1.0, strikes=np.array([1e-8]),
                                             optiontypes=np.array(["C"]))
        for n in BRANCHING:
            price = svm.HawkesJDPricer().price_chain(option_chain=one, params=clustered(n))[0][0]
            assert abs(price - 1.0) < 1e-7  # a call struck at zero is worth the forward

    elif local == Locals.POISSON_LIMIT:
        params, pricer = clustered(0.0), svm.HawkesJDPricer()
        phi, _, _ = svm.get_transform_var_grid(vol_scaler=0.45 * np.sqrt(1.0 / 12.0), max_phi=500)
        packaged = pricer.price_chain(option_chain=chain(), params=params)
        for idx, ttm in enumerate(TTMS):
            closed = svm.vanilla_slice_pricer_with_mgf_grid(
                log_mgf_grid=poisson_log_mgf(params, ttm, phi), phi_grid=phi, forward=1.0,
                strikes=chain().strikes_ttms[idx], optiontypes=chain().optiontypes_ttms[idx])
            # the Riccati solution equals the closed form when intensities are constant
            assert np.max(np.abs(packaged[idx] - closed)) < 1e-6

    elif local == Locals.SMILES:
        vols = {n: 100 * np.array(smiles(clustered(n))) for n in BRANCHING}
        print({n: np.round(v, 2) for n, v in vols.items()})
        atm = 3
        # the same mean intensity: clustering lowers the at-the-money volatility, lifts the wings
        np.testing.assert_allclose([vols[n][:, atm] for n in BRANCHING],
                                   [[57.57, 57.99], [56.74, 57.55], [55.51, 55.94]], atol=0.01)
        np.testing.assert_allclose([vols[n][1, [0, -1]] for n in BRANCHING],
                                   [[58.49, 58.21], [59.70, 59.62], [62.94, 64.22]], atol=0.01)
        # at six months the Poisson smile is nearly flat, the clustered one still curved
        assert np.ptp(vols[0.0][1]) < 0.6 and np.ptp(vols[0.7][1]) > 8.0

    elif local == Locals.MONTE_CARLO:
        z, gaps = {}, {}
        for n in BRANCHING:
            params = clustered(n)
            analytic = svm.HawkesJDPricer().price_chain(option_chain=chain(), params=params)
            mc, se = monte_carlo(params)
            mc_vols = chain().compute_model_ivols_from_chain_data(model_prices=mc)
            z[n] = [(a - m) / s for a, m, s in zip(analytic, mc, se)]
            gaps[n] = 100 * (np.array(smiles(params)) - np.array(mc_vols))
            print(n, [np.round(v, 2) for v in z[n]], np.round(gaps[n], 2))
        # 200,000 paths, seed 2026: agreement within two standard errors without and with
        # moderate clustering; with strong clustering the simulation is lower at six months
        assert max(np.max(np.abs(np.concatenate(z[n]))) for n in (0.0, 0.35)) < 2.0
        assert np.all(z[0.7][1] > 3.0)
        np.testing.assert_allclose([gaps[0.7][1].min(), gaps[0.7][1].max()], [0.81, 2.17],
                                   atol=0.01)

if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
