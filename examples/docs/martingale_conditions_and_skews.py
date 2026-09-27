"""Canonical script of docs/martingale_conditions_and_skews.md.

Every Python block on that page is a verbatim excerpt of this file, and every number the page
quotes is asserted here. The martingale conditions of Theorem 3.7 of Sepp and Rakhmonov (2023) are
checked by Monte Carlo: inside the admissible region the simulated E[Z_T] stays at Z_0, outside it
falls below. Select a case in ``Locals`` or run the file to execute every case.
"""
from enum import Enum

import numpy as np

import stochvolmodels as svm
from stochvolmodels.utils.funcs import set_seed


def admissibility(params: svm.LogSvParams) -> dict:
    """Valuation measures that Theorem 3.7 admits, and the fourth-moment calibration condition."""
    kappa = params.kappa1 + params.kappa2 * params.theta
    vartheta2 = params.beta ** 2 + params.volvol ** 2
    return {"mma": params.kappa2 >= params.beta,
            "inverse": params.kappa2 >= 2.0 * params.beta,
            "fourth_moment": kappa >= 1.5 * vartheta2}


def test_params(kappa2: float, beta: float, vartheta: float = 1.5) -> svm.LogSvParams:
    """kappa1 = 2 and theta = sigma0 = 1; the total vol-of-vol is split between beta and volvol."""
    return svm.LogSvParams(sigma0=1.0, theta=1.0, kappa1=2.0, kappa2=kappa2, beta=beta,
                           volvol=np.sqrt(vartheta ** 2 - beta ** 2))


def expected_price_ratio(params: svm.LogSvParams, ttm: float = 1.0, nb_path: int = 100000,
                         seed: int = 5) -> tuple:
    """Monte Carlo estimate of E[Z_T] / Z_0 under the MMA measure, with its standard error."""
    set_seed(seed)
    log_returns, _, _ = svm.LogSVPricer().simulate_terminal_values(params=params, ttm=ttm,
                                                                   nb_path=nb_path)
    ratio = np.exp(log_returns)
    return np.mean(ratio), np.std(ratio) / np.sqrt(nb_path)


def smile_params(beta: float) -> svm.LogSvParams:
    """Volatility 50%, kappa2 = 2.5 and a total vol-of-vol of 1.5 split between beta and volvol."""
    return svm.LogSvParams(sigma0=0.5, theta=0.5, kappa1=2.0, kappa2=2.5, beta=beta,
                           volvol=np.sqrt(1.5 ** 2 - beta ** 2))


def smiles_in_beta(betas: tuple = (-1.0, -0.5, 0.0, 0.5, 1.0)) -> dict:
    """One-month smiles at a total vol-of-vol of 1.5 for volatility betas from -1 to 1."""
    strikes = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    optiontypes = np.array(["P", "P", "C", "C", "C"])
    smiles = {}
    for beta in betas:
        _, ivols = svm.LogSVPricer().price_slice(params=smile_params(beta), ttm=1.0 / 12.0,
                                                 forward=1.0, strikes=strikes,
                                                 optiontypes=optiontypes)
        smiles[beta] = ivols
    return smiles


class Locals(Enum):
    """Cases of this script; each asserts the numbers the page quotes."""
    ADMISSIBILITY = 1
    MARTINGALE_TEST = 2
    SMILES = 3


def run_local(local: Locals) -> None:
    """Run one case and assert its documented results."""
    if local == Locals.ADMISSIBILITY:
        assert admissibility(test_params(kappa2=1.0, beta=0.4)) == {
            "mma": True, "inverse": True, "fourth_moment": False}
        assert admissibility(test_params(kappa2=1.0, beta=0.8))["inverse"] is False
        assert admissibility(test_params(kappa2=0.0, beta=0.1))["mma"] is False
        # the constraint types of calibration impose the same inequalities
        assert [c.name for c in svm.ConstraintsType] == [
            "UNCONSTRAINT", "MMA_MARTINGALE", "INVERSE_MARTINGALE",
            "MMA_MARTINGALE_MOMENT4", "INVERSE_MARTINGALE_MOMENT4"]

    elif local == Locals.MARTINGALE_TEST:
        # well inside the region the estimate stays at one; well outside it falls far below
        inside, outside = ((1.0, -0.5), (1.0, 0.0)), ((0.0, 0.9), (1.0, 1.4))
        for kappa2, beta in (*inside, *outside):
            mean, standard_error = expected_price_ratio(test_params(kappa2=kappa2, beta=beta))
            if (kappa2, beta) in inside:
                assert abs(mean - 1.0) < 3.0 * standard_error, (kappa2, beta, mean, standard_error)
            else:
                assert mean < 0.9 and mean < 1.0 - 10.0 * standard_error, (kappa2, beta, mean)
                np.testing.assert_allclose(mean, {(0.0, 0.9): 0.804, (1.0, 1.4): 0.831}[
                    (kappa2, beta)], atol=5e-4)
            print(kappa2, beta, mean, standard_error)

    elif local == Locals.SMILES:
        smiles = smiles_in_beta()
        skews = {beta: ivols[-1] - ivols[0] for beta, ivols in smiles.items()}
        assert all(np.sign(skews[beta]) == np.sign(beta) for beta in (-1.0, -0.5, 0.5, 1.0))
        np.testing.assert_allclose([skews[b] for b in (-1.0, 1.0)], [-0.174, 0.168], atol=5e-4)
        print(skews)


if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
