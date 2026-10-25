"""Portable synthetic quote challenges for repository-only SVI research.

Extracted from the October 2026 agent studies ``run_wing_reliability_expanded.py``
and ``run_svi_research_policy_challenge.py``. This retains their three profiles,
tenors, grids, input seeds and noise transformations without archived paths,
parallel fit runners, OCA dependencies or fitted cutoff artifacts. Existing
``_svi_benchmark.raw_svi`` owns the total-variance formula.

All strikes are log-moneyness, volatilities are annualized decimal IV and tenors
are years. The panel contains 48 observed datasets and optionally 12 alternative
hidden-wing truths sharing the Gaussian controls' exact observations. The four
regimes within each geometry share a random driver: this is a paired comparison,
not 48 independent calibration draws. The truth arrays are evaluation evidence,
never calibration inputs. These synthetic error
laws and alternatives do not establish market error distributions, arbitrage
freedom, warning thresholds or population-wide accuracy.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr

from _svi_benchmark import RawSvi, raw_svi


Regime = Literal["gaussian_control", "spread_uniform", "spread_correlated", "localized_bump"]
REGIMES: tuple[Regime, ...] = (
    "gaussian_control", "spread_uniform", "spread_correlated", "localized_bump",
)
_PROFILES = (
    RawSvi(.004, .018, -.4, .01, .08),
    RawSvi(.007, .025, .3, -.015, .12),
    RawSvi(.003, .025, -.65, .025, .045),
)


@dataclass(frozen=True)
class SyntheticChallenge:
    """One observed smile and its separately supplied evaluation truth.

    ``observations_id`` identifies a distinct quote set, not an independent draw;
    the hidden alternative deliberately reuses its control's identity and quotes. Array fields are
    read-only so a consumer cannot accidentally alter another paired case.
    """

    case_id: str
    observations_id: str
    regime: Regime
    hidden_wing: bool
    parameters: RawSvi
    ttm: float
    half_width: float
    input_seed: int
    k: NDArray[np.float64]
    bid_iv: NDArray[np.float64]
    ask_iv: NDArray[np.float64]
    evaluation_k: NDArray[np.float64]
    evaluation_truth_iv: NDArray[np.float64]


def _readonly(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Own an immutable array rather than a view into caller-owned state."""
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def synthetic_challenge_panel(*, include_hidden: bool = True) -> tuple[SyntheticChallenge, ...]:
    """Build the fixed inexpensive panel without fitting or reading any files.

    Returns
    -------
    tuple of SyntheticChallenge
        Three profiles by two tenors by two half-widths by four regimes, followed
        by each geometry's paired hidden truth when requested. Gaussian and bump
        cases use IV noise standard deviation .001 and half-spread .005. Uniform
        errors remain inside the heteroskedastic spreads; correlated Gaussian
        errors have marginal variance ``half_spread**2 / 3`` and correlation .7.
        Hidden changes vanish on the observed interval and reach .03 decimal IV
        at twice its half-width. The caller decides which data enter a study.
    """
    cases: list[SyntheticChallenge] = []
    geometry = 0
    for profile in _PROFILES:
        for ttm in (.1, .75):
            parameters = RawSvi(profile.a * ttm / .25, profile.b * ttm / .25,
                                profile.rho, profile.m, profile.sigma)
            for width in (.2, .4):
                k = np.linspace(-width, width, 31)
                evaluation_k = np.linspace(-2 * width, 2 * width, 81)
                input_seed = 15000000 + geometry
                driver = np.random.default_rng(input_seed).normal(0, .001, len(k))
                for regime in REGIMES:
                    spread = np.full(len(k), .005)
                    noise = driver.copy()
                    if regime in ("spread_uniform", "spread_correlated"):
                        spread *= 1 + 2 * (np.abs(k) / width)**2
                        z = driver / .001
                        if regime == "spread_uniform":
                            noise = spread * (2 * ndtr(z) - 1)
                        else:
                            root = np.sqrt(1 - .7)
                            noise = spread / np.sqrt(3) * (
                                root * z + (np.sqrt(1 + (len(k) - 1) * .7) - root)
                                * np.sum(z) / len(k))
                    mid = np.sqrt(raw_svi(k, parameters)[0] / ttm)
                    truth = np.sqrt(raw_svi(evaluation_k, parameters)[0] / ttm)
                    if regime == "localized_bump":
                        mid += .03 * np.exp(-.5 * ((k - .6 * width) / (.12 * width))**2)
                        truth += .03 * np.exp(
                            -.5 * ((evaluation_k - .6 * width) / (.12 * width))**2)
                    mid += noise
                    identity = f"g{geometry:02d}_{regime}"
                    case = SyntheticChallenge(
                        identity, identity, regime, False, parameters, ttm, width, input_seed,
                        _readonly(k), _readonly(mid - spread), _readonly(mid + spread),
                        _readonly(evaluation_k), _readonly(truth),
                    )
                    cases.append(case)
                    if include_hidden and regime == "gaussian_control":
                        hidden = truth + .03 * np.minimum(
                            np.maximum((np.abs(evaluation_k) - width) / width, 0), 1)**2
                        cases.append(SyntheticChallenge(
                            identity + "_hidden_wing", identity, regime, True,
                            parameters, ttm, width, input_seed, case.k, case.bid_iv,
                            case.ask_iv, case.evaluation_k, _readonly(hidden),
                        ))
                geometry += 1
    return tuple(cases)
