"""Gaussian-mixture parameters and terminal log-return densities.

Each component carries its own weight, drift and volatility at a fixed maturity.
Mixture calibration and option pricing remain with GmmPricer; this module owns
parameter sorting and per-state or aggregate density evaluation.
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from stochvolmodels.models.base import ModelParams
from stochvolmodels.utils.funcs import npdf


@dataclass
class GmmParams(ModelParams):
    """
    parameters of a Gaussian mixture model for terminal log-returns.

    The terminal density is a weighted sum of normals, one per state, each with its
    own drift and volatility. Weights sum to one and the mixture must reprice the
    forward, which are the two equality constraints imposed during calibration.
    """
    gmm_weights: np.ndarray
    gmm_mus: np.ndarray
    gmm_vols: np.ndarray
    # TTM is fixed for each fitted slice and is not itself calibrated.
    ttm: float

    def sort_by_mus(self):
        """order the mixture states by drift, so fitted states stay comparable across slices."""
        indices = np.argsort(self.gmm_mus)
        self.gmm_weights = self.gmm_weights[indices]
        self.gmm_mus = self.gmm_mus[indices]
        self.gmm_vols = self.gmm_vols[indices]

    def get_get_avg_vol(self) -> float:
        """weight-averaged volatility, sqrt(sum w_i vol_i^2)."""
        return np.sqrt(np.sum(self.gmm_weights*np.square(self.gmm_vols)))

    def compute_state_pdfs(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """per-state densities and their weighted aggregate on a log-return grid."""
        state_pdfs = np.zeros((len(x), len(self.gmm_weights)))
        agg_pdf = np.zeros_like(x)
        states = zip(self.gmm_weights, self.gmm_mus, self.gmm_vols)
        for idx, (gmm_weight, mu, vol) in enumerate(states):
            state_pdf = npdf(x, mu=mu*self.ttm, vol=vol*np.sqrt(self.ttm))
            state_pdfs[:, idx] = state_pdf
            agg_pdf += gmm_weight*state_pdf
        return state_pdfs, agg_pdf

    def compute_pdf(self, x: np.ndarray):
        """aggregate mixture density on a log-return grid."""
        pdfs = np.zeros_like(x)
        for gmm_weight, mu, vol in zip(self.gmm_weights, self.gmm_mus, self.gmm_vols):
            pdfs = pdfs + gmm_weight*npdf(x, mu=mu*self.ttm, vol=vol*np.sqrt(self.ttm))
        return pdfs
