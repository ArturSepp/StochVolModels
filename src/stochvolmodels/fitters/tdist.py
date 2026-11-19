"""Compatibility imports for ``stochvolmodels.models.tdist.distribution``.

The canonical implementation owns the definitions and numerical behavior.
Existing import paths retain the same objects without deprecation warnings.
"""

from stochvolmodels.models.tdist.distribution import (
    np as np,
    njit as njit,
    betainc as betainc,
    gamma as gamma,
    fsolve as fsolve,
    Union as Union,
    compute_upsilon as compute_upsilon,
    pdf_tdist as pdf_tdist,
    cdf_tdist as cdf_tdist,
    cum_mean_tdist as cum_mean_tdist,
    imply_drift_tdist as imply_drift_tdist,
    compute_default_prob_tdist as compute_default_prob_tdist,
    compute_forward_tdist as compute_forward_tdist,
)

from stochvolmodels.pricers.tdist.analytics import (
    compute_vanilla_price_tdist as compute_vanilla_price_tdist,
    infer_implied_vol_tdist as infer_implied_vol_tdist,
    infer_tdist_implied_vols_from_model_slice_prices as infer_tdist_implied_vols_from_model_slice_prices,
)
