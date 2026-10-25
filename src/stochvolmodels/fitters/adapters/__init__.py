"""Adapters from external quotes to SVM calibration and diagnostic inputs.

The OCA adapter converts an already reconstructed observation into OptionChain
arrays. Providers own snapshot reconstruction and quote conventions; importing
this namespace does not import or require OptionChainAnalytics itself. Native
quote diagnostics accept explicitly aligned arrays without a provider dependency.
"""

from stochvolmodels.fitters.adapters.native_quotes import (
    diagnose_native_quotes,
    normalized_vanilla_price_bounds,
)
from stochvolmodels.fitters.adapters.oca import option_chain_from_oca

__all__ = ['option_chain_from_oca', 'diagnose_native_quotes', 'normalized_vanilla_price_bounds']
