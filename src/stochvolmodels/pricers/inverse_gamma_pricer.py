"""Price a single-expiry European option chain under the inverse-gamma/normal terminal law."""

from __future__ import annotations

import numpy as np
from numba.typed import List

from stochvolmodels.data.option_chain import OptionChain, OptionSlice
from stochvolmodels.models.inverse_gamma import (
    InverseGammaNormalParams,
    InverseGammaNormalTerminalModel,
)
from stochvolmodels.pricers.model_pricer import ModelPricer
from stochvolmodels.utils.config import VariableType


class InverseGammaNormalPricer(ModelPricer):
    """Provisional chain adapter for discounted European calls and puts at one maturity."""

    def price_chain(
        self,
        option_chain: OptionChain,
        params: InverseGammaNormalParams,
        *,
        variable_type: VariableType = VariableType.LOG_RETURN,
        is_spot_measure: bool = True,
        quadrature_order: int = 256,
    ) -> List[np.ndarray]:
        """Return one quote-currency price vector in the chain's original strike order."""
        if not isinstance(option_chain, OptionChain):
            raise TypeError("option_chain must be an OptionChain")
        if len(option_chain.ttms) != 1:
            raise ValueError("InverseGammaNormalPricer requires a single maturity")
        if variable_type is not VariableType.LOG_RETURN:
            raise ValueError("InverseGammaNormalPricer requires VariableType.LOG_RETURN")
        if is_spot_measure is not True:
            raise ValueError("InverseGammaNormalPricer requires the spot measure")

        model = InverseGammaNormalTerminalModel(params, quadrature_order=quadrature_order)
        ttm = option_chain.ttms[0]
        option_slice = OptionSlice(
            id=option_chain.ids[0] if option_chain.ids is not None else str(ttm),
            ttm=ttm,
            forward=option_chain.forwards[0],
            discfactor=option_chain.discfactors[0],
            strikes=option_chain.strikes_ttms[0],
            optiontypes=option_chain.optiontypes_ttms[0],
        )
        return List([model.price_european(option_slice)])
