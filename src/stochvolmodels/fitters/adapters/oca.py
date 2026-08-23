"""Convert point-in-time OptionChainAnalytics objects into SVM calibration inputs.

Observation-time reconstruction and frequency sampling remain owned by
OptionChainAnalytics. This optional adapter only selects expiries and quotes inside one
already-reconstructed ``SlicesChain`` and materializes an SVM ``OptionChain``.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from numba.typed import List

from stochvolmodels.data.option_chain import OptionChain


def option_chain_from_oca(
    chain: object,
    value_time: Optional[pd.Timestamp] = None,
    days_map: Optional[Dict[str, int]] = None,
    delta_bounds: Tuple[Optional[float], Optional[float]] = (-0.1, 0.1),
    is_filtered: bool = True,
) -> OptionChain:
    """Convert one reconstructed OCA chain into SVM calibration inputs.

    Parameters
    ----------
    chain : option_chain_analytics.SlicesChain
        Point-in-time chain reconstructed by OCA.
    value_time : pandas.Timestamp, optional
        Anchor for maturity targets. Defaults to the chain's actual observation time.
    days_map : dict[str, int], optional
        Calibration maturity labels and calendar-day offsets. Defaults to 7 and 21 days.
    delta_bounds : tuple[float or None, float or None], default=(-0.1, 0.1)
        OCA joint-slice delta filter.
    is_filtered : bool, default=True
        Apply OCA's quote filtering before conversion.

    Returns
    -------
    OptionChain
        SVM's array-oriented calibration container.
    """
    try:
        from option_chain_analytics import SliceColumn, SlicesChain
    except ImportError as error:
        raise ImportError(
            'option_chain_from_oca requires the stochvolmodels research extra: '
            'pip install "stochvolmodels[research]"'
        ) from error

    if not isinstance(chain, SlicesChain):
        raise TypeError('chain must be an OptionChainAnalytics SlicesChain')
    if value_time is None:
        value_time = chain.value_time
    value_time = pd.Timestamp(value_time)
    if days_map is None:
        days_map = {'1w': 7, '1m': 21}

    records = []
    seen_expiries = set()
    for label, day in days_map.items():
        next_date = value_time + pd.DateOffset(days=day)
        slice_date = chain.get_next_slice_after_date(mat_date=next_date)
        if slice_date in seen_expiries:
            continue
        slice_t = chain.expiry_slices[slice_date]
        df = slice_t.get_joint_slice(
            delta_bounds=delta_bounds,
            is_filtered=is_filtered,
        )
        if df.empty:
            continue

        if SliceColumn.DISCOUNT.value in df.columns:
            discounts = pd.to_numeric(
                df[SliceColumn.DISCOUNT.value],
                errors='coerce',
            ).to_numpy()
            discounts = discounts[np.isfinite(discounts) & (discounts > 0.0)]
            if discounts.size == 0:
                raise ValueError(f'missing positive discount factor for {slice_t.expiry_id}')
            discfactor = float(np.median(discounts))
        else:
            discfactor = 1.0

        records.append(
            {
                'id': f'{label}: {slice_t.expiry_id}',
                'ttm': slice_t.get_ttm(),
                'forward': slice_t.get_future_price(),
                'discfactor': discfactor,
                'strikes': df.index.to_numpy(),
                'optiontypes': df[SliceColumn.OPTION_TYPE].to_numpy(dtype=str),
                'bid_ivs': df[SliceColumn.BID_IV].to_numpy(),
                'ask_ivs': df[SliceColumn.ASK_IV].to_numpy(),
                'bid_prices': df[SliceColumn.BID_PRICE].to_numpy(),
                'ask_prices': df[SliceColumn.ASK_PRICE].to_numpy(),
            }
        )
        seen_expiries.add(slice_date)

    if not records:
        raise ValueError('no non-empty OCA maturity slices matched days_map')
    records.sort(key=lambda record: record['ttm'])

    strikes_ttms, optiontypes_ttms = List(), List()
    bid_ivs, ask_ivs = List(), List()
    bid_prices, ask_prices = List(), List()
    for record in records:
        strikes_ttms.append(record['strikes'])
        optiontypes_ttms.append(record['optiontypes'])
        bid_ivs.append(record['bid_ivs'])
        ask_ivs.append(record['ask_ivs'])
        bid_prices.append(record['bid_prices'])
        ask_prices.append(record['ask_prices'])

    return OptionChain(
        ttms=np.array([record['ttm'] for record in records]),
        forwards=np.array([record['forward'] for record in records]),
        discfactors=np.array([record['discfactor'] for record in records]),
        ids=np.array([record['id'] for record in records]),
        strikes_ttms=strikes_ttms,
        optiontypes_ttms=optiontypes_ttms,
        bid_ivs=bid_ivs,
        ask_ivs=ask_ivs,
        bid_prices=bid_prices,
        ask_prices=ask_prices,
    )
