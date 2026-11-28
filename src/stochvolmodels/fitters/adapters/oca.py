"""Convert point-in-time OptionChainAnalytics objects into SVM calibration inputs.

Observation-time reconstruction and frequency sampling remain owned by
OptionChainAnalytics. This optional adapter only selects expiries and quotes inside one
already-reconstructed ``SlicesChain`` and materializes an SVM ``OptionChain``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from numba.typed import List

from stochvolmodels.data.option_chain import OptionChain

if TYPE_CHECKING:
    from option_chain_analytics import SlicesChain


def option_chain_from_oca(
    chain: SlicesChain,
    value_time: Optional[pd.Timestamp] = None,
    days_map: Optional[Dict[str, int]] = None,
    delta_bounds: Tuple[Optional[float], Optional[float]] | Literal['default'] | None = 'default',
    is_filtered: bool = True,
    *,
    iv_source: Literal['auto', 'bid_ask', 'mark'] = 'auto',
) -> OptionChain:
    """Convert one reconstructed OCA chain into SVM calibration inputs.

    Parameters
    ----------
    chain : option_chain_analytics.SlicesChain
        Point-in-time chain reconstructed by OCA.
    value_time : pandas.Timestamp, optional
        Anchor for maturity targets. Defaults to the chain's actual observation time.
    days_map : dict[str, int], optional
        Calibration maturity labels and calendar-day offsets. Defaults to every
        available expiry for marks and to 7 and 21 days for two-sided IVs.
    delta_bounds : tuple, None or 'default'
        OCA delta filter. Defaults to (-0.1, 0.1) for two-sided inputs and all
        available pillars for mark inputs.
    is_filtered : bool, default=True
        Apply OCA's quote filtering before conversion.
    iv_source : {'auto', 'bid_ask', 'mark'}
        Auto uses marks for a snapshot with no quoted IV sides. Mark mode
        uses all available expiries by default and stores only positive mark
        IVs, with absent bid and ask fields. Two-sided selection is unchanged.

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
    if iv_source not in ('auto', 'bid_ask', 'mark'):
        raise ValueError("iv_source must be 'auto', 'bid_ask' or 'mark'")
    sides = chain.options_df.reindex(columns=[SliceColumn.BID_IV, SliceColumn.ASK_IV])
    mark_only = iv_source == 'mark' or (iv_source == 'auto' and not sides.notna().any().any())
    if delta_bounds == 'default':
        delta_bounds = None if mark_only else (-0.1, 0.1)
    if days_map is None and mark_only:
        selections = [(None, expiry) for expiry in chain.expiry_slices]
    else:
        if days_map is None:
            days_map = {'1w': 7, '1m': 21}
        selections = [(label, chain.get_next_slice_after_date(value_time + pd.DateOffset(days=day)))
                      for label, day in days_map.items()]

    records = []
    seen_expiries = set()
    for label, slice_date in selections:
        if slice_date in seen_expiries:
            continue
        slice_t = chain.expiry_slices[slice_date]
        if mark_only:
            df = slice_t.options_df
            if is_filtered:
                mids = pd.to_numeric(df[SliceColumn.MARK_IV], errors='coerce')
                df = df.loc[np.isfinite(mids) & (mids > 0)]
            if delta_bounds is not None:
                low, high = delta_bounds
                put = df[SliceColumn.OPTION_TYPE] == 'P'
                if low is not None:
                    df = df.loc[~put | (df[SliceColumn.DELTA] <= low)]
                if high is not None:
                    df = df.loc[(df[SliceColumn.OPTION_TYPE] == 'P') | (df[SliceColumn.DELTA] >= high)]
            # Prefer the OTM quote if both sides of parity exist at a strike.
            otm_type = np.where(df[SliceColumn.STRIKE] < slice_t.get_future_price(), 'P', 'C')
            df = (df.assign(_otm=df[SliceColumn.OPTION_TYPE] == otm_type)
                  .sort_values([SliceColumn.STRIKE, '_otm'], kind='stable')
                  .drop_duplicates(SliceColumn.STRIKE, keep='last').set_index(SliceColumn.STRIKE))
        else:
            df = slice_t.get_joint_slice(delta_bounds=delta_bounds, is_filtered=is_filtered)
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
                'id': slice_t.expiry_id if label is None else f'{label}: {slice_t.expiry_id}',
                'ttm': slice_t.get_ttm(),
                'forward': slice_t.get_future_price(),
                'discfactor': discfactor,
                'strikes': df.index.to_numpy(),
                'optiontypes': df[SliceColumn.OPTION_TYPE].to_numpy(dtype=str),
                **({'mid_ivs': df[SliceColumn.MARK_IV].to_numpy(dtype=float)} if mark_only else {
                    'bid_ivs': df[SliceColumn.BID_IV].to_numpy(),
                    'ask_ivs': df[SliceColumn.ASK_IV].to_numpy(),
                    'bid_prices': df[SliceColumn.BID_PRICE].to_numpy(),
                    'ask_prices': df[SliceColumn.ASK_PRICE].to_numpy(),
                }),
            }
        )
        seen_expiries.add(slice_date)

    if not records:
        raise ValueError('no non-empty OCA maturity slices matched days_map')
    records.sort(key=lambda record: record['ttm'])

    fields = ('mid_ivs',) if mark_only else ('bid_ivs', 'ask_ivs', 'bid_prices', 'ask_prices')

    return OptionChain(
        ttms=np.array([record['ttm'] for record in records]),
        forwards=np.array([record['forward'] for record in records]),
        discfactors=np.array([record['discfactor'] for record in records]),
        ids=np.array([record['id'] for record in records]),
        strikes_ttms=List(record['strikes'] for record in records),
        optiontypes_ttms=List(record['optiontypes'] for record in records),
        **{field: List(record[field] for record in records) for field in fields},
    )
