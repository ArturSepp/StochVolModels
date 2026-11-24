"""Diagnostic relaxation must not hide missing, crossed or zero-bid observations."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("papers.risk_premia_gmm.replication", reason="requires the GMM paper checkout")
from papers.risk_premia_gmm.replication.hyg_quote_diagnostics import quote_masks


def test_width_rules_are_explicit_and_do_not_fill_missing_quotes():
    frame=pd.DataFrame({'strike':[76.]*5,'optiontype':['C']*5,
                       'bid_price':[.04,.01,0.,np.nan,.1],
                       'ask_price':[.07,.08,.04,.1,.09]})
    expected={'relative_50':[False]*5,'relative_100':[True,False,False,False,False],
              'absolute_10c':[True,True,False,False,False]}
    for rule,mask in expected.items():
        assert quote_masks(frame,75.,75.,rule)['OTM'].tolist()==mask


def test_forward_determines_option_side_and_spot_determines_strike_band():
    frame=pd.DataFrame({'strike':[74.75,74.75,75.,75.,90.],
        'optiontype':['C','P','C','P','C'],'bid_price':[1.]*5,'ask_price':[1.1]*5})
    mask=quote_masks(frame,75.,74.5,'relative_50')
    assert mask['OTM'].tolist()==[True,False,True,False,False]
    with pytest.raises(ValueError):
        quote_masks(frame,75.,74.5,'unknown')
