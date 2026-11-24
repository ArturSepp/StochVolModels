"""Independent par repricing, timing, coverage and fixed-prior checks for GLD."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("papers.risk_premia_gmm.replication", reason="requires the GMM paper checkout")
from papers.risk_premia_gmm.replication import gld_treasury as curve
from papers.risk_premia_gmm.replication import gld_wing_study as wings


def test_bootstrap_reprices_par_bonds_and_flat_curve_closed_form():
    t, d, y = curve.bootstrap([4., 8., 6.])
    assert np.array_equal(t, [0., .5, 1., 1.5, 2.])
    assert np.allclose(y, [.04, .08, .07, .06], rtol=0, atol=1e-15)
    # Independent cash-flow dot products, including the interpolated 18M instrument.
    for j, coupon in enumerate([.02, .04, .035, .03], 1):
        cash = np.full(j, coupon)
        cash[-1] += 1
        assert cash@d[1:j+1] == pytest.approx(1., abs=1e-14)
    t, d, _ = curve.bootstrap([5., 5., 5.])
    assert np.allclose(d, 1.025**(-2*t), rtol=0, atol=1e-15)
    # A par yield is not a continuously compounded zero rate.
    assert abs(d[2]-np.exp(-.05)) > .0005


def test_prior_curve_roll_excludes_signal_day_and_prohibits_stale_extrapolation():
    frame = pd.DataFrame({'date': ['2023-06-29', '2023-06-30'],
        'par_6m': [5., 20.], 'par_1y': [5., 20.], 'par_2y': [5., 20.]})
    row = curve.funding(frame, '2023-06-30', '2024-06-21')
    assert row['rate_information_date'] == '2023-06-29'
    assert row['rate'] == pytest.approx(2*np.log(1.025), abs=1e-14)
    assert row['discount'] == pytest.approx(1.025**(-2*357/365), abs=1e-14)
    with pytest.raises(ValueError, match='stale'):
        curve.funding(frame, '2023-07-10', '2024-06-21')
    with pytest.raises(ValueError, match='outside'):
        curve.funding(frame, '2023-06-30', '2026-06-21')
    with pytest.raises(ValueError, match='Duplicate'):
        curve.funding(pd.concat([frame, frame]), '2023-06-30', '2024-06-21')


def test_roll_uses_discount_ratio_on_a_sloped_curve():
    frame = pd.DataFrame({'date': ['2023-06-29'], 'par_6m': [4.],
                          'par_1y': [5.], 'par_2y': [6.]})
    row = curve.funding(frame, '2023-06-30', '2024-06-21')
    d6 = 1/1.02
    d12 = (1-.025*d6)/1.025
    a, b = 1/365, 358/365
    start = d6**(a/.5)
    end = d6**((1-b)/.5)*d12**((b-.5)/.5)
    assert row['discount'] == pytest.approx(end/start, abs=1e-14)


def quotes():
    strikes = np.arange(40., 201., 5.)
    return pd.DataFrame({'strike': np.r_[strikes, strikes],
        'optiontype': ['P']*len(strikes)+['C']*len(strikes),
        'bid_price': 1., 'ask_price': 2., 'bid_size': 1, 'ask_size': 1,
        'bid_condition': 50, 'ask_condition': 50})


def test_volatility_band_uses_sqrt_time_is_nested_and_forward_centered():
    q = quotes()
    job = {'spot': 100., 'forward': 105., 'band_sigma': 2., 'anchor_atm': .25, 'ttm': .25}
    mask, n = wings.screen(q, job)
    assert n['band_low'] == pytest.approx(105*np.exp(-.25))
    assert n['band_high'] == pytest.approx(105*np.exp(.25))
    old, _ = wings.screen(q, {**job, 'band_sigma': 0.})
    outer, _ = wings.screen(q, {**job, 'band_sigma': 2.5})
    assert (mask | ~old).all() and (outer | ~mask).all()
    assert q[mask & q.optiontype.eq('C')].strike.min() >= 105
    assert q[mask & q.optiontype.eq('P')].strike.max() < 105
    _, tiny = wings.screen(q, {**job, 'anchor_atm': .01})
    assert tiny['band_low'] == 85. and tiny['band_high'] == pytest.approx(115.)


def test_widening_never_relaxes_bid_size_condition_or_spread():
    q = quotes()
    job = {'spot': 100., 'forward': 105., 'band_sigma': 2., 'anchor_atm': .4, 'ttm': 1.}
    row = q.index[q.strike.eq(80.) & q.optiontype.eq('P')][0]
    assert wings.screen(q, job)[0][row]
    for field, value in [('bid_price', 0.), ('ask_price', 4.), ('bid_size', 0),
                         ('ask_size', 0), ('bid_condition', 0), ('ask_condition', 0)]:
        changed = q.copy()
        changed.loc[row, field] = value
        assert not wings.screen(changed, job)[0][row]
    q.loc[row, ['bid_price', 'ask_price']] = [1., 3.]
    assert wings.screen(q, job)[0][row]  # Exactly 100% of midpoint.


def test_frozen_anchor_reconstructs_known_smile_not_new_wing_maximum():
    k = np.array([.9, .98, 1.02, 1.1])
    t, f, s, vol = .75, 105., 100., .2
    x = np.array([0., 0., np.log(vol), np.log(vol)])
    prices = wings.old.pilot.base.mixture_prices(x, 2, t, k, np.where(k >= 1, 1., -1.))
    q = pd.DataFrame({'date': '2023-06-30', 'accepted': True, 'strike': k*f,
                      'forward': f, 'spot': s, 'ttm': t, 'european_mid': s*prices})
    anchor = wings.anchors(q)['2023-06-30']
    assert anchor['anchor_atm'] == pytest.approx(vol, abs=1e-7)
    assert anchor['volcap'] == pytest.approx(1.2*vol, abs=1e-7)
    assert anchor['logcap'] == pytest.approx(20*vol**2*t, abs=1e-7)


def test_xml_rejects_wrong_year_and_duplicate_records():
    properties = ('<properties><NEW_DATE>2023-06-29</NEW_DATE><BC_6MONTH>5</BC_6MONTH>'
                  '<BC_1YEAR>4.5</BC_1YEAR><BC_2YEAR>4</BC_2YEAR></properties>')
    assert len(curve.parse_xml(('<feed>'+properties+'</feed>').encode(), 2023)) == 1
    with pytest.raises(ValueError, match='wrong year'):
        curve.parse_xml(('<feed>'+properties+'</feed>').encode(), 2024)
    with pytest.raises(ValueError, match='duplicate'):
        curve.parse_xml(('<feed>'+2*properties+'</feed>').encode(), 2023)
