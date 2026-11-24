"""Independent cashflow, information-timing and CMA mapping checks."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("papers.risk_premia_gmm.replication", reason="requires the GMM paper checkout")
from papers.risk_premia_gmm.replication import tlt_economic_comparison as study


def test_duration_scales_only_rates_excess_and_uses_holding_years():
    # Two units of the seven-year rates exposure, with one unit of financing/fees.
    days = 25
    expected = (2*.01+2*.002-.0015+.003)*days/365
    result = study.mapped_excess(.01, .002, 14., days/365, cash_bridge=.003)
    assert result == pytest.approx(expected, abs=1e-14)
    assert study.mapped_excess(0., 0., 21., days/365) == pytest.approx(-.0015*days/365)


def test_invalid_duration_or_horizon_rejected():
    for duration, years in [(-1., .1), (7., 0.), (7., 2.)]:
        with pytest.raises(ValueError):
            study.mapped_excess(.01, .01, duration, years)


def test_prior_information_excludes_current_and_future_rows():
    frame = pd.DataFrame({'a': [1., 2., 99.]}, index=pd.to_datetime([
        '2023-05-31', '2023-06-30', '2023-07-31']))
    row, date = study.prior_row(frame, '2023-06-30', 65)
    assert date == pd.Timestamp('2023-05-31')
    assert row.a == 1.
    frame.iloc[1:, 0] = -999
    assert study.prior_row(frame, '2023-06-30', 65)[0].a == 1.


def test_stale_or_unavailable_information_rejected():
    frame = pd.DataFrame({'a': [1.]}, index=pd.to_datetime(['2023-05-31']))
    with pytest.raises(ValueError, match='Stale'):
        study.prior_row(frame, '2023-09-30', 65)
    with pytest.raises(ValueError, match='No strictly'):
        study.prior_row(frame, '2023-05-31', 65)


def test_duration_and_convexity_against_independent_repricing():
    duration, maturity, rate = 16.5, 26., .045
    times, cash, convexity = study.coupon_proxy(duration, maturity, rate)
    price = lambda y: sum(float(c)/(1+y/2)**(2*float(t)) for t, c in zip(times, cash))
    dy = 1e-5
    p, up, down = price(rate), price(rate+dy), price(rate-dy)
    assert (down-up)/(2*dy*p) == pytest.approx(duration, rel=1e-7)
    assert (up+down-2*p)/(dy*dy*p) == pytest.approx(convexity, rel=2e-7)
    roll_shift = -.0001*25/365
    actual = price(rate+roll_shift)/p-1
    estimate = -duration*roll_shift+.5*convexity*roll_shift**2
    assert actual == pytest.approx(estimate, abs=1e-11)


def test_correlations_use_average_tied_ranks():
    # Ranks x=(1,2.5,2.5,4), y=(4,2,3,1), including one omitted missing pair.
    result = study.correlations([1., 2., 2., 4., np.nan], [5., 3., 4., 1., 6.])
    expected = np.dot([-1.5, 0., 0., 1.5], [1.5, -.5, .5, -1.5])/np.sqrt(4.5*5.)
    assert result['n'] == 4
    assert result['spearman'] == pytest.approx(expected)


def test_calendar_bootstrap_rejects_duplicate_months_and_preserves_inputs():
    dates = pd.to_datetime(['2023-01-31', '2023-03-31', '2023-04-30', '2023-07-31'])
    x, y = np.array([1., 3., 2., 4.]), np.array([5., 7., 6., 8.])
    original = x.copy()
    result = study.calendar_intervals(dates, x, y, 3, draws=100)
    assert result['pearson_low'] == pytest.approx(1.)
    assert result['spearman_high'] == pytest.approx(1.)
    np.testing.assert_array_equal(x, original)
    with pytest.raises(ValueError, match='one observation'):
        study.calendar_intervals([dates[0]]*4, x, y, 3, draws=10)


def test_monthly_source_reconciles_workbook_blend_and_percent_units(monkeypatch):
    dates = pd.to_datetime(['2023-03-31', '2023-06-30'])
    frame = lambda col, data: pd.DataFrame({col: data}, index=dates)
    sheets = {'term_premium': frame('US', [.01, .02]),
              'roll_down': frame('US', [.007, .014]),
              'rates_cma_excess': frame('US', [.016, .0245]),
              'rates_govt_3y': frame('US', [.04, .05]),
              'rates_cma_nominal': frame('US', [.057, .084]),
              'factor_vols': frame('Rates', [.06, .06]),
              'config_snapshot': pd.DataFrame({'value': [.5, .25]}, index=[
                  'blend_weight_current', 'sharpe_ratio.Rates']),
              'rates_US': pd.DataFrame({'yield_10y': [4., 5.], 'ois_5y5y': [3., 3.],
                  'yield_7y': [3.7, 4.4], 'yield_3y': [4., 5.]}, index=dates)}
    # The real workbook has dated leading rows with no rates components.
    for name in ('term_premium', 'roll_down', 'rates_cma_excess', 'rates_govt_3y',
                 'rates_cma_nominal'):
        sheets[name].loc[pd.Timestamp('2022-12-31')] = np.nan
    from contextlib import nullcontext
    monkeypatch.setattr(pd, 'ExcelFile', lambda _: nullcontext('book'))
    monkeypatch.setattr(pd, 'read_excel', lambda _, sheet_name, index_col: sheets[sheet_name].copy())
    monthly, _, checks = study.read_cma('book', 'prepared')
    np.testing.assert_allclose(monthly.current, [.017, .034], atol=1e-12)
    assert checks['quarters_reconstructed'] == 2
    assert checks['missing_quarters'] == ['2022-12-31']
    sheets['rates_cma_excess'].iloc[0, 0] += .001
    with pytest.raises(AssertionError):
        study.read_cma('book', 'prepared')


def test_panel_uses_saved_actual_holding_window_and_prior_duration(tmp_path):
    pilot, inputs = tmp_path/'pilot', tmp_path/'inputs'
    pilot.mkdir()
    inputs.mkdir()
    pd.DataFrame({'date': ['2023-06-30'], 'rate': [.05], 'days': [28]}).to_csv(
        pilot/'selection.csv', index=False)
    pd.DataFrame({'date': ['2023-06-30'], 'requested_asof': ['2023-06-29'],
        'returned_asof': ['2023-06-29'], 'duration': [16.5], 'coverage': [1.]}).to_csv(
            inputs/'historical_duration.csv', index=False)
    pd.DataFrame({'WEIGHTED_AVERAGE_MATURITY_YEARS': [26.]},
        index=['2023-06-29']).to_csv(inputs/'fund_history.csv')
    pd.DataFrame({'USGG20YR Index': [4.5], 'USGG30YR Index': [4.6]},
        index=['2023-06-29']).to_csv(inputs/'curve_history.csv')
    pd.DataFrame({'date': ['2023-06-30'], 'components': [3], 'entry_date': ['2023-07-03'],
        'end_date': ['2023-07-28'], 'method': ['fixed_3'], 'holding_years': [25/365],
        'forecast': [.004]}).to_csv(pilot/'return_forecasts.csv', index=False)
    monthly = pd.DataFrame({'term': [.01], 'roll': [.002], 'govt_3y': [.04], 'current': [.012]},
                            index=pd.to_datetime(['2023-05-31']))
    quarterly = monthly.copy()
    quarterly.index = pd.to_datetime(['2023-03-31'])
    quarterly['blended'] = .013
    _, panel = study.build_panel(pilot, inputs, monthly, quarterly)
    row = panel.iloc[0]
    assert row.monthly_current_holding == pytest.approx((16.5/7*.012-.0015)*25/365)
    assert row.option_annual == pytest.approx(.004*365/25)
    assert row.cma_date == pd.Timestamp('2023-05-31')
