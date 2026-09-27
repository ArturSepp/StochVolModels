"""Canonical script of docs/app_swaptions_and_sofr_options.md.

Every Python block on that page is a verbatim excerpt of this file, and every number the page
quotes is asserted here. The Nelson-Siegel HJM model with a log-normal stochastic volatility driver
of Sepp and Rakhmonov (2025) is evaluated on the USD swaption surface and the 3M SOFR options
hard-coded in ``papers/sv_for_factor_hjm``, with the parameters those modules record. The Monte
Carlo cases take minutes and are listed in ``SLOW_CASES``. ``stochvolmodels.pricers.factor_hjm`` is
an experimental research surface. Select a case in ``Locals`` or run the file to execute every case.
"""
import sys
from enum import Enum
from pathlib import Path

import numpy as np
import vanilla_option_pricers as bachelier

from stochvolmodels import ExpansionOrder, LogSvParams
from stochvolmodels.pricers.factor_hjm.factor_hjm_pricer import calc_mc_vols, do_mc_simulation
from stochvolmodels.pricers.factor_hjm.rate_affine_expansion import UnderlyingType
from stochvolmodels.pricers.factor_hjm.rate_logsv_pricer import (FutSettleType, Measure,
                                                                 calc_futures_rate,
                                                                 logsv_chain_de_pricer)
from stochvolmodels.utils.rate_core import (generate_ttms_grid, get_default_swap_term_structure,
                                            get_futures_start_and_pmt)

# the market data are hard-coded in the paper's modules, which live in the repository, not the
# package; put the repository root on the path so that they import from a checkout
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from papers.sv_for_factor_hjm import calibration_fig_5_6_7 as swaption_module  # noqa: E402
from papers.sv_for_factor_hjm import calibration_fig_8_9 as sofr_module  # noqa: E402

SLOW_CASES = ("SWAPTION_FIT", "SWAPTION_MONTE_CARLO", "SHOCK_SCENARIOS")
EXPIRY_IDS = ["1y", "2y", "3y", "5y"]
# Table 1 of the published article, per expiry: yield vols a, betas and beta0, with kappa2 = 0.5
TABLE_1 = {"1y": ([0.0143, 0.0129, 0.0113], [0.0705, -0.3236, 0.7477], 0.0300),
           "2y": ([0.0128, 0.0129, 0.0112], [0.5837, 0.5938, -0.5586], 0.1067),
           "3y": ([0.0113, 0.0132, 0.0113], [-0.0093, 0.3647, -0.2500], 0.0757),
           "5y": ([0.0055, 0.0097, 0.0086], [0.7293, -0.4866, -0.1628], 0.0715)}
# Table 3 of the published article, per expiry: yield vols a, beta of (beta, -beta / 2, 0), beta0
TABLE_3 = {"75d": ([0.0049, 0.0024, 0.0005], -1.995, 1.449),
           "103d": ([0.0041, 0.0017, 0.0039], -1.005, 0.213)}


def swaption_chain():
    """USD swaptions of 18 August 2023: tenors 2y, 5y, 10y; expiries 1y to 5y; five strikes each."""
    return swaption_module.get_swaption_data().reduce_tenors(["2y", "5y", "10y"]) \
        .reduce_strikes(2).reduce_ttms(EXPIRY_IDS)


def swaption_params(published: bool = False):
    """Parameters recorded in the module, or those of Table 1 of the article."""
    params = swaption_module.getCalibRateLogSVParams()["USD"]
    if published:
        for idx, expiry in enumerate(EXPIRY_IDS):
            a, beta, beta0 = TABLE_1[expiry]
            params.update_params(idx=idx, A_idx=np.array(a), beta_idx=np.array(beta),
                                 volvol_idx=beta0, kappa2=0.5)
    return params


def swaption_fit(params, chain) -> list:
    """Model normal volatilities of every slice, [tenor][expiry], by the first-order expansion."""
    t_grid = generate_ttms_grid(chain.ttms)
    x0, y0 = np.zeros(params.basis.get_nb_factors()), np.zeros(params.basis.get_nb_aux_factors())
    vols = [[] for _ in chain.tenors]
    for idx, expiry in enumerate(chain.ttms):
        _, slice_vols = logsv_chain_de_pricer(
            params=params, t_grid=t_grid, ttms=np.array([expiry]),
            forwards=[forwards[[idx]] for forwards in chain.forwards],
            strikes_ttms=[strikes[idx:idx + 1] for strikes in chain.strikes_ttms],
            optiontypes_ttms=[chain.optiontypes_ttms[idx]], expansion_order=ExpansionOrder.FIRST,
            x0=x0, y0=y0)
        for j, tenor_vols in enumerate(slice_vols):
            vols[j].append(np.asarray(tenor_vols[0]))
    return vols


def fit_errors(params, chain) -> np.ndarray:
    """Model minus market normal volatility, in bp, [tenor, expiry, strike]."""
    model = swaption_fit(params, chain)
    return 1e4 * np.array([[m - market for m, market in zip(model_tenor, market_tenor)]
                           for model_tenor, market_tenor in zip(model, chain.bid_ivs)])


def pooled_mc_vols(params, expiry: float, tenors: np.ndarray, strikes: list,
                   nb_batch: int = 20, nb_path_batch: int = 10000) -> dict:
    """calc_mc_vols in batches of paths with seeds 1, 2, ..., so that 200,000 paths fit in memory.

    The simulation (risk-neutral, 360 steps a year), payoff and Bachelier inversion are those of
    calc_mc_vols; with one batch and seed 16 the result equals calc_mc_vols.
    """
    sums, squares = [np.zeros_like(k) for k in strikes], [np.zeros_like(k) for k in strikes]
    level, level2, forwards = np.zeros(len(tenors)), np.zeros(len(tenors)), np.zeros(len(tenors))
    for seed in (range(1, nb_batch + 1) if nb_batch > 1 else [16]):
        x0 = np.zeros((nb_path_batch, params.basis.get_nb_factors()))
        y0 = np.zeros((nb_path_batch, params.basis.get_nb_aux_factors()))
        xs, ys, integrals, _ = do_mc_simulation(
            basis_type="NELSON-SIEGEL", ccy=params.ccy, ttms=np.array([expiry]), x0=x0, y0=y0,
            I0=np.zeros(nb_path_batch), sigma0=np.ones((nb_path_batch, 1)), params=params,
            nb_path=nb_path_batch, seed=seed, measure_type=Measure.RISK_NEUTRAL)
        for j, tenor in enumerate(tenors):
            ts_sw = get_default_swap_term_structure(expiry=expiry, tenor=tenor)
            annuity0 = params.basis.annuity(t=expiry, ts_sw=ts_sw, x=x0, y=y0, ccy=params.ccy,
                                            m=0)[0]
            bond0 = params.basis.bond(0, expiry, x=x0, y=y0, ccy=params.ccy, m=0)[0]
            forwards[j] = params.basis.swap_rate(t=expiry, ts_sw=ts_sw, x=x0, y=y0,
                                                 ccy=params.ccy)[0][0]
            swap, annuity, numeraire = params.basis.calculate_swap_rate(
                ttm=expiry, x0=xs[-1], y0=ys[-1], I0=integrals[-1], ts_sw=ts_sw, ccy=params.ccy)
            weight = annuity / numeraire / annuity0 / bond0  # annuity-measure density
            level[j] += np.nansum(weight * swap)
            level2[j] += np.nansum((weight * swap) ** 2)
            for i, strike in enumerate(strikes[j]):
                payoff = weight * np.maximum(swap - strike, 0.0)
                sums[j][i] += np.nansum(payoff)
                squares[j][i] += np.nansum(payoff ** 2)
    nb_path = nb_path_batch * (nb_batch if nb_batch > 1 else 1)
    out = {"mid": [], "up": [], "down": [], "forward": forwards,
           "simulated_forward": level / nb_path,
           "simulated_forward_se": np.sqrt((level2 / nb_path - (level / nb_path) ** 2) / nb_path)}
    for j, k in enumerate(strikes):
        mean = sums[j] / nb_path
        error = 1.96 * np.sqrt(np.maximum(squares[j] / nb_path - mean ** 2, 0.0) / nb_path)
        prices = {"mid": mean, "up": mean + error, "down": np.maximum(mean - error, 0.0)}
        for key, price in prices.items():
            out[key].append(bachelier.infer_normal_ivols_from_chain_prices(
                ttms=np.array([expiry]), forwards=np.array([forwards[j]]), discfactors=np.ones(1),
                strikes_ttms=[k], optiontypes_ttms=[np.repeat("C", k.size)],
                model_prices_ttms=[price])[0])
    return out


def expansion_vs_monte_carlo(params, expiry: float, nb_batch: int = 20) -> dict:
    """The expansion against pooled Monte Carlo on 21 strikes spanning each tenor's market range."""
    chain = swaption_chain()
    idx = int(np.argmin(np.abs(chain.ttms - expiry)))
    strikes = [np.linspace(k[idx][0], k[idx][-1], 21) for k in chain.strikes_ttms]
    mc = pooled_mc_vols(params, expiry, chain.tenors, strikes, nb_batch=nb_batch)
    _, vols = logsv_chain_de_pricer(
        params=params, t_grid=generate_ttms_grid(np.array([expiry])), ttms=np.array([expiry]),
        forwards=[np.array([f]) for f in mc["forward"]], strikes_ttms=[[k] for k in strikes],
        optiontypes_ttms=[np.repeat("C", 21)], expansion_order=ExpansionOrder.FIRST)
    mc["strikes"], mc["expansion"] = strikes, [np.asarray(v[0]) for v in vols]
    mc["inside"] = [int(np.sum((e >= d) & (e <= u)))
                    for e, u, d in zip(mc["expansion"], mc["up"], mc["down"])]
    return mc


def sofr_data():
    """3M SOFR options with 75 and 103 days to expiry: the raw quotes and their SABR refit."""
    raw = sofr_module.get_futures_data()
    fit, _ = sofr_module.refit_to_sabr(futoption_chain=raw)
    return raw, fit


def sofr_params(published: bool = False):
    """Parameters recorded in the module, or those of Table 3 of the article."""
    params = sofr_module.getFutCalibRateLogSVParams()["USD"].reduce(["75d", "103d"])
    params.q = params.theta  # as in the module
    if published:
        for idx, expiry in enumerate(["75d", "103d"]):
            a, beta, beta0 = TABLE_3[expiry]
            params.update_params(idx=idx, A_idx=np.array(a), volvol_idx=beta0,
                                 beta_idx=np.array([beta, -0.5 * beta, 0.0]))
    return params


def sofr_model_vols(params, expiry: float, forward: float, strikes: np.ndarray,
                    order: ExpansionOrder = ExpansionOrder.FIRST) -> np.ndarray:
    """Normal volatilities of calls on the 3M SOFR futures rate by the affine expansion.

    The time grid spans both expiries, so that it contains the date at which the parameters change.
    """
    _, vols = logsv_chain_de_pricer(
        params=params, t_grid=generate_ttms_grid(params.ts[1:]), ttms=np.array([expiry]),
        forwards=[[forward]], strikes_ttms=[[strikes]],
        optiontypes_ttms=[np.repeat("C", strikes.size)],
        is_stiff_solver=True, expansion_order=order, underlying_type=UnderlyingType.FUTURES, lag=0)
    return np.asarray(vols[0][0])


def sofr_fit_quality(params) -> dict:
    """Errors against the SABR refit at five deltas and against the raw quotes, in bp."""
    raw, fit = sofr_data()
    out = {"rms_refit": [], "inside": [], "rms_quotes": []}
    for idx, expiry in enumerate(fit.ttms):
        model = sofr_model_vols(params, expiry, fit.forwards[idx], fit.strikes_ttms[idx])
        up, down = one_tick_band(expiry, fit.forwards[idx], fit.strikes_ttms[idx],
                                 fit.ivs_call_ttms[idx])
        quotes = sofr_model_vols(params, expiry, raw.forwards[idx], raw.strikes_ttms[idx])
        out["rms_refit"].append(1e4 * np.sqrt(np.mean((model - fit.ivs_call_ttms[idx]) ** 2)))
        out["inside"].append(int(np.sum((model >= down) & (model <= up))))
        out["rms_quotes"].append(1e4 * np.sqrt(np.mean((quotes - raw.ivs_call_ttms[idx]) ** 2)))
    return out


def moment_stability(params) -> np.ndarray:
    """Largest real part of the eigenvalues of the four-moment system, per expiry.

    Footnote 5 of the article requires every real part to be negative.
    """
    out = []
    for idx in range(params.ts.size - 1):
        vartheta = np.sqrt(params.beta.xs[idx] @ params.beta.xs[idx] + params.volvol.xs[idx] ** 2)
        driver = LogSvParams(sigma0=params.sigma0, theta=params.theta, kappa1=params.kappa1,
                             kappa2=params.kappa2, beta=0.0, volvol=vartheta)
        out.append(np.max(np.real(np.linalg.eigvals(driver.get_vol_moments_lambda(n_terms=4)))))
    return np.array(out)

def one_tick_band(expiry: float, forward: float, strikes: np.ndarray, vols: np.ndarray,
                  tick: float = 0.25e-4) -> tuple:
    """Normal volatilities of the option prices one tick above and below the quote."""
    prices = np.array([bachelier.compute_normal_price(forward=forward, strike=k, ttm=expiry,
                                                      vol=v, optiontype="C")
                       for k, v in zip(strikes, vols)])
    types = np.repeat("C", strikes.size)
    up, down = (bachelier.infer_normal_ivols_from_slice_prices(
        ttm=expiry, forward=forward, strikes=strikes, model_prices=p, optiontypes=types,
        discfactor=1.0) for p in (prices + tick, np.maximum(prices - tick, 0.0)))
    return up, down


def sofr_benchmark(params, fit, idx: int, nb_path: int = 2 ** 17, seed: int = 20) -> dict:
    """First- and second-order expansions against the module's Monte Carlo, 21 strikes.

    As in the module, the strikes are re-centred on the model futures rate of the expiry.
    """
    expiry = fit.ttms[idx]
    start, end = get_futures_start_and_pmt(t0=expiry, lag=0.0, libor_tenor=0.25)
    futures_rate = calc_futures_rate(
        ccy=params.ccy, basis_type="NELSON-SIEGEL", params=params,
        x0=np.zeros(params.basis.get_nb_factors()), y0=np.zeros(params.basis.get_nb_aux_factors()),
        sigma0=np.ones((1, 1)), t0=0.0, t_start=start, t_end=end, Delta=0.25,
        expansion_order=ExpansionOrder.ZERO, settlement_type=FutSettleType.SOFR)[0][0]
    shifted = fit.strikes_ttms[idx] - fit.forwards[idx] + futures_rate
    strikes = np.linspace(shifted[0], shifted[-1], 21)
    _, mid, up, down = sofr_module.calc_mc_vols(
        basis_type="NELSON-SIEGEL", params=params, ttm=expiry, lag=0.0, forward=futures_rate,
        strikes=strikes, optiontypes=np.repeat("P", 21), measure_type=Measure.FORWARD,
        T_fwd=expiry, nb_path=nb_path, seed=seed)
    out = {"futures_rate": futures_rate, "strikes": strikes, "mid": np.asarray(mid[0]),
           "up": np.asarray(up[0]), "down": np.asarray(down[0])}
    for order in (ExpansionOrder.FIRST, ExpansionOrder.SECOND):
        vols = np.asarray(logsv_chain_de_pricer(
            params=params, t_grid=generate_ttms_grid(params.ts[1:]), ttms=np.array([expiry]),
            forwards=[[futures_rate]], strikes_ttms=[[strikes]],
            optiontypes_ttms=[np.repeat("P", 21)],
            is_stiff_solver=True, expansion_order=order, underlying_type=UnderlyingType.FUTURES,
            lag=0)[1][0][0])
        out[order.name] = vols
        out[f"inside_{order.name}"] = int(np.sum((vols >= out["down"]) & (vols <= out["up"])))
    return out


class Locals(Enum):
    """Cases of this script; each asserts the numbers the page quotes."""
    SWAPTION_DATA = 1
    PARAMETERS = 2
    SWAPTION_FIT = 3
    POOLING = 4
    SWAPTION_MONTE_CARLO = 5
    SHOCK_SCENARIOS = 6
    SOFR_FIT = 7
    SOFR_MONTE_CARLO = 8


def run_local(local: Locals) -> None:
    """Run one case and assert its documented results."""
    if local == Locals.SWAPTION_DATA:
        chain = swaption_chain()
        assert chain.ttms.tolist() == [1.0, 2.0, 3.0, 5.0]
        assert chain.tenors.tolist() == [2.0, 5.0, 10.0]
        assert np.array(chain.strikes_ttms).shape == (3, 4, 5)  # 60 options
        # the module replaces the quoted forwards by those of its flat 4.3% curve, keeping the
        # distance of each strike to the forward
        forward = np.exp(0.043) - 1.0
        assert all(np.allclose(f, forward, rtol=1e-12) for f in chain.forwards)
        assert all(np.allclose(k[:, 2], forward, rtol=1e-12) for k in np.array(chain.strikes_ttms))
        # bid and ask are the same quote: the surface carries no spreads
        assert all(np.array_equal(b, a) for b, a in zip(chain.bid_ivs, chain.ask_ivs))

    elif local == Locals.PARAMETERS:
        recorded, published = swaption_params(), swaption_params(published=True)
        assert (recorded.kappa1, recorded.kappa2, published.kappa2) == (0.25, 0.25, 0.5)
        np.testing.assert_allclose(np.round(recorded.beta.xs[0], 4), [0.0152, 0.1063, 0.6667])
        np.testing.assert_allclose(sofr_params().beta.xs[0][0], -0.567)
        # every expiry of the recorded parameters differs from Table 1 at four decimals
        for idx in range(4):
            same = [np.allclose(np.round(x[idx], 4), y[idx], atol=1e-9) for x, y in (
                (recorded.A, published.A), (recorded.beta.xs, published.beta.xs),
                (recorded.volvol.xs, published.volvol.xs))]
            assert not all(same)
        chain = swaption_chain()
        assert all(recorded.check_QA_kappa2(expiry=t, tenor=s)
                   for t in chain.ttms for s in chain.tenors)
        # footnote 5: the truncated moment system of the SOFR parameters is not stable
        np.testing.assert_allclose(moment_stability(sofr_params()), [3.798, 0.030], atol=1e-3)

    elif local == Locals.SWAPTION_FIT:
        chain = swaption_chain()
        rms = {published: np.sqrt(np.mean(fit_errors(swaption_params(published), chain) ** 2,
                                           axis=(1, 2))) for published in (False, True)}
        print(rms)
        np.testing.assert_allclose(rms[False], [1.15, 0.48, 0.70], atol=0.01)
        np.testing.assert_allclose(rms[True], [2.29, 1.38, 1.95], atol=0.01)
        errors = fit_errors(swaption_params(), chain)
        np.testing.assert_allclose(np.max(np.abs(errors)), 3.35, atol=0.01)
        assert np.unravel_index(np.argmax(np.abs(errors)), errors.shape) == (0, 1, 0)

    elif local == Locals.POOLING:
        chain = swaption_chain()
        params = swaption_params()
        strikes = [np.linspace(k[3][0], k[3][-1], 5) for k in chain.strikes_ttms]
        pooled = pooled_mc_vols(params, 5.0, chain.tenors, strikes, nb_batch=1, nb_path_batch=5000)
        _, mid, _, _ = calc_mc_vols(
            basis_type="NELSON-SIEGEL", params=params, ttm=5.0, tenors=chain.tenors,
            forwards=[np.array([f]) for f in pooled["forward"]],
            strikes_ttms=[[k] for k in strikes],
            optiontypes=np.repeat("C", 5), is_annuity_measure=False, nb_path=5000)
        # one batch with calc_mc_vols's fixed seed of 16 reproduces it
        np.testing.assert_allclose(np.array(pooled["mid"]), np.array(mid), atol=1e-15)

    elif local == Locals.SWAPTION_MONTE_CARLO:
        result = expansion_vs_monte_carlo(swaption_params(), expiry=5.0)
        gaps = [1e4 * (e - m) for e, m in zip(result["expansion"], result["mid"])]
        shortfall = 1e4 * (result["simulated_forward"] - result["forward"])
        print(result["inside"], gaps, shortfall, 1e4 * result["simulated_forward_se"])
        # at 5y the expansion leaves the 95% interval for the 10y tenor at every strike
        assert result["inside"] == [19, 15, 0]
        np.testing.assert_allclose([gaps[2].min(), gaps[2].max()], [0.55, 1.46], atol=0.01)
        # the simulated forward swap rate is below the model's by 1.3 to 2.1 standard errors
        np.testing.assert_allclose(shortfall, [-0.66, -0.70, -0.72], atol=0.01)
        np.testing.assert_allclose(shortfall / (1e4 * result["simulated_forward_se"]),
                                   [-1.3, -1.6, -2.1], atol=0.05)

    elif local == Locals.SHOCK_SCENARIOS:
        inside, gaps = [], []
        for scenario in [(1.0, 1.0, 0.0), (1.0, 1.0, 0.02), (1.0, 4.0, 0.0), (-2.0, 1.0, 0.0)]:
            result = expansion_vs_monte_carlo(swaption_module.get_scenarios(*scenario), expiry=2.0)
            inside.append(result["inside"])
            gaps.append(max(1e4 * np.mean(e - m)
                            for e, m in zip(result["expansion"], result["mid"])))
        print(inside, gaps)
        # Table 2: base, a + 0.02, beta0 x 4, beta x -2; the last two leave the interval
        assert inside == [[21, 21, 21], [21, 21, 21], [9, 11, 10], [9, 7, 7]]
        np.testing.assert_allclose(gaps, [0.39, 0.59, 0.94, 0.71], atol=0.01)

    elif local == Locals.SOFR_FIT:
        quality = {published: sofr_fit_quality(sofr_params(published))
                   for published in (False, True)}
        print(quality)
        np.testing.assert_allclose(quality[False]["rms_refit"], [0.03, 0.48], atol=0.01)
        assert quality[False]["inside"] == [5, 5]
        np.testing.assert_allclose(quality[False]["rms_quotes"], [1.46, 2.07], atol=0.01)
        np.testing.assert_allclose(quality[True]["rms_refit"], [18.84, 12.33], atol=0.01)
        raw, fit = sofr_data()
        assert [k.size for k in raw.strikes_ttms] == [17, 15]
        np.testing.assert_allclose(1e4 * fit.forwards, 432.32, atol=0.01)
        np.testing.assert_allclose(1e4 * fit.strikes_ttms[0][[0, -1]], [402.8, 457.3], atol=0.05)

    elif local == Locals.SOFR_MONTE_CARLO:
        _, fit = sofr_data()
        results = [sofr_benchmark(sofr_params(), fit, idx) for idx in range(2)]
        # the module centres the strikes on its model futures rate, 134 bp above the chain forward
        np.testing.assert_allclose(1e4 * results[0]["futures_rate"], 566.59, atol=0.01)
        assert [(r["inside_FIRST"], r["inside_SECOND"]) for r in results] == [(21, 21), (15, 21)]
        np.testing.assert_allclose([1e4 * np.mean(r["up"] - r["down"]) / 2 for r in results],
                                   [0.91, 0.92], atol=0.01)


if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
