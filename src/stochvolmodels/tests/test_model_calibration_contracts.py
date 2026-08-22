from dataclasses import dataclass
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numba.typed import List

import stochvolmodels.pricers.heston_pricer as heston_module
import stochvolmodels.pricers.logsv_pricer as logsv_module
import stochvolmodels.pricers.model_pricer as model_pricer_module
from stochvolmodels import compute_bsm_vanilla_price
from stochvolmodels.data.option_chain import OptionChain, OptionSlice
from stochvolmodels.pricers.heston_pricer import HestonParams, HestonPricer
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pricer import (
    CalibrationEngine,
    ConstraintsType,
    LogsvModelCalibrationType,
    LogSVPricer,
)
from stochvolmodels.pricers.model_pricer import (
    CalibrationError,
    ModelParams,
    ModelPricer,
    validate_optimization_result,
)
from stochvolmodels.utils.config import VariableType


@dataclass
class _FlatVolParams(ModelParams):
    vol: float = 0.2


class _FlatVolPricer(ModelPricer):
    """Small deterministic pricer used to exercise the shared public interface."""

    def price_chain(self, option_chain, params, **kwargs):
        prices = List()
        for ttm, forward, discfactor, strikes, optiontypes in zip(
            option_chain.ttms,
            option_chain.forwards,
            option_chain.discfactors,
            option_chain.strikes_ttms,
            option_chain.optiontypes_ttms,
        ):
            prices.append(
                np.array(
                    [
                        compute_bsm_vanilla_price(
                            forward=forward,
                            strike=strike,
                            ttm=ttm,
                            vol=params.vol,
                            optiontype=optiontype,
                            discfactor=discfactor,
                        )
                        for strike, optiontype in zip(strikes, optiontypes)
                    ]
                )
            )
        return prices

    def model_mc_price_chain(self, option_chain, params, **kwargs):
        prices = self.price_chain(option_chain, params)
        errors = List(np.full_like(price, 1.0e-4) for price in prices)
        return prices, errors

    def simulate_terminal_values(self, params, **kwargs):
        return np.array([-0.2, -0.1, 0.0, 0.1, 0.2, np.nan, np.inf, -np.inf])


class _PriceOnlyPricer(_FlatVolPricer):
    model_mc_price_chain = ModelPricer.model_mc_price_chain
    simulate_terminal_values = ModelPricer.simulate_terminal_values


def _calibration_chain() -> OptionChain:
    return OptionChain.get_uniform_chain(
        ttms=np.array([0.25]),
        ids=np.array(["3m"]),
        forwards=np.array([1.0]),
        strikes=np.array([0.9, 1.0, 1.1]),
        flat_vol=0.2,
    )


def _constraint_items(constraints):
    if constraints is None:
        return ()
    if isinstance(constraints, dict):
        return (constraints,)
    return constraints


def test_model_pricer_common_price_and_mc_interfaces() -> None:
    chain = _calibration_chain()
    params = _FlatVolParams()
    pricer = _FlatVolPricer()

    prices, ivols = pricer.compute_chain_prices_with_vols(chain, params)
    model_ivols = pricer.compute_model_ivols_for_chain(chain, params)
    mc_result = pricer.compute_mc_chain_implied_vols(
        chain,
        params,
        variable_type=VariableType.LOG_RETURN,
        nb_path=100,
    )

    np.testing.assert_allclose(ivols[0], params.vol, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(model_ivols[0], ivols[0], rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(mc_result[0][0], prices[0], rtol=0.0, atol=0.0)
    assert np.all(mc_result[1][0] > mc_result[0][0])
    assert np.all(mc_result[2][0] < mc_result[0][0])
    assert np.all(mc_result[4][0] > mc_result[3][0])
    assert np.all(mc_result[5][0] < mc_result[3][0])
    np.testing.assert_allclose(mc_result[6][0], 1.0e-4, rtol=0.0, atol=0.0)


def test_model_pricer_mc_pdf_filters_invalid_paths_and_normalizes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    grid = np.linspace(-0.5, 0.5, 51)
    density = _FlatVolPricer().get_log_return_mc_pdf(
        ttm=0.25,
        params=_FlatVolParams(),
        x_grid=grid,
        nb_path=8,
    )

    assert np.all(np.isfinite(density))
    assert np.all(density >= 0.0)
    np.testing.assert_allclose(np.sum(density), 1.0, rtol=0.0, atol=1.0e-14)
    output = capsys.readouterr().out
    assert "num -inf = 1" in output
    assert "num +inf = 1" in output
    assert "num nans = 1" in output


def test_model_pricer_default_unsupported_interfaces_raise() -> None:
    pricer = _PriceOnlyPricer()
    params = _FlatVolParams()
    chain = _calibration_chain()

    with pytest.raises(NotImplementedError):
        pricer.model_mc_price_chain(chain, params)
    with pytest.raises(NotImplementedError):
        pricer.calibrate_model_params_to_chain(chain)
    with pytest.raises(NotImplementedError):
        pricer.simulate_vol_paths(params)
    with pytest.raises(NotImplementedError):
        pricer.simulate_terminal_values(params)
    with pytest.raises(NotImplementedError):
        pricer.compute_logreturn_pdf(params)


def test_model_pricer_plotting_interfaces_delegate_complete_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _calibration_chain()
    params = _FlatVolParams()
    pricer = _FlatVolPricer()
    captured = {"model": [], "fit": []}
    monkeypatch.setattr(
        model_pricer_module.plot,
        "model_vols_ts",
        lambda **kwargs: captured["model"].append(kwargs),
    )
    monkeypatch.setattr(
        model_pricer_module.plot,
        "vol_slice_fit",
        lambda **kwargs: captured["fit"].append(kwargs),
    )

    figure = pricer.plot_model_ivols(
        chain,
        params,
        is_log_strike_xaxis=True,
        headers=["Synthetic"],
    )
    option_slice = OptionSlice(
        ttm=chain.ttms[0],
        forward=chain.forwards[0],
        strikes=np.asarray(chain.strikes_ttms[0]),
        optiontypes=np.asarray(chain.optiontypes_ttms[0]),
        id="3m",
        discfactor=chain.discfactors[0],
    )
    slices_figure = pricer.plot_model_slices_in_params(
        option_slice,
        {"flat": params},
        is_log_strike_xaxis=True,
    )
    fit_figure = pricer.plot_model_ivols_vs_bid_ask(
        chain,
        params,
        is_log_strike_xaxis=True,
        headers=["Synthetic"],
    )
    mc_figure = pricer.plot_model_ivols_vs_mc(chain, params)
    comparison_figure = pricer.plot_comp_mma_inverse_options_with_mc(
        chain,
        params,
        is_plot_vols=False,
    )

    assert all(
        item is not None
        for item in (figure, slices_figure, fit_figure, mc_figure, comparison_figure)
    )
    assert len(captured["model"]) == 2
    assert len(captured["fit"]) == 3
    assert captured["model"][0]["xlabel"] == "log-strike"
    assert captured["fit"][0]["strike_name"] == "log-strike"
    assert captured["fit"][-1]["ylabel"] == "Model prices"
    plt.close("all")


def test_heston_calibration_builds_weighted_objective_and_feller_constraint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    chain = _calibration_chain()
    params0 = HestonParams(v0=0.04, theta=0.05, kappa=2.0, rho=-0.5, volvol=0.4)
    captured = {}
    pricer = HestonPricer()
    monkeypatch.setattr(
        pricer,
        "compute_model_ivols_for_chain",
        lambda **kwargs: [chain.get_mid_vols()[0] + 0.01],
    )

    def fake_minimize(fun, x0, args, method, constraints, bounds, options):
        captured["objective"] = fun(np.asarray(x0), args)
        captured["constraint_values"] = [
            item["fun"](np.asarray(x0)) for item in _constraint_items(constraints)
        ]
        captured["method"] = method
        captured["bounds"] = bounds
        captured["options"] = options
        return SimpleNamespace(x=np.asarray(x0), success=True, message="converged")

    monkeypatch.setattr(heston_module, "minimize", fake_minimize)
    fitted = pricer.calibrate_model_params_to_chain(
        chain, params0=params0, is_vega_weighted=False
    )

    np.testing.assert_allclose(captured["objective"], 3.0e-4, atol=1.0e-15)
    assert captured["constraint_values"] == pytest.approx([0.04])
    assert captured["method"] == "SLSQP"
    assert len(captured["bounds"]) == 5
    assert captured["options"]["ftol"] == 1.0e-8
    assert fitted == params0
    assert capsys.readouterr().out == ""


def test_logsv_calibration_builds_objective_and_requested_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _calibration_chain()
    params0 = LogSvParams(
        sigma0=0.2,
        theta=0.22,
        kappa1=3.0,
        kappa2=12.0,
        beta=-0.3,
        volvol=0.4,
    )
    captured = {}
    pricer = LogSVPricer()
    monkeypatch.setattr(
        pricer,
        "compute_model_ivols_for_chain",
        lambda **kwargs: [chain.get_mid_vols()[0] + 0.01],
    )

    def fake_minimize(fun, x0, args, method, constraints, bounds, options):
        captured["objective"] = fun(np.asarray(x0), args)
        captured["constraint_values"] = [
            item["fun"](np.asarray(x0)) for item in _constraint_items(constraints)
        ]
        captured["method"] = method
        captured["bounds"] = bounds
        return SimpleNamespace(x=np.asarray(x0), success=True, message="converged")

    monkeypatch.setattr(logsv_module, "minimize", fake_minimize)
    fitted = pricer.calibrate_model_params_to_chain(
        chain,
        params0=params0,
        is_vega_weighted=False,
        model_calibration_type=LogsvModelCalibrationType.PARAMS4,
        constraints_type=ConstraintsType.INVERSE_MARTINGALE_MOMENT4,
        calibration_engine=CalibrationEngine.ANALYTIC,
    )

    np.testing.assert_allclose(captured["objective"], 3.0e-4, atol=1.0e-15)
    assert captured["constraint_values"] == pytest.approx([12.6, 5.265])
    assert captured["method"] == "SLSQP"
    assert len(captured["bounds"]) == 4
    assert fitted == params0


def test_logsv_parameter_codec_preserves_all_supported_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params0 = LogSvParams(
        sigma0=0.2,
        theta=0.25,
        kappa1=3.0,
        kappa2=4.0,
        beta=-0.3,
        volvol=0.5,
        H=0.45,
        nodes=np.array([0.1, 0.2]),
        weights=np.array([0.6, 0.4]),
    )
    params_min = LogSvParams(
        sigma0=0.1,
        theta=0.1,
        kappa1=0.25,
        kappa2=0.25,
        beta=-3.0,
        volvol=0.2,
    )
    params_max = LogSvParams(
        sigma0=1.5,
        theta=1.5,
        kappa1=10.0,
        kappa2=10.0,
        beta=3.0,
        volvol=3.0,
    )

    codec4 = logsv_module._LogSvParameterCodec(
        params0,
        params_min,
        params_max,
        LogsvModelCalibrationType.PARAMS4,
        None,
    )
    p4, bounds4 = codec4.initial_and_bounds()
    parsed4 = codec4.parse(p4)
    np.testing.assert_allclose(p4, np.array([0.2, 0.25, -0.3, 0.5]))
    assert len(bounds4) == 4
    assert parsed4.kappa1 == params0.kappa1
    assert parsed4.kappa2 == params0.kappa2
    assert parsed4.nodes is params0.nodes
    assert parsed4.weights is params0.weights

    codec5 = logsv_module._LogSvParameterCodec(
        params0,
        params_min,
        params_max,
        LogsvModelCalibrationType.PARAMS5,
        None,
    )
    p5, bounds5 = codec5.initial_and_bounds()
    parsed5 = codec5.parse(p5)
    np.testing.assert_allclose(p5, np.array([0.2, 0.25, 3.0, -0.3, 0.5]))
    assert len(bounds5) == 5
    assert parsed5.kappa2 == pytest.approx(parsed5.kappa1 / parsed5.theta)

    monkeypatch.setattr(
        logsv_module,
        "fit_model_vol_backbone_to_varswaps",
        lambda **kwargs: "backbone",
    )
    codec_varswap = logsv_module._LogSvParameterCodec(
        params0,
        params_min,
        params_max,
        LogsvModelCalibrationType.PARAMS_WITH_VARSWAP_FIT,
        np.array([0.04]),
    )
    p_varswap, bounds_varswap = codec_varswap.initial_and_bounds()
    parsed_varswap = codec_varswap.parse(p_varswap)
    np.testing.assert_allclose(p_varswap, np.array([-0.3, 0.5]))
    assert len(bounds_varswap) == 2
    assert parsed_varswap.vol_backbone == "backbone"


def test_logsv_calibration_components_preserve_weights_and_constraints() -> None:
    chain = _calibration_chain()
    market_vols = np.concatenate(chain.get_mid_vols())
    unweighted = logsv_module._compute_logsv_calibration_weights(
        chain,
        market_vols,
        is_vega_weighted=False,
        is_unit_ttm_vega=False,
    )
    weighted = logsv_module._compute_logsv_calibration_weights(
        chain,
        market_vols,
        is_vega_weighted=True,
        is_unit_ttm_vega=False,
    )
    np.testing.assert_array_equal(unweighted, np.ones_like(market_vols))
    assert np.all(weighted > 0.0)
    np.testing.assert_allclose(np.sum(weighted), 1.0, atol=1.0e-15)

    params0 = LogSvParams(theta=0.22, kappa1=3.0, kappa2=12.0, beta=-0.3, volvol=0.4)
    codec = logsv_module._LogSvParameterCodec(
        params0,
        LogSvParams(),
        LogSvParams(sigma0=2.0, theta=2.0, kappa1=20.0, kappa2=20.0, beta=3.0, volvol=3.0),
        LogsvModelCalibrationType.PARAMS4,
        None,
    )
    pars, _ = codec.initial_and_bounds()

    assert logsv_module._build_logsv_constraints(
        codec, ConstraintsType.UNCONSTRAINT
    ) is None
    mma = _constraint_items(
        logsv_module._build_logsv_constraints(codec, ConstraintsType.MMA_MARTINGALE)
    )
    inverse_moment = _constraint_items(
        logsv_module._build_logsv_constraints(
            codec, ConstraintsType.INVERSE_MARTINGALE_MOMENT4
        )
    )
    assert [item["fun"](pars) for item in mma] == pytest.approx([12.3])
    assert [item["fun"](pars) for item in inverse_moment] == pytest.approx(
        [12.6, 5.265]
    )

    inverse = _constraint_items(
        logsv_module._build_logsv_constraints(codec, ConstraintsType.INVERSE_MARTINGALE)
    )
    mma_moment = _constraint_items(
        logsv_module._build_logsv_constraints(
            codec, ConstraintsType.MMA_MARTINGALE_MOMENT4
        )
    )
    assert [item["fun"](pars) for item in inverse] == pytest.approx([12.6])
    assert [item["fun"](pars) for item in mma_moment] == pytest.approx([12.3, 5.265])
    with pytest.raises(NotImplementedError):
        logsv_module._build_logsv_constraints(codec, object())


@pytest.mark.parametrize("engine", [CalibrationEngine.MC, CalibrationEngine.ROUGH_MC])
def test_logsv_calibration_objective_supports_simulation_engines(
    monkeypatch: pytest.MonkeyPatch,
    engine: CalibrationEngine,
) -> None:
    chain = _calibration_chain()
    params0 = LogSvParams(
        sigma0=0.2,
        theta=0.22,
        kappa1=3.0,
        kappa2=12.0,
        beta=-0.3,
        volvol=0.4,
    )
    codec = logsv_module._LogSvParameterCodec(
        params0,
        LogSvParams(),
        LogSvParams(
            sigma0=2.0,
            theta=2.0,
            kappa1=20.0,
            kappa2=20.0,
            beta=3.0,
            volvol=3.0,
        ),
        LogsvModelCalibrationType.PARAMS4,
        None,
    )
    pars, _ = codec.initial_and_bounds()
    exact_prices = _FlatVolPricer().price_chain(chain, _FlatVolParams())
    if engine == CalibrationEngine.MC:
        monkeypatch.setattr(
            logsv_module,
            "logsv_mc_chain_pricer_fixed_randoms",
            lambda **kwargs: (exact_prices, None),
        )
        random_inputs = ((), (), ())
    else:
        monkeypatch.setattr(
            logsv_module,
            "rough_logsv_mc_chain_pricer_fixed_randoms",
            lambda **kwargs: (exact_prices, None),
        )
        random_inputs = (np.empty((0, 0)), np.empty((0, 0)), ())

    objective = logsv_module._LogSvCalibrationObjective(
        pricer=LogSVPricer(),
        option_chain=chain,
        codec=codec,
        market_vols=np.concatenate(chain.get_mid_vols()),
        weights=np.ones(3),
        calibration_engine=engine,
        vol_scaler=0.1,
        random_inputs=random_inputs,
    )

    np.testing.assert_allclose(objective(pars, None), 0.0, rtol=0.0, atol=1.0e-18)


def test_logsv_calibration_codec_and_objective_reject_unknown_modes() -> None:
    chain = _calibration_chain()
    params0 = LogSvParams()
    codec = logsv_module._LogSvParameterCodec(
        params0,
        LogSvParams(),
        LogSvParams(
            sigma0=2.0,
            theta=2.0,
            kappa1=20.0,
            kappa2=20.0,
            beta=3.0,
            volvol=3.0,
        ),
        object(),
        None,
    )
    with pytest.raises(NotImplementedError):
        codec.parse(np.ones(2))
    with pytest.raises(NotImplementedError):
        codec.initial_and_bounds()

    valid_codec = logsv_module._LogSvParameterCodec(
        params0,
        LogSvParams(),
        LogSvParams(
            sigma0=2.0,
            theta=2.0,
            kappa1=20.0,
            kappa2=20.0,
            beta=3.0,
            volvol=3.0,
        ),
        LogsvModelCalibrationType.PARAMS4,
        None,
    )
    pars, _ = valid_codec.initial_and_bounds()
    objective = logsv_module._LogSvCalibrationObjective(
        pricer=LogSVPricer(),
        option_chain=chain,
        codec=valid_codec,
        market_vols=np.concatenate(chain.get_mid_vols()),
        weights=np.ones(3),
        calibration_engine=object(),
        vol_scaler=0.1,
        random_inputs=None,
    )
    with pytest.raises(NotImplementedError):
        objective(pars, None)


def test_heston_calibration_rejects_failed_optimizer_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _calibration_chain()
    params0 = HestonParams()

    def failed_minimize(fun, x0, **kwargs):
        return SimpleNamespace(x=np.asarray(x0), success=False, message="iteration limit")

    monkeypatch.setattr(heston_module, "minimize", failed_minimize)
    with pytest.raises(RuntimeError, match="iteration limit"):
        HestonPricer().calibrate_model_params_to_chain(chain, params0=params0)


def test_logsv_calibration_rejects_failed_optimizer_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _calibration_chain()
    params0 = LogSvParams()

    def failed_minimize(fun, x0, **kwargs):
        return SimpleNamespace(x=np.asarray(x0), success=False, message="iteration limit")

    monkeypatch.setattr(logsv_module, "minimize", failed_minimize)
    with pytest.raises(RuntimeError, match="iteration limit"):
        LogSVPricer().calibrate_model_params_to_chain(chain, params0=params0)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.array([np.nan]), "non-finite"),
        (np.array([-0.1]), "below bounds"),
        (np.array([1.1]), "above bounds"),
    ],
)
def test_optimizer_result_validation_rejects_invalid_success_vectors(
    values: np.ndarray, message: str
) -> None:
    result = SimpleNamespace(x=values, success=True, message="reported success")

    with pytest.raises(CalibrationError, match=message):
        validate_optimization_result(result, bounds=((0.0, 1.0),))
