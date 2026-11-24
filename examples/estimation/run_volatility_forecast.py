"""Walk a five-business-day Parkinson-variance forecast forward against persistence."""

from enum import Enum

import numpy as np
import pandas as pd

from examples.estimation.run_ohlc import synthetic_bars
from stochvolmodels.estimation import (
    TRADING_1W,
    VolForecastConfig,
    VolatilityForecastModel,
    build_volatility_features,
    compare_volatility_forecasts,
    estimate_ohlc_var,
    fit_volatility_forecaster,
    make_forward_variance_target,
    predict_volatility_forecaster,
    walk_forward_volatility_forecast,
)


def forecasts(bars: pd.DataFrame):
    """Fit a HAR on completed labels and compare its expanding-window predictions with persistence."""
    variance = estimate_ohlc_var(bars)
    features = build_volatility_features(variance.to_frame("pk"), feature_space="variance")
    names = ("pk_current_var", "pk_mean5_var", "pk_mean21_var")
    config = VolForecastConfig(feature_names=names, forecast_space="variance")
    options = dict(
        features=features,
        per_period_variance=variance,
        horizon=TRADING_1W,
        annualization_factor=252,
        min_train_size=25,
        refit_every=5,
    )
    har = walk_forward_volatility_forecast(
        **options, model=VolatilityForecastModel.HAR, config=config
    )
    persistence = walk_forward_volatility_forecast(
        **options,
        model=VolatilityForecastModel.PERSISTENCE,
        config=VolForecastConfig(feature_names=names[:1], forecast_space="variance"),
    )
    # At the first refit, the last eligible label ends exactly on the forecast origin.
    origin = har.refit_origins[0]
    last = features.index[features.index.get_loc(origin) - TRADING_1W.periods]
    target = make_forward_variance_target(variance, TRADING_1W, annualization_factor=252)
    fit = fit_volatility_forecaster(
        features.loc[:last], target.loc[:last], VolatilityForecastModel.HAR, config
    )
    prediction = predict_volatility_forecaster(fit, features.loc[[origin]]).to_series().iloc[0]
    frame = har.to_frame()
    np.testing.assert_allclose(prediction, frame.loc[origin, "predicted_variance"])
    return frame["predicted_variance"], compare_volatility_forecasts(har, persistence)


class Locals(Enum):
    HAR_VS_PERSISTENCE = 1


def run_local(local: Locals) -> None:
    """Verify that extending the history leaves all previously issued forecasts unchanged."""
    if local != Locals.HAR_VS_PERSISTENCE:
        raise ValueError(local)
    bars = synthetic_bars()
    predicted, comparison = forecasts(bars)
    prefix, _ = forecasts(bars.iloc[:-12])
    pd.testing.assert_series_equal(predicted.loc[prefix.index], prefix)
    assert comparison.n_obs > 0 and np.isfinite(comparison.volatility_rmse_gain)
    print(
        f"{comparison.n_obs} common realized five-day targets; annual volatility RMSE gain "
        f"over persistence {comparison.volatility_rmse_gain:.3%}; "
        f"{prefix.count()} earlier HAR predictions reproduced"
    )


if __name__ == "__main__":
    run_local(local=Locals.HAR_VS_PERSISTENCE)
