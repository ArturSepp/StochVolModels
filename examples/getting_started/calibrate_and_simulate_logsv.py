"""Fit classical LogSV to the bundled Bitcoin IVs and simulate the fitted risk-neutral paths."""

import numpy as np

import stochvolmodels as svm
from stochvolmodels.models.logsv import LogSvModel


def main() -> None:
    """Calibrate once and use the fitted dynamics at the four observed expiries."""
    chain = svm.get_btc_test_chain_data()
    start = svm.LogSvParams(sigma0=1.2, theta=0.6, kappa1=2.21, kappa2=2.18,
                            beta=-0.5, volvol=1.0)
    fitted = svm.LogSVPricer().calibrate_model_params_to_chain(
        option_chain=chain, params0=start, fourier={},
        model_calibration_type=svm.LogsvModelCalibrationType.PARAMS4,
        constraints_type=svm.ConstraintsType.MMA_MARTINGALE,
    )
    paths = LogSvModel(fitted).simulate_paths(
        measure="Q_MMA", observation_times=np.r_[0.0, chain.ttms],
        spot0=1.0, n_paths=4096, steps_per_year=360, seed=17,
    )
    levels = paths.assets[:, :, 0]
    sigma = paths.states["sigma"]
    integrated_variance = paths.states["quadratic_variance"]

    assert levels.shape == sigma.shape == integrated_variance.shape == (4096, 1 + len(chain.ttms))
    assert paths.diagnostics["failed_path_count"] == 0
    np.testing.assert_allclose(sigma[:, 0], fitted.sigma0)
    np.testing.assert_array_equal(levels[:, 0], 1.0)
    assert np.all(np.diff(integrated_variance, axis=1) >= 0.0)
    print(f"Fitted sigma0={fitted.sigma0:.4f}, theta={fitted.theta:.4f}, "
          f"beta={fitted.beta:.4f}, volvol={fitted.volvol:.4f}")
    print(f"{levels.shape[0]} paths at times in years {paths.observation_times.round(4)}")
    print(f"Mean terminal normalized level={levels[:, -1].mean():.4f}, "
          f"mean terminal annualized volatility={sigma[:, -1].mean():.4f}")


if __name__ == "__main__":
    main()
