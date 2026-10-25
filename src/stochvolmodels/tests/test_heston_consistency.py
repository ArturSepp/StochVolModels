"""Regression contracts between Heston wrappers, continuation states and path models."""

from __future__ import annotations

import numpy as np
import pytest

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.pricers import heston_pricer as heston
from stochvolmodels.utils.config import VariableType
from stochvolmodels.utils.funcs import set_seed


def test_heston_wrapper_preserves_requested_pricing_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public wrapper must dispatch annualized variance requests to the variance engine."""
    captured: dict[str, object] = {}

    def capture(**kwargs: object) -> list[np.ndarray]:
        captured.update(kwargs)
        return [np.array([0.01])]

    monkeypatch.setattr(heston, "heston_chain_pricer", capture)
    chain = OptionChain.slice_to_chain(
        ttm=0.5,
        forward=0.04,
        strikes=np.array([0.03]),
        optiontypes=np.array(["C"]),
    )
    heston.HestonPricer().price_chain(chain, heston.HestonParams(), variable_type=VariableType.Q_VAR)
    assert captured["variable_type"] is VariableType.Q_VAR


def test_heston_terminal_wrapper_preserves_initial_log_return(monkeypatch: pytest.MonkeyPatch) -> None:
    """The existing x0 argument supplies each path's initial log-return state."""
    captured: dict[str, object] = {}

    def capture(**kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        captured.update(kwargs)
        return kwargs["x0"], kwargs["var0"], kwargs["qvar0"]

    monkeypatch.setattr(heston, "simulate_heston_x_vol_terminal", capture)
    x, _, _ = heston.HestonPricer().simulate_terminal_values(
        heston.HestonParams(), ttm=0.5, nb_path=4, x0=0.3
    )
    np.testing.assert_array_equal(x, np.full(4, 0.3))
    np.testing.assert_array_equal(captured["x0"], np.full(4, 0.3))


@pytest.mark.parametrize("n_paths", [1, 4])
def test_heston_continuation_preserves_scalar_shaped_states(n_paths: int) -> None:
    """An initial state vector of length one broadcasts its values, including on one path."""
    kwargs = dict(
        ttm=0.02, var0=np.array([0.04]), theta=0.04, kappa=2.0, rho=-0.5,
        volvol=0.0, nb_path=n_paths, nb_steps_per_year=360,
    )
    set_seed(123)
    base = heston.simulate_heston_x_vol_terminal(
        x0=np.array([0.0]), qvar0=np.array([0.0]), **kwargs
    )
    set_seed(123)
    continued = heston.simulate_heston_x_vol_terminal(
        x0=np.array([0.3]), qvar0=np.array([0.2]), **kwargs
    )
    np.testing.assert_allclose(continued[0] - base[0], 0.3, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(continued[1], base[1])
    np.testing.assert_allclose(continued[2] - base[2], 0.2, rtol=0.0, atol=1e-15)
