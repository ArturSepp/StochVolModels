"""Numerical and payload contracts for the terminal TGARCH path model."""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import stochvolmodels
import stochvolmodels.models as model_capabilities
from stochvolmodels.models import PathModel
from stochvolmodels.models import (
    TerminalDistributionModel,
    TerminalSmileModel,
    TransformModel,
)
from stochvolmodels.models.tgarch import (
    TgarchMeasure,
    TgarchModel,
    TgarchParams,
    derive_tgarch_limit_params,
)


REFERENCE_SOURCE_SHA256 = (
    "4a07de7c20591276a9f241b1c18b96e2f2c3fd1b953e29375ab5096b3fb35f38"
)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_TERMINALS = {
    ("crypto", TgarchMeasure.P): {
        "spot": [
            0.8602685056600672,
            0.8638368188335316,
            0.9471816269440217,
            0.8530545753920327,
            1.0365692702077187,
            1.075263429328865,
            0.9639498770029068,
            1.115874381682727,
        ],
        "log_spot": [
            -0.1505107225800212,
            -0.1463713951065363,
            -0.05426441227023124,
            -0.15893175300719192,
            0.03591648154119099,
            0.07256568209812608,
            -0.03671598053363115,
            0.10963829640743573,
        ],
        "sigma": [
            0.5182583435715801,
            0.5395749801274664,
            0.6507130760607134,
            0.44654185118227663,
            0.6541550576094471,
            0.8527595600461183,
            0.6715673872914589,
            0.5765180470526559,
        ],
        "log_weights": [
            -0.04262308533848233,
            -0.046517056783785045,
            -0.027566564516974368,
            0.11932231387408981,
            -0.16245111496375653,
            -0.17736181249716082,
            -0.01977010397910056,
            0.010043357767623287,
        ],
        "ess": 7.937715366423826,
        "ess_fraction": 0.9922144208029783,
    },
    ("crypto", TgarchMeasure.Q_EXACT): {
        "spot": [
            0.8358021720572095,
            0.8519536840247589,
            0.9299124760945747,
            0.8429078612953754,
            1.00436678994413,
            1.0312882383378925,
            0.9363752280847721,
            1.0819726269064622,
        ],
        "log_spot": [
            -0.17936333019542516,
            -0.16022311510374748,
            -0.07266480898988556,
            -0.17089762553580523,
            0.004357283182893579,
            0.0308087375797029,
            -0.06573899813139689,
            0.07878588149656882,
        ],
        "sigma": [
            0.4927230454994407,
            0.5166365397032728,
            0.6007507956754092,
            0.42001657964689754,
            0.585628683627495,
            0.7775680904171415,
            0.6274128657745263,
            0.5373764645155664,
        ],
    },
    ("crypto", TgarchMeasure.Q_LIMIT): {
        "spot": [
            0.8336258363498804,
            0.8493744212406097,
            0.9229376358976386,
            0.8383968684440025,
            0.9921549970695158,
            1.0246013233458469,
            0.9296000316485745,
            1.0786114003344536,
        ],
        "log_spot": [
            -0.1819706147614576,
            -0.16325517547138674,
            -0.08019361350347337,
            -0.1762637005797748,
            -0.007875936856679651,
            0.024303584095278205,
            -0.07300085883912238,
            0.07567447343165692,
        ],
        "sigma": [
            0.46824131348302167,
            0.49109111234152947,
            0.591259227085676,
            0.41035750990538605,
            0.5838058506299707,
            0.7759262127723531,
            0.6067197949695515,
            0.5256513864257265,
        ],
    },
    ("equity", TgarchMeasure.P): {
        "spot": [
            0.9825253762635643,
            0.9761127327016111,
            1.0141682009541342,
            0.9667563748647888,
            1.0637000263638765,
            1.0586798012436665,
            1.0221959793039181,
            1.0551053073824515,
        ],
        "log_spot": [
            -0.017629107314183715,
            -0.02417719442198783,
            0.014068770065641212,
            -0.03380875439721766,
            0.061753421060549485,
            0.05702266135612664,
            0.021953233969153017,
            0.053640579370726735,
        ],
        "sigma": [
            0.22380581905498861,
            0.2843414906881642,
            0.22073978143526998,
            0.20807051449351657,
            0.1759923051266912,
            0.17550591212883956,
            0.21275095951647324,
            0.15843552737384953,
        ],
        "log_weights": [
            -0.04454818789063366,
            -0.04351394613522393,
            -0.021279199854717528,
            0.040990109394855576,
            -0.0768392037539565,
            -0.07513337583812273,
            -0.03070388171548223,
            -0.012652196185570817,
        ],
        "ess": 7.9897826540643475,
        "ess_fraction": 0.9987228317580434,
    },
    ("equity", TgarchMeasure.Q_EXACT): {
        "spot": [
            0.9759812146114837,
            0.9722439977140559,
            1.009510123668692,
            0.9633595083801669,
            1.0585416651738524,
            1.0530903106963325,
            1.0175673303689623,
            1.0508285949555236,
        ],
        "log_spot": [
            -0.024311940078576226,
            -0.028148479568776728,
            0.009465187119373535,
            -0.037328615584009794,
            0.05689217328749401,
            0.051728994617934336,
            0.017414808498107982,
            0.04957899101659052,
        ],
        "sigma": [
            0.22456142878578783,
            0.28527755751966594,
            0.2198286493293046,
            0.20808529514827157,
            0.17498513695559775,
            0.17495838956048707,
            0.21266394151682025,
            0.1584440735065741,
        ],
    },
    ("equity", TgarchMeasure.Q_LIMIT): {
        "spot": [
            0.9760030824473953,
            0.9718442347654737,
            1.0095219254214949,
            0.9626414682548575,
            1.0589561497243387,
            1.053374321639686,
            1.0175040104006166,
            1.0512431513969238,
        ],
        "log_spot": [
            -0.024289534328749755,
            -0.028559739662364976,
            0.009476877625037666,
            -0.038074243615653675,
            0.05728365851393809,
            0.05199865112269081,
            0.017352579752512906,
            0.0499734175603028,
        ],
        "sigma": [
            0.22378280505775017,
            0.28431798164790695,
            0.22072018548498093,
            0.20805388246346718,
            0.17597738781173133,
            0.17549124191972262,
            0.21273347267104423,
            0.15842359006464002,
        ],
    },
}


def _study_params(name: str) -> TgarchParams:
    """Return one of the two parameter sets used by the companion study."""
    common = {
        "r": 0.0,
        "gamma0": 0.0,
        "gamma1": 0.5,
        "eta0": 0.0,
        "eta1": -0.38,
        "spot0": 1.0,
    }
    if name == "crypto":
        return TgarchParams(
            theta=0.6,
            kappa1=3.0,
            kappa2=3.0,
            beta=1.0,
            eps=1.5,
            sigma0=0.6,
            **common,
        )
    if name == "equity":
        return TgarchParams(
            theta=0.2,
            kappa1=4.0,
            kappa2=4.0,
            beta=-1.0,
            eps=1.0,
            sigma0=0.2,
            **common,
        )
    raise ValueError(f"unknown parameter set: {name}")


def _simulate(
    name: str,
    measure: TgarchMeasure,
    *,
    n_paths: int = 8,
    seed: int = 20260825,
    chunk_steps: int = 8,
    track_log_weights: bool | None = None,
):
    """Run the bounded terminal adapter with the study grid."""
    if track_log_weights is None:
        track_log_weights = measure is TgarchMeasure.P
    return TgarchModel(_study_params(name)).simulate_paths(
        measure=measure,
        maturity=0.25,
        max_dt=1.0 / 52.0,
        n_paths=n_paths,
        seed=seed,
        chunk_steps=chunk_steps,
        track_log_weights=track_log_weights,
    )


def _paired_standard_error(values: np.ndarray) -> float:
    """Return the standard error respecting first-half antithetic pairing."""
    pair_count = values.size // 2
    pair_means = 0.5 * (values[:pair_count] + values[pair_count:])
    return float(np.std(pair_means, ddof=1) / math.sqrt(pair_count))


def _stable_raw_weights(log_weights: np.ndarray) -> np.ndarray:
    """Exponentiate raw ratios using the companion study's shifted calculation."""
    maximum = float(np.max(log_weights))
    scaled = np.exp(log_weights - maximum)
    return math.exp(maximum) * scaled


def _controlled_call_pair_values(
    terminal_spot: np.ndarray,
    *,
    strike: float,
    discount: float,
    expected_discounted_spot: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Reproduce E2's separately fitted discounted-terminal-spot control variate."""
    half = terminal_spot.size // 2
    left = terminal_spot[:half]
    right = terminal_spot[half:]
    payoff_left = discount * np.maximum(left - strike, 0.0)
    payoff_right = discount * np.maximum(right - strike, 0.0)
    spot_left = discount * left
    spot_right = discount * right
    if weights is not None:
        weight_left = weights[:half]
        weight_right = weights[half:]
        payoff_left = weight_left * payoff_left
        payoff_right = weight_right * payoff_right
        spot_left = weight_left * spot_left
        spot_right = weight_right * spot_right
    pair_payoff = 0.5 * (payoff_left + payoff_right)
    pair_spot = 0.5 * (spot_left + spot_right)
    spot_variance = float(np.var(pair_spot, ddof=1))
    coefficient = (
        0.0
        if spot_variance <= 1.0e-30
        else float(np.cov(pair_payoff, pair_spot, ddof=1)[0, 1] / spot_variance)
    )
    return pair_payoff - coefficient * (pair_spot - expected_discounted_spot)


@pytest.mark.parametrize("parameter_name", ["crypto", "equity"])
@pytest.mark.parametrize("measure", list(TgarchMeasure))
@pytest.mark.parametrize("chunk_steps", [8, 3])
def test_terminal_paths_match_pre_lift_legacy_capture(
    parameter_name: str,
    measure: TgarchMeasure,
    chunk_steps: int,
) -> None:
    """Both chunk contracts reproduce raw arrays captured from the authoritative simulator."""
    paths = _simulate(parameter_name, measure, chunk_steps=chunk_steps)
    expected = REFERENCE_TERMINALS[(parameter_name, measure)]

    np.testing.assert_allclose(paths.assets[:, -1, 0], expected["spot"], rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(
        paths.states["log_spot"][:, -1], expected["log_spot"], rtol=2e-15, atol=0.0
    )
    np.testing.assert_allclose(
        paths.states["sigma"][:, -1], expected["sigma"], rtol=2e-15, atol=0.0
    )
    assert paths.diagnostics["floor_hits"] == 0
    assert paths.diagnostics["spot_overflow_count"] == 0
    assert paths.provenance["chunk_steps"] == chunk_steps
    assert paths.provenance["legacy_source_sha256"] == REFERENCE_SOURCE_SHA256

    if measure is TgarchMeasure.P:
        np.testing.assert_allclose(
            paths.log_likelihood_ratios,
            expected["log_weights"],
            rtol=2e-15,
            atol=0.0,
        )
        assert paths.diagnostics["effective_sample_size"] == pytest.approx(
            expected["ess"], rel=2e-15
        )
        assert paths.diagnostics["ess_fraction"] == pytest.approx(
            expected["ess_fraction"], rel=2e-15
        )
    else:
        assert paths.log_likelihood_ratios is None
        assert paths.diagnostics["effective_sample_size"] is None
        assert paths.diagnostics["ess_fraction"] is None


def test_odd_path_capture_preserves_singleton_last_random_ordering() -> None:
    """An unpaired seventh path uses the fourth independent draw and remains last."""
    paths = _simulate("crypto", TgarchMeasure.P, n_paths=7)

    np.testing.assert_allclose(
        paths.assets[:, -1, 0],
        [
            0.8602685056600672,
            0.8638368188335316,
            0.9471816269440217,
            1.0365692702077187,
            1.075263429328865,
            0.9639498770029068,
            0.8530545753920327,
        ],
        rtol=2e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        paths.states["log_spot"][:, -1],
        [
            -0.1505107225800212,
            -0.1463713951065363,
            -0.05426441227023124,
            0.03591648154119099,
            0.07256568209812608,
            -0.03671598053363115,
            -0.15893175300719192,
        ],
        rtol=2e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        paths.states["sigma"][:, -1],
        [
            0.5182583435715801,
            0.5395749801274664,
            0.6507130760607134,
            0.6541550576094471,
            0.8527595600461183,
            0.6715673872914589,
            0.44654185118227663,
        ],
        rtol=2e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        paths.log_likelihood_ratios,
        [
            -0.04262308533848233,
            -0.04651705678378504,
            -0.02756656451697437,
            -0.16245111496375653,
            -0.17736181249716082,
            -0.01977010397910056,
            0.11932231387408981,
        ],
        rtol=2e-15,
        atol=0.0,
    )


def test_floor_capture_preserves_post_update_floor_and_hit_count() -> None:
    """A one-step legacy fixture applies the floor after returns and counts both hits."""
    params = TgarchParams(
        theta=0.2,
        kappa1=1.0,
        kappa2=0.0,
        beta=0.0,
        eps=5.0,
        sigma0=0.05,
    )
    paths = TgarchModel(params).simulate_paths(
        measure=TgarchMeasure.P,
        maturity=0.25,
        max_dt=0.25,
        n_paths=4,
        seed=1,
    )

    np.testing.assert_allclose(
        paths.assets[:, -1, 0],
        [1.0083618715739915, 1.0204339250905095, 0.9910878459851447, 0.9793629657923952],
        rtol=2e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        paths.states["log_spot"][:, -1],
        [
            0.008327104801619651,
            0.02022795358752896,
            -0.008952104801619652,
            -0.02085295358752896,
        ],
        rtol=2e-15,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        paths.states["sigma"][:, -1],
        [1.0e-6, 0.09242144537075174, 1.0e-6, 0.09242144537075174],
    )
    assert paths.diagnostics["sigma_floor"] == 1.0e-6
    assert paths.diagnostics["floor_hits"] == 2


def test_terminal_model_returns_two_observation_capability_payload() -> None:
    """The first lift labels state, measures, RNG ordering and realized grid explicitly."""
    model = TgarchModel(_study_params("crypto"))
    assert isinstance(model, PathModel)

    paths = _simulate("crypto", TgarchMeasure.P)

    np.testing.assert_array_equal(paths.observation_times, [0.0, 0.25])
    assert paths.assets.shape == (8, 2, 1)
    assert paths.asset_ids == ("spot",)
    np.testing.assert_array_equal(paths.assets[:, 0, 0], 1.0)
    np.testing.assert_array_equal(paths.states["log_spot"][:, 0], 0.0)
    np.testing.assert_array_equal(paths.states["sigma"][:, 0], 0.6)
    assert paths.state_units == {"log_spot": "log price", "sigma": "annualized volatility"}
    assert paths.sampling_measure == "P"
    assert paths.target_measure == "Q_EXACT"
    assert paths.numeraire == "money_market_account"
    assert paths.scheme == "tgarch_terminal_recursion"
    assert paths.provenance == {
        "requested_max_dt": 1.0 / 52.0,
        "realized_dt": 1.0 / 52.0,
        "n_steps": 13,
        "seed": 20260825,
        "generator": "numpy.random.Generator",
        "bit_generator": "PCG64",
        "chunk_steps": 8,
        "antithetic_layout": "first_half_draws_then_negatives_singleton_last",
        "source_tag": "tgarch-study-round2-v1",
        "source_commit": "add76d1909b4223d005e31fcf377845501021362",
        "legacy_source_sha256": REFERENCE_SOURCE_SHA256,
        "numpy_version": np.__version__,
    }
    assert paths.diagnostics["weight_convention"] == "raw_log_dQ_EXACT_dP"
    assert paths.diagnostics["low_ess"] is False


def test_tgarch_submodule_fresh_import_stays_outside_heavy_pricing_layers() -> None:
    """The bounded model import loads NumPy and path data, not pricers or optional analytics."""
    source_root = str(PACKAGE_ROOT.parent)
    code = f"""
import sys
sys.path.insert(0, {source_root!r})
import stochvolmodels.models.tgarch
forbidden = ("numba", "pandas", "qis", "scipy", "vanilla_option_pricers")
loaded = sorted(
    root
    for root in forbidden
    if any(name == root or name.startswith(root + ".") for name in sys.modules)
)
assert not loaded, loaded
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_tgarch_types_remain_submodule_only() -> None:
    """The bounded lift does not widen either provisional capabilities or the root API."""
    assert "TgarchModel" not in stochvolmodels.__all__
    assert not hasattr(stochvolmodels, "TgarchModel")
    assert not hasattr(model_capabilities, "TgarchModel")
    model = TgarchModel(_study_params("equity"))
    assert isinstance(model, PathModel)
    assert not isinstance(model, TransformModel)
    assert not isinstance(model, TerminalDistributionModel)
    assert not isinstance(model, TerminalSmileModel)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"theta": 0.0}, "theta"),
        ({"kappa1": np.inf}, "kappa1"),
        ({"kappa2": -1.0}, "kappa2"),
        ({"beta": np.nan}, "beta"),
        ({"eps": True}, "eps"),
        ({"sigma0": 0.0}, "sigma0"),
        ({"spot0": -1.0}, "spot0"),
    ],
)
def test_tgarch_parameter_validation(changes: dict[str, object], message: str) -> None:
    """Representative bool, non-finite and non-positive parameters fail at construction."""
    with pytest.raises(ValueError, match=message):
        replace(_study_params("equity"), **changes)


def test_model_measure_and_limit_payload_validation() -> None:
    """Model ownership, measure labels and limit payloads reject invented inputs."""
    with pytest.raises(ValueError, match="TgarchParams"):
        TgarchModel(object())  # type: ignore[arg-type]

    request = {
        "maturity": 0.25,
        "max_dt": 1.0 / 52.0,
        "n_paths": 8,
        "seed": 20260825,
    }
    model = TgarchModel(_study_params("equity"))
    with pytest.raises(ValueError, match="measure must be one of"):
        model.simulate_paths(measure="Q", **request)
    with pytest.raises(ValueError, match="TgarchLimitParams"):
        model.simulate_paths(
            measure=TgarchMeasure.Q_LIMIT,
            limit_params=object(),  # type: ignore[arg-type]
            **request,
        )

    invalid_limit_map = replace(
        _study_params("equity"),
        beta=-1.0,
        gamma1=10.0,
        eta1=0.0,
    )
    with pytest.raises(ValueError, match="derived kappa2_hat is negative"):
        derive_tgarch_limit_params(invalid_limit_map)


def test_unweighted_p_paths_keep_the_physical_target_and_no_ratios() -> None:
    """Physical samples are not relabelled exact Q unless raw ratios are requested."""
    paths = _simulate("equity", TgarchMeasure.P, track_log_weights=False)

    assert paths.sampling_measure == "P"
    assert paths.target_measure == "P"
    assert paths.log_likelihood_ratios is None
    assert paths.diagnostics["weight_convention"] is None


@pytest.mark.parametrize("measure", [TgarchMeasure.Q_EXACT, TgarchMeasure.Q_LIMIT])
def test_log_weights_are_rejected_outside_physical_sampling(measure: TgarchMeasure) -> None:
    """The sole lifted density ratio is raw exact-Q over P on P-sampled paths."""
    with pytest.raises(ValueError, match="only be accumulated on P"):
        _simulate("equity", measure, track_log_weights=True)


def test_zero_kernel_measures_have_identical_terminal_laws() -> None:
    """P, exact Q and limit Q agree pathwise when both pricing-kernel loadings vanish."""
    params = replace(
        _study_params("crypto"),
        gamma0=0.0,
        gamma1=0.0,
        eta0=0.0,
        eta1=0.0,
    )
    model = TgarchModel(params)
    kwargs = {
        "maturity": 0.25,
        "max_dt": 1.0 / 52.0,
        "n_paths": 128,
        "seed": 20260826,
        "chunk_steps": 3,
    }
    paths = {measure: model.simulate_paths(measure=measure, **kwargs) for measure in TgarchMeasure}
    reference = paths[TgarchMeasure.P]

    for measure in (TgarchMeasure.Q_EXACT, TgarchMeasure.Q_LIMIT):
        np.testing.assert_allclose(paths[measure].assets, reference.assets, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(
            paths[measure].states["log_spot"],
            reference.states["log_spot"],
            rtol=2e-11,
            atol=2e-11,
        )
        np.testing.assert_allclose(
            paths[measure].states["sigma"],
            reference.states["sigma"],
            rtol=2e-11,
            atol=2e-11,
        )
        assert paths[measure].diagnostics["floor_hits"] == reference.diagnostics["floor_hits"]


@pytest.mark.slow
@pytest.mark.parametrize("parameter_name", ["crypto", "equity"])
def test_exact_q_discounted_spot_is_a_three_se_martingale(parameter_name: str) -> None:
    """The exact Gaussian variance compensator preserves the discounted-spot martingale."""
    paths = _simulate(
        parameter_name,
        TgarchMeasure.Q_EXACT,
        n_paths=65_536,
        seed=20260827,
    )
    params = _study_params(parameter_name)
    discounted_spot = paths.assets[:, -1, 0] * math.exp(-params.r * 0.25)
    error = float(np.mean(discounted_spot) - params.spot0)
    standard_error = _paired_standard_error(discounted_spot)

    assert abs(error) <= 3.0 * standard_error + 2.0e-13 * params.spot0


@pytest.mark.slow
@pytest.mark.parametrize("parameter_name", ["crypto", "equity"])
def test_raw_weighted_p_and_direct_exact_q_prices_agree_within_three_se(
    parameter_name: str,
) -> None:
    """An ATM call reproduces the companion study's paired direct-Q/weighted-P gate."""
    p_paths = _simulate(
        parameter_name,
        TgarchMeasure.P,
        n_paths=65_536,
        seed=20260825,
    )
    q_paths = _simulate(
        parameter_name,
        TgarchMeasure.Q_EXACT,
        n_paths=65_536,
        seed=20260825,
    )
    params = _study_params(parameter_name)
    discount = math.exp(-params.r * 0.25)
    likelihood_ratios = _stable_raw_weights(p_paths.log_likelihood_ratios)
    weighted_p_pairs = _controlled_call_pair_values(
        p_paths.assets[:, -1, 0],
        strike=params.spot0,
        discount=discount,
        expected_discounted_spot=params.spot0,
        weights=likelihood_ratios,
    )
    direct_q_pairs = _controlled_call_pair_values(
        q_paths.assets[:, -1, 0],
        strike=params.spot0,
        discount=discount,
        expected_discounted_spot=params.spot0,
    )
    differences = direct_q_pairs - weighted_p_pairs
    error = float(np.mean(differences))
    standard_error = float(np.std(differences, ddof=1) / math.sqrt(differences.size))

    assert abs(error) <= 3.0 * standard_error + 2.0e-13 * params.spot0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"maturity": 0.0}, "maturity"),
        ({"max_dt": 0.0}, "max_dt"),
        ({"n_paths": True}, "n_paths"),
        ({"seed": -1}, "seed"),
        ({"chunk_steps": 0}, "chunk_steps"),
    ],
)
def test_terminal_request_validation(changes: dict[str, object], message: str) -> None:
    """Grid, path count, seed and chunking reject ambiguous or inadmissible values."""
    kwargs = {
        "measure": TgarchMeasure.P,
        "maturity": 0.25,
        "max_dt": 1.0 / 52.0,
        "n_paths": 8,
        "seed": 20260825,
        "chunk_steps": 8,
    }
    kwargs.update(changes)

    with pytest.raises(ValueError, match=message):
        TgarchModel(_study_params("equity")).simulate_paths(**kwargs)


def test_exact_q_rejects_an_inadmissible_variance_kernel() -> None:
    """The finite-step Gaussian law requires positive conditional variance pathwise."""
    params = replace(_study_params("equity"), eta0=4.0, eta1=0.0)

    with pytest.raises(ValueError, match="exact-Q kernel is inadmissible"):
        TgarchModel(params).simulate_paths(
            measure=TgarchMeasure.Q_EXACT,
            maturity=0.25,
            max_dt=1.0 / 52.0,
            n_paths=8,
            seed=20260825,
        )
