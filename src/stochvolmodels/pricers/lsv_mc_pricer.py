"""European LSV vanilla prices and sampling errors on the original option-chain strikes.

Examples
--------
Price a call under deterministic volatility with a seeded antithetic simulation.

>>> from stochvolmodels.models.lsv import LSVModel
>>> from stochvolmodels.pricers.lsv_mc_pricer import LSVMonteCarloPricer
>>> from stochvolmodels.models.lsv.simulation import LSVMonteCarloConfig
>>> from stochvolmodels.models.logsv.params import LogSvParams
>>> model = LSVModel.from_params(LogSvParams(sigma0=.2, theta=.2, beta=0., volvol=0.))
>>> pricer = LSVMonteCarloPricer(config=LSVMonteCarloConfig(paths=8192, steps_per_year=24))
>>> price, iv = pricer.price_vanilla(model, 0.25, 100., 100., "C")
>>> bool(3.8 < price < 4.2 and abs(iv - 0.2) < 0.012)
True
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from statistics import NormalDist

import numpy as np
from numba.typed import List

from stochvolmodels.data.model_paths import ModelPaths
from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.logsv import LogSvMeasure
from stochvolmodels.models.lsv import LSVModel
from stochvolmodels.models.lsv.simulation import LSVMonteCarloConfig, LSVMonteCarloEngine
from stochvolmodels.pricers.model_pricer import ModelPricer
from stochvolmodels.products.payoffs import _vanilla_payoff
from stochvolmodels.utils.config import OptionType, VariableType
from stochvolmodels.valuation import _mean_and_standard_error


@dataclass(frozen=True, slots=True)
class LSVMonteCarloExpiry:
    """Price confidence interval by strike and forward martingale diagnostics in quote units."""

    maturity: float
    price_lower: np.ndarray
    price_upper: np.ndarray
    forward_mean: float
    forward_standard_error: float


@dataclass(frozen=True, slots=True)
class LSVMonteCarloResult:
    """Raw discounted PVs, pair-aware standard errors, Black IVs and their simulated paths."""

    prices: List
    standard_errors: List
    implied_vols: List
    invalid_iv: tuple[np.ndarray, ...]
    expiries: tuple[LSVMonteCarloExpiry, ...]
    paths: ModelPaths


def _estimate(samples: np.ndarray, antithetic: bool) -> tuple[float, float]:
    """Estimate the mean and standard error from independent draws or pair means."""
    if antithetic:
        half = len(samples) // 2
        samples = 0.5 * samples[:half] + 0.5 * samples[half:]
    return _mean_and_standard_error(samples)


@dataclass(kw_only=True)
class LSVMonteCarloPricer(ModelPricer):
    """Price all ordinary calls and puts on a chain using one common set of Q_MMA paths."""

    config: LSVMonteCarloConfig = field(default_factory=LSVMonteCarloConfig)
    confidence_level: float = 0.95

    def _project_chain(
        self,
        option_chain: OptionChain,
        params: LSVModel,
        *,
        measure: LogSvMeasure | str = LogSvMeasure.MMA,
        is_spot_measure: bool = True,
        variable_type: VariableType = VariableType.LOG_RETURN,
        nb_path: int | None = None,
    ) -> tuple[List, List, tuple[LSVMonteCarloExpiry, ...], ModelPaths]:
        """Integrate raw terminal payoffs against the simulated normalized-forward law."""
        if not isinstance(option_chain, OptionChain):
            raise TypeError("option_chain must be an OptionChain")
        if is_spot_measure is not True or LogSvMeasure(measure) is not LogSvMeasure.MMA:
            raise ValueError("LSVMonteCarloPricer requires the money-market measure")
        if variable_type is not VariableType.LOG_RETURN:
            raise ValueError("LSVMonteCarloPricer requires VariableType.LOG_RETURN")
        if any(not np.all(np.isin(kinds, ["C", "P"])) for kinds in option_chain.optiontypes_ttms):
            raise ValueError("LSVMonteCarloPricer supports only ordinary C and P contracts")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must lie strictly between zero and one")
        if not isinstance(self.config, LSVMonteCarloConfig):
            raise TypeError("config must be an LSVMonteCarloConfig")
        config = self.config if nb_path is None else replace(self.config, paths=nb_path)
        if config.paths < (4 if config.antithetic else 2):
            raise ValueError("pricing requires at least two independent paths or antithetic pairs")
        quantile = -NormalDist().inv_cdf(0.5 * (1.0 - self.confidence_level))
        paths = LSVMonteCarloEngine(params, config).simulate_paths(
            observation_times=np.r_[0.0, option_chain.ttms],
            measure=measure,
        )
        prices, errors, expiries = List(), List(), []
        for i, maturity in enumerate(option_chain.ttms):
            underlying = option_chain.forwards[i] * paths.assets[:, i + 1, 0]
            discount = option_chain.discfactors[i]
            strikes, kinds = option_chain.strikes_ttms[i], option_chain.optiontypes_ttms[i]
            pv, se = np.empty(len(strikes)), np.empty(len(strikes))
            for j, (strike, kind) in enumerate(zip(strikes, kinds)):
                payoff = _vanilla_payoff(underlying, strike=strike, option_type=OptionType(kind))
                mean, error = _estimate(payoff, config.antithetic)
                pv[j], se[j] = discount * mean, discount * error
            prices.append(pv)
            errors.append(se)
            forward, forward_error = _estimate(underlying, config.antithetic)
            expiries.append(
                LSVMonteCarloExpiry(
                    maturity, pv - quantile * se, pv + quantile * se, forward, forward_error
                )
            )
        return prices, errors, tuple(expiries), paths

    def price_chain(self, option_chain: OptionChain, params: LSVModel, **kwargs) -> List:
        """Return raw discounted prices in original expiry and quote order."""
        prices, _, _, _ = self._project_chain(option_chain, params, **kwargs)
        return prices

    def model_mc_price_chain(
        self,
        option_chain: OptionChain,
        params: LSVModel,
        variable_type: VariableType = VariableType.LOG_RETURN,
        **kwargs,
    ) -> tuple[List, List]:
        """Return prices and standard errors through the ModelPricer Monte Carlo interface."""
        prices, errors, _, _ = self._project_chain(
            option_chain, params, variable_type=variable_type, **kwargs
        )
        return prices, errors

    def price_chain_detailed(
        self, option_chain: OptionChain, params: LSVModel, **kwargs
    ) -> LSVMonteCarloResult:
        """Also return confidence intervals, raw-price IV inversion masks and common paths."""
        prices, errors, expiries, paths = self._project_chain(option_chain, params, **kwargs)
        ivs = option_chain.compute_model_ivols_from_chain_data(prices)
        return LSVMonteCarloResult(
            prices, errors, ivs, tuple(~np.isfinite(v) for v in ivs), expiries, paths
        )
