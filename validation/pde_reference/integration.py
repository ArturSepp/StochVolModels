"""
Private OptionChain adapters for the staged lognormal-LSV pipeline.

Adapters normalize ragged `OptionChain` quotes into calibration targets without silent
extrapolation, then map historical Monte Carlo results back to the original chain layout. Price
clipping, Black inversion bounds, confidence intervals, and martingale diagnostics remain
visible in each result slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray
from vanilla_option_pricers import (
    compute_bsm_vanilla_slice_prices,
    infer_bsm_ivols_from_model_slice_prices,
)

from stochvolmodels.data.option_chain import OptionChain
from validation.pde_reference.calibration import ImpliedVolCalibrationTarget
from validation.pde_reference.lsv_mc import (
    LSVMonteCarloPricer,
    LSVMonteCarloPricingResult,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
StringArray = NDArray[np.str_]


class LSVQuoteSelection(str, Enum):
    """Market quotes retained when constructing one normalized IV surface."""

    OTM = "otm"
    ALL = "all"


@dataclass(frozen=True, slots=True)
class LSVCalibrationInput:
    """Normalized calibration target plus an immutable market-data audit trail."""

    target: ImpliedVolCalibrationTarget
    log_moneyness: FloatArray
    source_forwards: FloatArray
    source_discount_factors: FloatArray
    slice_ids: tuple[str, ...]
    source_log_moneyness: tuple[FloatArray, ...]
    source_bid_implied_volatilities: tuple[FloatArray, ...]
    source_ask_implied_volatilities: tuple[FloatArray, ...]
    source_mid_implied_volatilities: tuple[FloatArray, ...]
    quote_selection: LSVQuoteSelection

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVCalibrationInput`` instance."""
        if not isinstance(self.target, ImpliedVolCalibrationTarget):
            raise TypeError("target must be an ImpliedVolCalibrationTarget")
        log_moneyness = _readonly_vector(self.log_moneyness, "log_moneyness")
        forwards = _readonly_vector(self.source_forwards, "source_forwards")
        discounts = _readonly_vector(
            self.source_discount_factors, "source_discount_factors"
        )
        if (
            log_moneyness.size != self.target.strikes.size
            or np.any(np.diff(log_moneyness) <= 0.0)
        ):
            raise ValueError("log_moneyness must be strictly increasing and match target strikes")
        if not np.allclose(
            self.target.strikes, np.exp(log_moneyness), rtol=2.0e-15, atol=0.0
        ):
            raise ValueError("target strikes must equal exp(log_moneyness)")
        maturity_count = self.target.times.size
        if forwards.shape != (maturity_count,) or discounts.shape != (maturity_count,):
            raise ValueError("source forward and discount vectors must match target maturities")
        if np.any(forwards <= 0.0) or np.any(discounts <= 0.0):
            raise ValueError("source forwards and discount factors must be positive")
        slice_ids = tuple(str(value) for value in self.slice_ids)
        if len(slice_ids) != maturity_count:
            raise ValueError("slice_ids must contain one label per maturity")
        source_coordinates = _readonly_vector_tuple(
            self.source_log_moneyness, "source_log_moneyness"
        )
        source_bids = _readonly_vector_tuple(
            self.source_bid_implied_volatilities,
            "source_bid_implied_volatilities",
        )
        source_asks = _readonly_vector_tuple(
            self.source_ask_implied_volatilities,
            "source_ask_implied_volatilities",
        )
        source_mids = _readonly_vector_tuple(
            self.source_mid_implied_volatilities,
            "source_mid_implied_volatilities",
        )
        if not (
            len(source_coordinates)
            == len(source_bids)
            == len(source_asks)
            == len(source_mids)
            == maturity_count
        ):
            raise ValueError("source quote tuples must contain one array per maturity")
        for coordinates, bids, asks, mids in zip(
            source_coordinates, source_bids, source_asks, source_mids
        ):
            if not (coordinates.shape == bids.shape == asks.shape == mids.shape):
                raise ValueError("source coordinates and quote arrays must be aligned")
            if coordinates.size < 2 or np.any(np.diff(coordinates) <= 0.0):
                raise ValueError("source log-moneyness arrays must be strictly increasing")
            if np.any(bids <= 0.0) or np.any(bids > asks):
                raise ValueError("source bid/ask implied volatilities must be positive and ordered")
            if not np.allclose(mids, 0.5 * (bids + asks), rtol=1.0e-14):
                raise ValueError("source mid implied volatilities must equal bid/ask mids")
        if not isinstance(self.quote_selection, LSVQuoteSelection):
            raise TypeError("quote_selection must be an LSVQuoteSelection")
        object.__setattr__(self, "log_moneyness", log_moneyness)
        object.__setattr__(self, "source_forwards", forwards)
        object.__setattr__(self, "source_discount_factors", discounts)
        object.__setattr__(self, "slice_ids", slice_ids)
        object.__setattr__(self, "source_log_moneyness", source_coordinates)
        object.__setattr__(self, "source_bid_implied_volatilities", source_bids)
        object.__setattr__(self, "source_ask_implied_volatilities", source_asks)
        object.__setattr__(self, "source_mid_implied_volatilities", source_mids)


@dataclass(frozen=True, slots=True)
class LSVOptionChainAdapter:
    """Convert ragged market quotes into a common normalized-forward target."""

    option_chain: OptionChain
    quote_selection: LSVQuoteSelection = LSVQuoteSelection.OTM

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVOptionChainAdapter`` instance."""
        if not isinstance(self.option_chain, OptionChain):
            raise TypeError("option_chain must be an OptionChain")
        if not isinstance(self.quote_selection, LSVQuoteSelection):
            raise TypeError("quote_selection must be an LSVQuoteSelection")
        if self.option_chain.bid_ivs is None or self.option_chain.ask_ivs is None:
            raise ValueError("OptionChain bid and ask implied volatilities are required")
        self._selected_quotes()

    def common_log_moneyness_grid(self, size: int = 21) -> FloatArray:
        """Return an increasing grid inside every selected market slice."""
        point_count = _positive_integer(size, "size")
        if point_count < 2:
            raise ValueError("size must be at least two")
        selected = self._selected_quotes()
        lower = max(item[0][0] for item in selected)
        upper = min(item[0][-1] for item in selected)
        if not lower < upper:
            raise ValueError("selected quote slices have no common log-moneyness domain")
        grid = np.linspace(lower, upper, point_count)
        if lower <= 0.0 <= upper:
            grid[int(np.argmin(np.abs(grid)))] = 0.0
        if np.any(np.diff(grid) <= 0.0):
            raise RuntimeError("failed to construct a strictly increasing common grid")
        return _readonly_vector(grid, "common_log_moneyness_grid")

    def build_calibration_input(
        self,
        log_moneyness: ArrayLike | None = None,
        *,
        grid_size: int = 21,
    ) -> LSVCalibrationInput:
        """Interpolate selected mid IVs without leaving the observed common domain."""
        selected = self._selected_quotes()
        if log_moneyness is None:
            common_grid = self.common_log_moneyness_grid(grid_size)
        else:
            common_grid = _readonly_vector(log_moneyness, "log_moneyness")
            if common_grid.size < 2 or np.any(np.diff(common_grid) <= 0.0):
                raise ValueError(
                    "log_moneyness must be strictly increasing with at least two points"
                )
        lower = max(item[0][0] for item in selected)
        upper = min(item[0][-1] for item in selected)
        tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(lower), abs(upper))
        if common_grid[0] < lower - tolerance or common_grid[-1] > upper + tolerance:
            raise ValueError("log_moneyness is outside the common observed domain")

        source_coordinates = tuple(item[0] for item in selected)
        source_bids = tuple(item[1] for item in selected)
        source_asks = tuple(item[2] for item in selected)
        source_mids = tuple(item[3] for item in selected)
        target_implied = np.vstack(
            [
                np.interp(common_grid, coordinates, mids)
                for coordinates, mids in zip(source_coordinates, source_mids)
            ]
        )
        chain = self.option_chain
        slice_ids = (
            tuple(str(value) for value in chain.ids)
            if chain.ids is not None
            else tuple(f"T={maturity:.6g}" for maturity in chain.ttms)
        )
        target = ImpliedVolCalibrationTarget(
            times=np.asarray(chain.ttms, dtype=np.float64),
            strikes=np.exp(common_grid),
            implied_volatilities=target_implied,
            forward=1.0,
            discount_factor=1.0,
        )
        return LSVCalibrationInput(
            target=target,
            log_moneyness=common_grid,
            source_forwards=chain.forwards,
            source_discount_factors=chain.discfactors,
            slice_ids=slice_ids,
            source_log_moneyness=source_coordinates,
            source_bid_implied_volatilities=source_bids,
            source_ask_implied_volatilities=source_asks,
            source_mid_implied_volatilities=source_mids,
            quote_selection=self.quote_selection,
        )

    def _selected_quotes(
        self,
    ) -> tuple[tuple[FloatArray, FloatArray, FloatArray, FloatArray], ...]:
        """Select and validate market quotes used by one calibration slice."""
        chain = self.option_chain
        selected: list[tuple[FloatArray, FloatArray, FloatArray, FloatArray]] = []
        for maturity_index, (forward, strikes, option_types, bids, asks) in enumerate(
            zip(
                chain.forwards,
                chain.strikes_ttms,
                chain.optiontypes_ttms,
                chain.bid_ivs,
                chain.ask_ivs,
            )
        ):
            strike_array = np.asarray(strikes, dtype=np.float64)
            type_array = np.asarray(option_types).astype(str)
            bid_array = np.asarray(bids, dtype=np.float64)
            ask_array = np.asarray(asks, dtype=np.float64)
            if np.any(~np.isin(type_array, ("C", "P"))):
                raise ValueError("LSV calibration accepts only standard C/P option quotes")
            if self.quote_selection is LSVQuoteSelection.OTM:
                mask = ((type_array == "P") & (strike_array < forward)) | (
                    (type_array == "C") & (strike_array >= forward)
                )
            else:
                mask = np.ones(strike_array.size, dtype=np.bool_)
            if np.count_nonzero(mask) < 2:
                raise ValueError(
                    f"maturity index {maturity_index} has fewer than two selected quotes"
                )
            coordinates = np.log(strike_array[mask] / forward)
            order = np.argsort(coordinates, kind="stable")
            coordinates = coordinates[order]
            selected_bids = bid_array[mask][order]
            selected_asks = ask_array[mask][order]
            if np.any(np.diff(coordinates) <= 0.0):
                raise ValueError(
                    f"maturity index {maturity_index} has duplicate normalized strikes"
                )
            selected.append(
                (
                    _readonly_vector(coordinates, "source_log_moneyness"),
                    _readonly_vector(selected_bids, "source_bid_implied_volatilities"),
                    _readonly_vector(selected_asks, "source_ask_implied_volatilities"),
                    _readonly_vector(
                        0.5 * (selected_bids + selected_asks),
                        "source_mid_implied_volatilities",
                    ),
                )
            )
        return tuple(selected)


@dataclass(frozen=True, slots=True)
class LSVOptionChainSliceResult:
    """One ragged OptionChain slice with price, IV, and MC diagnostics."""

    slice_id: str
    maturity: float
    forward: float
    discount_factor: float
    strikes: FloatArray
    option_types: StringArray
    market_bid_implied_volatilities: FloatArray | None
    market_ask_implied_volatilities: FloatArray | None
    model_prices: FloatArray
    standard_errors: FloatArray
    price_confidence_lower: FloatArray
    price_confidence_upper: FloatArray
    black_inversion_price_lower_bound: FloatArray
    black_inversion_price_upper_bound: FloatArray
    model_implied_volatilities: FloatArray
    implied_volatility_confidence_lower: FloatArray
    implied_volatility_confidence_upper: FloatArray
    model_price_clipped_mask: BoolArray
    confidence_price_clipped_mask: BoolArray
    forward_mean: float
    forward_standard_error: float
    forward_confidence_interval: tuple[float, float]
    forward_martingale_z_score: float
    mean_jump_count: float

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVOptionChainSliceResult`` instance."""
        maturity = _positive_float(self.maturity, "maturity")
        forward = _positive_float(self.forward, "forward")
        discount = _positive_float(self.discount_factor, "discount_factor")
        strikes = _readonly_vector(self.strikes, "strikes")
        option_types = _readonly_option_types(self.option_types)
        if strikes.size == 0 or np.any(strikes <= 0.0) or option_types.shape != strikes.shape:
            raise ValueError("strikes and option_types must be aligned non-empty vectors")
        if np.any(~np.isin(option_types, ("C", "P"))):
            raise ValueError("option_types must contain only C/P")
        vector_names = (
            "model_prices",
            "standard_errors",
            "price_confidence_lower",
            "price_confidence_upper",
            "black_inversion_price_lower_bound",
            "black_inversion_price_upper_bound",
            "model_implied_volatilities",
            "implied_volatility_confidence_lower",
            "implied_volatility_confidence_upper",
        )
        vectors = {
            name: _readonly_vector(getattr(self, name), name) for name in vector_names
        }
        if any(values.shape != strikes.shape for values in vectors.values()):
            raise ValueError("all pricing and implied-volatility vectors must match strikes")
        masks = {
            "model_price_clipped_mask": _readonly_mask(
                self.model_price_clipped_mask, "model_price_clipped_mask"
            ),
            "confidence_price_clipped_mask": _readonly_mask(
                self.confidence_price_clipped_mask,
                "confidence_price_clipped_mask",
            ),
        }
        if any(values.shape != strikes.shape for values in masks.values()):
            raise ValueError("price clipping masks must match strikes")
        bid = _optional_readonly_vector(
            self.market_bid_implied_volatilities,
            "market_bid_implied_volatilities",
            strikes.shape,
        )
        ask = _optional_readonly_vector(
            self.market_ask_implied_volatilities,
            "market_ask_implied_volatilities",
            strikes.shape,
        )
        if (bid is None) != (ask is None):
            raise ValueError("market bid and ask implied volatilities must be supplied together")
        if bid is not None and (np.any(bid <= 0.0) or np.any(bid > ask)):
            raise ValueError("market bid/ask implied volatilities must be positive and ordered")
        if np.any(vectors["model_prices"] < 0.0) or np.any(
            vectors["standard_errors"] < 0.0
        ):
            raise ValueError("model prices and standard errors must be non-negative")
        if np.any(
            vectors["black_inversion_price_lower_bound"]
            >= vectors["black_inversion_price_upper_bound"]
        ):
            raise ValueError("Black inversion price bounds must be increasing")
        if np.any(
            vectors["implied_volatility_confidence_lower"]
            > vectors["implied_volatility_confidence_upper"]
        ):
            raise ValueError("implied-volatility confidence bounds must be ordered")
        forward_mean = _positive_float(self.forward_mean, "forward_mean")
        forward_error = _nonnegative_float(
            self.forward_standard_error, "forward_standard_error"
        )
        lower, upper = (float(value) for value in self.forward_confidence_interval)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            raise ValueError("forward_confidence_interval must be finite and ordered")
        martingale_z = float(self.forward_martingale_z_score)
        mean_jumps = _nonnegative_float(self.mean_jump_count, "mean_jump_count")
        if not np.isfinite(martingale_z):
            raise ValueError("forward_martingale_z_score must be finite")
        object.__setattr__(self, "slice_id", str(self.slice_id))
        object.__setattr__(self, "maturity", maturity)
        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "discount_factor", discount)
        object.__setattr__(self, "strikes", strikes)
        object.__setattr__(self, "option_types", option_types)
        object.__setattr__(self, "market_bid_implied_volatilities", bid)
        object.__setattr__(self, "market_ask_implied_volatilities", ask)
        for name, values in vectors.items():
            object.__setattr__(self, name, values)
        for name, values in masks.items():
            object.__setattr__(self, name, values)
        object.__setattr__(self, "forward_mean", forward_mean)
        object.__setattr__(self, "forward_standard_error", forward_error)
        object.__setattr__(self, "forward_confidence_interval", (lower, upper))
        object.__setattr__(self, "forward_martingale_z_score", martingale_z)
        object.__setattr__(self, "mean_jump_count", mean_jumps)


@dataclass(frozen=True, slots=True)
class LSVOptionChainPricingResult:
    """Ragged chain results backed by one common-path Monte Carlo simulation."""

    slices: tuple[LSVOptionChainSliceResult, ...]
    monte_carlo_result: LSVMonteCarloPricingResult

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVOptionChainPricingResult`` instance."""
        slices = tuple(self.slices)
        if not isinstance(self.monte_carlo_result, LSVMonteCarloPricingResult):
            raise TypeError("monte_carlo_result must be an LSVMonteCarloPricingResult")
        if len(slices) != self.monte_carlo_result.paths.times.size:
            raise ValueError("one OptionChain slice is required per simulated maturity")
        for index, result in enumerate(slices):
            if not isinstance(result, LSVOptionChainSliceResult):
                raise TypeError("slices must contain LSVOptionChainSliceResult objects")
            if not np.isclose(
                result.maturity,
                self.monte_carlo_result.paths.times[index],
                rtol=0.0,
                atol=1.0e-14,
            ):
                raise ValueError("slice maturities must match Monte Carlo path maturities")
        object.__setattr__(self, "slices", slices)

    @property
    def slice_ids(self) -> tuple[str, ...]:
        """Ordered source OptionChain identifiers."""
        return tuple(result.slice_id for result in self.slices)


@dataclass(frozen=True, slots=True)
class LSVOptionChainPricer:
    """Price one ragged OptionChain through a single LSV path simulation."""

    monte_carlo_pricer: LSVMonteCarloPricer

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVOptionChainPricer`` instance."""
        if not isinstance(self.monte_carlo_pricer, LSVMonteCarloPricer):
            raise TypeError("monte_carlo_pricer must be an LSVMonteCarloPricer")

    def price(
        self,
        option_chain: OptionChain,
        *,
        allow_time_extrapolation: bool = False,
    ) -> LSVOptionChainPricingResult:
        """Return prices and IVs in the chain's original ragged order."""
        if not isinstance(option_chain, OptionChain):
            raise TypeError("option_chain must be an OptionChain")
        if not isinstance(allow_time_extrapolation, bool):
            raise TypeError("allow_time_extrapolation must be bool")
        last_leverage_time = self.monte_carlo_pricer.leverage_surface.times[-1]
        if not allow_time_extrapolation and option_chain.ttms[-1] > last_leverage_time:
            raise ValueError(
                "OptionChain maturity exceeds the calibrated leverage horizon; "
                "set allow_time_extrapolation=True to clamp the final leverage bucket"
            )
        option_type_arrays = tuple(
            np.asarray(values).astype(str) for values in option_chain.optiontypes_ttms
        )
        if any(np.any(~np.isin(values, ("C", "P"))) for values in option_type_arrays):
            raise ValueError("LSV OptionChain pricing accepts only standard C/P option types")
        union_strikes = np.unique(
            np.concatenate(
                [np.asarray(values, dtype=np.float64) for values in option_chain.strikes_ttms]
            )
        )
        monte_carlo = self.monte_carlo_pricer.price(
            union_strikes,
            maturities=option_chain.ttms,
            forwards=option_chain.forwards,
            discount_factors=option_chain.discfactors,
        )
        selected_data: list[dict[str, FloatArray | BoolArray]] = []
        clipped_central: list[FloatArray] = []
        clipped_lower: list[FloatArray] = []
        clipped_upper: list[FloatArray] = []

        for index, (strikes, option_types) in enumerate(
            zip(option_chain.strikes_ttms, option_type_arrays)
        ):
            strike_array = np.asarray(strikes, dtype=np.float64)
            union_indices = np.searchsorted(union_strikes, strike_array)
            mc_slice = monte_carlo.slices[index]
            call_mask = option_types == "C"
            prices = np.where(
                call_mask,
                mc_slice.calls[union_indices],
                mc_slice.puts[union_indices],
            )
            errors = np.where(
                call_mask,
                mc_slice.call_standard_errors[union_indices],
                mc_slice.put_standard_errors[union_indices],
            )
            confidence_lower = np.where(
                call_mask,
                mc_slice.call_confidence_lower[union_indices],
                mc_slice.put_confidence_lower[union_indices],
            )
            confidence_upper = np.where(
                call_mask,
                mc_slice.call_confidence_upper[union_indices],
                mc_slice.put_confidence_upper[union_indices],
            )
            inversion_lower, inversion_upper = _black_inversion_price_bounds(
                maturity=float(option_chain.ttms[index]),
                forward=float(option_chain.forwards[index]),
                discount_factor=float(option_chain.discfactors[index]),
                strikes=strike_array,
                option_types=option_types,
            )
            central_safe, central_mask = _clip_prices(
                prices, inversion_lower, inversion_upper
            )
            lower_safe, lower_mask = _clip_prices(
                confidence_lower, inversion_lower, inversion_upper
            )
            upper_safe, upper_mask = _clip_prices(
                confidence_upper, inversion_lower, inversion_upper
            )
            selected_data.append(
                {
                    "prices": prices,
                    "errors": errors,
                    "confidence_lower": confidence_lower,
                    "confidence_upper": confidence_upper,
                    "inversion_lower": inversion_lower,
                    "inversion_upper": inversion_upper,
                    "central_mask": central_mask,
                    "confidence_mask": lower_mask | upper_mask,
                }
            )
            clipped_central.append(central_safe)
            clipped_lower.append(lower_safe)
            clipped_upper.append(upper_safe)

        central_iv = _implied_volatilities(option_chain, clipped_central)
        lower_iv = _implied_volatilities(option_chain, clipped_lower)
        upper_iv = _implied_volatilities(option_chain, clipped_upper)
        slice_ids = (
            tuple(str(value) for value in option_chain.ids)
            if option_chain.ids is not None
            else tuple(f"T={maturity:.6g}" for maturity in option_chain.ttms)
        )
        results: list[LSVOptionChainSliceResult] = []
        for index, data in enumerate(selected_data):
            mc_slice = monte_carlo.slices[index]
            results.append(
                LSVOptionChainSliceResult(
                    slice_id=slice_ids[index],
                    maturity=option_chain.ttms[index],
                    forward=option_chain.forwards[index],
                    discount_factor=option_chain.discfactors[index],
                    strikes=option_chain.strikes_ttms[index],
                    option_types=option_type_arrays[index],
                    market_bid_implied_volatilities=(
                        None
                        if option_chain.bid_ivs is None
                        else option_chain.bid_ivs[index]
                    ),
                    market_ask_implied_volatilities=(
                        None
                        if option_chain.ask_ivs is None
                        else option_chain.ask_ivs[index]
                    ),
                    model_prices=data["prices"],
                    standard_errors=data["errors"],
                    price_confidence_lower=data["confidence_lower"],
                    price_confidence_upper=data["confidence_upper"],
                    black_inversion_price_lower_bound=data["inversion_lower"],
                    black_inversion_price_upper_bound=data["inversion_upper"],
                    model_implied_volatilities=central_iv[index],
                    implied_volatility_confidence_lower=lower_iv[index],
                    implied_volatility_confidence_upper=upper_iv[index],
                    model_price_clipped_mask=data["central_mask"],
                    confidence_price_clipped_mask=data["confidence_mask"],
                    forward_mean=mc_slice.forward_mean,
                    forward_standard_error=mc_slice.forward_standard_error,
                    forward_confidence_interval=mc_slice.forward_confidence_interval,
                    forward_martingale_z_score=mc_slice.forward_martingale_z_score,
                    mean_jump_count=mc_slice.mean_jump_count,
                )
            )
        return LSVOptionChainPricingResult(tuple(results), monte_carlo)


def _black_inversion_price_bounds(
    *,
    maturity: float,
    forward: float,
    discount_factor: float,
    strikes: FloatArray,
    option_types: StringArray,
) -> tuple[FloatArray, FloatArray]:
    """Return arbitrage-consistent Black price bounds for one option slice."""
    lower_boundary = compute_bsm_vanilla_slice_prices(
        ttm=maturity,
        forward=forward,
        strikes=strikes,
        vols=np.full(strikes.size, 0.01),
        optiontypes=option_types,
        discfactor=discount_factor,
    )
    upper_boundary = compute_bsm_vanilla_slice_prices(
        ttm=maturity,
        forward=forward,
        strikes=strikes,
        vols=np.full(strikes.size, 5.0),
        optiontypes=option_types,
        discfactor=discount_factor,
    )
    width = upper_boundary - lower_boundary
    if np.any(~np.isfinite(width)) or np.any(width <= 0.0):
        raise RuntimeError("current Black inversion bounds are invalid")
    fraction = 1.0e-10
    lower = lower_boundary + fraction * width
    upper = upper_boundary - fraction * width
    return (
        _readonly_vector(lower, "black_inversion_price_lower_bound"),
        _readonly_vector(upper, "black_inversion_price_upper_bound"),
    )


def _clip_prices(
    values: ArrayLike, lower: FloatArray, upper: FloatArray
) -> tuple[FloatArray, BoolArray]:
    """Clip model prices to inversion bounds and report which entries changed."""
    prices = np.asarray(values, dtype=np.float64)
    clipped = np.clip(prices, lower, upper)
    mask = clipped != prices
    return (
        _readonly_vector(clipped, "clipped_prices"),
        _readonly_mask(mask, "price_clipped_mask"),
    )


def _implied_volatilities(
    option_chain: OptionChain, prices: list[FloatArray]
) -> tuple[FloatArray, ...]:
    """Invert bounded option prices into Black implied-volatility slices."""
    results: list[FloatArray] = []
    for maturity, forward, discount, strikes, option_types, price_slice in zip(
        option_chain.ttms,
        option_chain.forwards,
        option_chain.discfactors,
        option_chain.strikes_ttms,
        option_chain.optiontypes_ttms,
        prices,
    ):
        implied = infer_bsm_ivols_from_model_slice_prices(
            ttm=float(maturity),
            forward=float(forward),
            strikes=np.asarray(strikes, dtype=np.float64),
            optiontypes=np.asarray(option_types).astype(str),
            model_prices=np.asarray(price_slice, dtype=np.float64),
            discfactor=float(discount),
        )
        if np.any(~np.isfinite(implied)):
            raise RuntimeError("Black implied-volatility inversion failed after price clipping")
        results.append(_readonly_vector(implied, "model_implied_volatilities"))
    return tuple(results)


def _readonly_vector_tuple(
    values: tuple[ArrayLike, ...], name: str
) -> tuple[FloatArray, ...]:
    """Validate vector tuple and return an immutable NumPy value."""
    return tuple(_readonly_vector(value, name) for value in values)


def _optional_readonly_vector(
    values: ArrayLike | None, name: str, shape: tuple[int, ...]
) -> FloatArray | None:
    """Validate an optional vector and make its values immutable."""
    if values is None:
        return None
    result = _readonly_vector(values, name)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return result


def _readonly_vector(values: ArrayLike, name: str) -> FloatArray:
    """Validate vector and return an immutable NumPy value."""
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    array.setflags(write=False)
    return array


def _readonly_option_types(values: ArrayLike) -> StringArray:
    """Validate option types and return an immutable NumPy value."""
    array = np.array(values, dtype="U2", copy=True, order="C")
    if array.ndim != 1:
        raise ValueError("option_types must be a one-dimensional array")
    array.setflags(write=False)
    return array


def _readonly_mask(values: ArrayLike, name: str) -> BoolArray:
    """Validate mask and return an immutable NumPy value."""
    array = np.array(values, dtype=np.bool_, copy=True, order="C")
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional mask")
    array.setflags(write=False)
    return array


def _positive_integer(value: int, name: str) -> int:
    """Validate and return strictly positive integer."""
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value: float, name: str) -> float:
    """Validate and return strictly positive float."""
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative_float(value: float, name: str) -> float:
    """Validate and return nonnegative float."""
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result
