"""
Benchmark rough LogSV pricing over parameter perturbations.
"""

from __future__ import annotations

import gc
import time
import tracemalloc
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import stochvolmodels.pricers.logsv_pricer as sv
from stochvolmodels import LogSvParams
from stochvolmodels.data.test_option_chain import get_btc_test_chain_data
from stochvolmodels.utils.funcs import set_time_grid

import psutil


@dataclass(frozen=True)
class PerfResult:
    name: str
    cpu_seconds: float
    mem_current_kib: float
    mem_peak_kib: float
    rss_before_kib: Optional[float]
    rss_after_kib: Optional[float]
    rss_delta_kib: Optional[float]
    rss_peak_delta_kib: Optional[float]
    rss_peak_kib: Optional[float]
    checksum: float


def _perturb_value(value: float, rel: float) -> float:
    return value * (1.0 + rel)


def build_param_variants(base: LogSvParams,
                         count: int = 100,
                         rel_scale: float = 0.02,
                         seed: int = 123,
                         param_dtype: Optional[np.dtype] = None) -> List[Tuple[str, LogSvParams]]:
    """Create a deterministic list of slightly perturbed parameter sets."""
    if param_dtype is not None:
        param_dtype = np.dtype(param_dtype)
    rng = np.random.default_rng(seed)
    keys = ("sigma0", "theta", "kappa1", "kappa2", "beta", "volvol")
    variants: List[Tuple[str, LogSvParams]] = []

    for i in range(count):
        params = LogSvParams.copy(base)
        deltas = rng.uniform(-rel_scale, rel_scale, size=len(keys))
        # Ensure a non-zero perturbation for each variant.
        if np.all(np.abs(deltas) < 1e-12):
            deltas[0] = rel_scale
        for key, rel in zip(keys, deltas):
            value = _perturb_value(getattr(base, key), rel)
            if param_dtype is not None:
                value = param_dtype.type(value)
            setattr(params, key, value)
        # Keep kernel approximation constant across variants.
        params.H = base.H
        params.weights = np.array(base.weights, copy=True)
        params.nodes = np.array(base.nodes, copy=True)
        variants.append((f"var-{i:03d}", params))
    return variants


def _rough_price_once(option_chain,
                      Z0: Optional[np.ndarray],
                      Z1: Optional[np.ndarray],
                      grid_ttms,
                      params: LogSvParams,
                      nb_path: int,
                      seed: int,
                      random_mode: str,
                      float32_mode: bool):
    sigma0 = params.sigma0
    theta = params.theta
    kappa1 = params.kappa1
    kappa2 = params.kappa2
    beta = params.beta
    volvol = params.volvol
    if float32_mode:
        sigma0 = np.float32(sigma0)
        theta = np.float32(theta)
        kappa1 = np.float32(kappa1)
        kappa2 = np.float32(kappa2)
        beta = np.float32(beta)
        volvol = np.float32(volvol)
    if random_mode == "seeded":
        return sv.rough_logsv_mc_chain_pricer_seeded(
            ttms=option_chain.ttms,
            forwards=option_chain.forwards,
            discfactors=option_chain.discfactors,
            strikes_ttms=option_chain.strikes_ttms,
            optiontypes_ttms=option_chain.optiontypes_ttms,
            nb_path=nb_path,
            seed=seed,
            sigma0=sigma0,
            theta=theta,
            kappa1=kappa1,
            kappa2=kappa2,
            beta=beta,
            orthog_vol=volvol,
            weights=params.weights,
            nodes=params.nodes,
            timegrids=grid_ttms,
        )
    return sv.rough_logsv_mc_chain_pricer_fixed_randoms(
        ttms=option_chain.ttms,
        forwards=option_chain.forwards,
        discfactors=option_chain.discfactors,
        strikes_ttms=option_chain.strikes_ttms,
        optiontypes_ttms=option_chain.optiontypes_ttms,
        Z0=Z0,
        Z1=Z1,
        sigma0=sigma0,
        theta=theta,
        kappa1=kappa1,
        kappa2=kappa2,
        beta=beta,
        orthog_vol=volvol,
        weights=params.weights,
        nodes=params.nodes,
        timegrids=grid_ttms,
    )


def _get_rss_kib() -> Optional[float]:
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / 1024.0


class _RssSampler:
    def __init__(self, interval_sec: float):
        self.interval_sec = interval_sec
        self.max_rss_kib: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        proc = psutil.Process()
        while not self._stop.is_set():
            rss = proc.memory_info().rss / 1024.0
            if self.max_rss_kib is None or rss > self.max_rss_kib:
                self.max_rss_kib = rss
            time.sleep(self.interval_sec)

    def start(self) -> None:
        if psutil is None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()


def run_rough_logsv_loop_bench(nb_path: int = 50000,
                               nb_steps_per_year: int = 240,
                               seed: int = 10,
                               dtype: np.dtype = np.float32,
                               warmup: int = 1,
                               variants: int = 100,
                               rss_sample_interval: float = 0.01,
                               random_mode: str = "fixed",
                               float32_mode: bool = True) -> pd.DataFrame:
    option_chain = get_btc_test_chain_data()

    if random_mode not in ("fixed", "seeded"):
        raise ValueError("random_mode must be 'fixed' or 'seeded'")

    Z0 = None
    Z1 = None
    grid_ttms: List[np.ndarray] = []
    if random_mode == "fixed":
        try:
            Z0, Z1, grid_ttms = sv.get_randoms_for_rough_vol_chain_valuation(
                ttms=option_chain.ttms,
                nb_path=nb_path,
                nb_steps_per_year=nb_steps_per_year,
                seed=seed,
                dtype=dtype,
                chunk_size=2048,
                use_legacy_rng=False,
            )
        except TypeError:
            # Backward-compatibility: older signatures may not support dtype/chunk_size/use_legacy_rng.
            Z0, Z1, grid_ttms = sv.get_randoms_for_rough_vol_chain_valuation(
                ttms=option_chain.ttms,
                nb_path=nb_path,
                nb_steps_per_year=nb_steps_per_year,
                seed=seed,
            )
            if Z0.dtype != dtype:
                Z0 = Z0.astype(dtype, copy=False)
            if Z1.dtype != dtype:
                Z1 = Z1.astype(dtype, copy=False)
        if float32_mode:
            grid_ttms = [np.asarray(g, dtype=np.float32) for g in grid_ttms]
    else:
        for ttm in option_chain.ttms:
            _, _, grid_t = set_time_grid(ttm=ttm, nb_steps_per_year=nb_steps_per_year)
            if float32_mode:
                grid_t = np.asarray(grid_t, dtype=np.float32)
            grid_ttms.append(grid_t)

    base_params = LogSvParams(sigma0=0.8, theta=1.0, kappa1=2.21, kappa2=2.18, beta=0.15, volvol=2.0)
    base_params.H = 0.1
    base_params.approximate_kernel(T=option_chain.ttms[-1])
    if float32_mode:
        base_params.weights = base_params.weights.astype(np.float32, copy=False)
        base_params.nodes = base_params.nodes.astype(np.float32, copy=False)
    param_dtype = np.float32 if float32_mode else None
    variants = build_param_variants(base=base_params, count=variants, param_dtype=param_dtype)

    # Warm up numba kernels before timing.
    for _ in range(max(0, warmup)):
        _rough_price_once(option_chain, Z0, Z1, grid_ttms, variants[0][1], nb_path, seed, random_mode, float32_mode)

    tracemalloc.start()
    results: List[PerfResult] = []

    for name, params in variants:
        gc.collect()
        tracemalloc.reset_peak()
        mem_before, _ = tracemalloc.get_traced_memory()
        rss_before = _get_rss_kib()
        sampler = None
        if rss_sample_interval > 0.0 and psutil is not None:
            sampler = _RssSampler(interval_sec=rss_sample_interval)
            sampler.start()
        cpu_start = time.process_time()

        option_prices_ttm, option_std_ttm = _rough_price_once(option_chain, Z0, Z1, grid_ttms, params, nb_path, seed, random_mode, float32_mode)

        cpu_end = time.process_time()
        if sampler is not None:
            sampler.stop()
        mem_after, mem_peak = tracemalloc.get_traced_memory()
        rss_after = _get_rss_kib()
        rss_peak_kib = None
        if sampler is not None and sampler.max_rss_kib is not None:
            rss_peak_kib = sampler.max_rss_kib
        elif rss_before is not None and rss_after is not None:
            rss_peak_kib = max(rss_before, rss_after)

        # Basic checksum to keep outputs used.
        checksum = 0.0
        for arr in option_prices_ttm:
            checksum += float(np.sum(np.asarray(arr)))

        results.append(
            PerfResult(
                name=name,
                cpu_seconds=cpu_end - cpu_start,
                mem_current_kib=(mem_after - mem_before) / 1024.0,
                mem_peak_kib=(mem_peak - mem_before) / 1024.0,
                rss_before_kib=rss_before,
                rss_after_kib=rss_after,
                rss_delta_kib=None if rss_before is None or rss_after is None else rss_after - rss_before,
                rss_peak_delta_kib=None if rss_before is None or rss_peak_kib is None else rss_peak_kib - rss_before,
                rss_peak_kib=rss_peak_kib,
                checksum=checksum,
            )
        )

        del option_prices_ttm, option_std_ttm

    tracemalloc.stop()

    df = pd.DataFrame([r.__dict__ for r in results])
    cols = [
        "name",
        "cpu_seconds",
        "mem_current_kib",
        "mem_peak_kib",
        "rss_before_kib",
        "rss_after_kib",
        "rss_peak_kib",
        "rss_delta_kib",
        "rss_peak_delta_kib",
        "checksum",
    ]
    return df[cols]


if __name__ == "__main__":
    df_results = run_rough_logsv_loop_bench(rss_sample_interval=0.01)
    pd.set_option("display.max_columns", None)
    print(df_results.to_string(index=False))
