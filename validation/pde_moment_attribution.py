"""Attribute the frozen two-year 2D volatility error with the original full BE operator."""

from __future__ import annotations

import ctypes
from hashlib import sha256
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from stochvolmodels.pde_solvers import Grid1D, TensorGrid2D
from validation.pde_monotone import Monotone2DReference
from validation.pde_qualification import (
    Variant, _hash, _json_value, _price_time_method, density_checks, input_snapshot,
)
from validation.pde_unsplit import UnsplitConfig
from validation.pde_volatility_moments import (
    CDF_CUTS, FROZEN_HASH, TAIL_CUTS, TIMES, _observations,
)

REGIONS = ("sigma_zero", "sigma_upper", "x_faces", "interior")
FUNCTIONS = ("one", "sigma", "sigma_squared")
ARCHIVE = {"d8": "d8_s2", "d16": "d16_s2_g108", "d32": "d32_s2_g108",
           "tail_grade": "d16_s2_g115", "time": "d16_s2_g108_t2"}


def load_archive(directory):
    """Check the input, generator configuration and exact grid of each historical solve."""
    records = {}
    for key, variant in ARCHIVE.items():
        path = directory / f"two_year_volatility_{variant}.json"
        record = json.loads(path.read_bytes())
        run = record["run"]
        if not (record["input_sha256_after"] == record["input_sha256"]
                == _hash(record["input"]) == FROZEN_HASH
                and record["model_sha256"] == _hash(record["input"]["model"])
                and run["grid_sha256"] == _hash(record["grid"])
                and run["backend"] == "positive" and run["status"] == "ok"
                and run["variant"] == variant and run["actual_steps"] == run["steps"][0]
                and run["actual_steps"] == (1024 if key == "time" else 512)):
            raise ValueError(f"invalid frozen archive {path}")
        records[key] = record
    return records


def _extend_from_archive(old, wide):
    """Use only exterior nodes from the larger domain, including both new bounds."""
    return np.r_[wide[wide < old[0]], old, wide[wide > old[-1]]]


def _bisect(nodes, selected):
    mid = (nodes[1:] + nodes[:-1]) / 2
    return np.sort(np.r_[nodes, mid[selected(mid)]])


def designs(records):
    def axes(key):
        return tuple(np.asarray(records[key]["grid"][k]) for k in ("x", "sigma"))

    x, s = axes("d16")
    xx, ss = axes("d32")
    x_wide = _extend_from_archive(x, xx)
    s_wide = _extend_from_archive(s, ss)
    # Mesh-only controls use the same d16 bounds and initial node as the factorial baseline.
    return {
        "d16": (x, s, 512), "x_only": (x_wide, s, 512), "sigma_only": (x, s_wide, 512),
        "both": (x_wide, s_wide, 512), "archive_d32": (xx, ss, 512),
        "tail_grade": (*axes("tail_grade"), 512),
        "near_zero": (x, _bisect(s, lambda m: m < .04), 512),
        "interior_sigma": (x, _bisect(s, lambda m: (.04 <= m) & (m <= 1.)), 512),
        "interior_x": (_bisect(x, lambda m: abs(m) <= 1.), s, 512),
        "time": (x, s, 1024),
    }


def _region_sums(mass, rates):
    """Disjoint faces; the two volatility faces own all four corner nodes."""
    def dot(m, v):
        return np.einsum("ij,kij->k", m, v)
    return np.stack((
        dot(mass[:, :1], rates[:, :, :1]), dot(mass[:, -1:], rates[:, :, -1:]),
        dot(mass[::mass.shape[0]-1, 1:-1], rates[:, ::mass.shape[0]-1, 1:-1]),
        dot(mass[1:-1, 1:-1], rates[:, 1:-1, 1:-1]),
    ))


class MomentBudget:
    """Integrate pointwise defects against BE arrival masses, in physical sigma units."""

    def __init__(self, ref, params, count):
        self.grid = grid = ref.grid
        self.count = count
        self.checkpoints = []
        self.steps = 0
        self.clock = 0.
        s = grid.y.nodes
        b = (params.kappa1 + params.kappa2*s) * (params.theta - s)
        a = (params.beta**2 + params.volvol**2) * s**2
        self.payoffs = np.stack((np.ones_like(s), s, s**2))
        self.physical = np.stack((np.zeros_like(s), b, 2*s*b+a))
        rates = np.stack([(ref.backward @ np.broadcast_to(f, grid.shape).ravel())
                          .reshape(grid.shape) for f in self.payoffs])
        self.defect = rates - self.physical[:, None, :]
        self.absolute_defect = np.abs(self.defect)
        self.signed = np.zeros((len(REGIONS), len(FUNCTIONS)))
        self.absolute = self.signed.copy()
        self.physical_integral = np.zeros(3)
        self.initial = self.payoffs @ np.sum(
            grid.project_delta(0., params.sigma0) * grid.quadrature_weights, axis=0)
        self.peak_reconciliation = np.zeros(3)
        self.peak_occupancy = np.zeros(4)

    def observe(self, values, dt):
        grid = self.grid
        self.steps += 1
        self.clock += dt
        mass = grid.quadrature_weights * values
        marginal = mass.sum(axis=0)
        occupancy = np.array([marginal[0], marginal[-1],
                              mass[[0, -1], 1:-1].sum(), mass[1:-1, 1:-1].sum()])
        self.peak_occupancy = np.maximum(self.peak_occupancy, occupancy)
        signed = _region_sums(mass, self.defect)
        absolute = _region_sums(np.abs(mass), self.absolute_defect)
        self.signed += dt * signed
        self.absolute += dt * absolute
        self.physical_integral += dt * (self.physical @ marginal)
        imbalance = self.payoffs @ marginal - self.initial - (
            self.physical_integral + self.signed.sum(axis=0))
        self.peak_reconciliation = np.maximum(self.peak_reconciliation, np.abs(imbalance))
        if self.clock in TIMES:
            s = grid.y.nodes
            obs = _observations(s, grid.y.cell_edges, marginal, logarithmic=False)
            tails = np.array(obs["upper_tail_second_moment"])
            obs.update(time=self.clock, sigma_cell_mass=marginal.tolist(),
                       band_second_moment=-np.diff(np.r_[obs["second_moment"], tails, 0.]),
                       last_cell_second_moment=marginal[-1]*s[-1]**2,
                       joint_faces=dict(x_lower=mass[0].sum(), x_upper=mass[-1].sum(),
                                        corners=mass[np.ix_([0, -1], [0, -1])]),
                       occupancy=occupancy, peak_occupancy=self.peak_occupancy.copy(),
                       signed_rate=signed, absolute_rate=absolute,
                       integrated_signed=self.signed.copy(),
                       integrated_absolute=self.absolute.copy(),
                       integrated_physical=self.physical_integral.copy(), imbalance=imbalance)
            self.checkpoints.append(obs)


class _ObservedOperator(Monotone2DReference):
    def step(self, values, direction, step_size, method="be"):
        result = super().step(values, direction, step_size, method)
        self.observer.observe(result.values, step_size)
        return result


def peak_memory_bytes():
    """OS high-water resident set for this freshly spawned worker, including sparse LU."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
                        *[(k, ctypes.c_size_t) for k in (
                            "peak", "rss", "paged_peak", "paged", "nonpaged_peak", "nonpaged",
                            "pagefile", "pagefile_peak")]]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        get_memory = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        get_memory.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        if not get_memory(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.peak
    import resource
    amount = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(amount if sys.platform == "darwin" else amount * 1024)


def solve(case, name, spec, *, spatial=None, inspect_operator=None):
    """Price with the historical backend, observing its actual state after every implicit solve."""
    start = perf_counter()
    x, sigma, count = spec
    grid = TensorGrid2D(Grid1D(x), Grid1D(sigma))
    config = UnsplitConfig(max_nodes=550_000)
    observer = None
    inspection = {}

    def factory(split, settings):
        nonlocal observer
        if observer is not None:
            raise ValueError("attribution assumes the frozen single coefficient interval")
        operator = _ObservedOperator.from_split(split, settings, spatial=spatial)
        if inspect_operator is not None:
            inspection.update(inspect_operator(operator, split))
        observer = MomentBudget(operator, case.model.dynamics.params, count)
        operator.observer = observer
        return operator

    raw = _price_time_method(case, Variant(name, grid, (count,), False), "be", config,
                             operator_factory=factory)
    if observer.steps != count or [r["time"] for r in observer.checkpoints] != list(TIMES):
        raise ValueError("incorrect backward-Euler observation times")
    raw["status"] = "ok"
    diagnostics = raw["diagnostics"]
    if inspect_operator is not None:
        inspection["peak_forward_error"] = max(abs(row["normalized_forward"]-1.)
                                              for row in diagnostics["step_moments"])
    del diagnostics["step_moments"]
    checks = density_checks(raw, case, .01)
    checks["transient_positivity"] = diagnostics["peak_negative_mass"] <= 1e-12
    return _json_value(dict(name=name, steps=count, grid=dict(x=x, sigma=sigma),
                            grid_sha256=_hash(dict(x=x, sigma=sigma)), shape=grid.shape,
                            initial=observer.initial, checkpoints=observer.checkpoints,
                            max_budget_imbalance=observer.peak_reconciliation,
                            peak_occupancy=observer.peak_occupancy, prices=raw["prices"],
                            diagnostics=diagnostics, density_checks=checks,
                            max_linear_residual=raw["max_linear_residual"],
                            elapsed_seconds=perf_counter()-start,
                            peak_rss_bytes=peak_memory_bytes(),
                            input_sha256=_hash(input_snapshot(case)), **inspection))


def _worker(job):
    from validation.pde_startup import restore_case

    frozen, name, spec, out, signature = job
    case = restore_case(frozen)
    result = solve(case, name, spec)
    result["signature"] = signature
    tmp = out / f"{name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    tmp.replace(out / f"{name}.json")
    return name, result["elapsed_seconds"], result["peak_rss_bytes"]


def _available_bytes():
    if sys.platform == "win32":
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32),
                        *[(k, ctypes.c_uint64) for k in (
                            "total", "available", "pt", "pa", "vt", "va", "ext")]]
        counters = Memory()
        counters.length = ctypes.sizeof(counters)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(counters)):
            raise OSError("failed to query available physical memory")
        return counters.available
    if sys.platform.startswith("linux"):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])*1024
    return None


def attribution(case, output, reference_path, archive_dir, metadata):
    """Resume only matching completed solves. A fresh worker measures peak RSS per grid."""
    from validation.pde_qualification import ROOT

    ref = json.loads(reference_path.read_bytes())
    records = load_archive(archive_dir)
    snapshot = input_snapshot(case)
    if not (ref["passed"] and ref["input"] == snapshot == records["d16"]["input"]
            and ref["input_sha256"] == _hash(snapshot) == FROZEN_HASH
            and ref["times"] == list(TIMES) and ref["cdf_cuts"] == CDF_CUTS.tolist()
            and ref["upper_tail_cuts"] == TAIL_CUTS.tolist()):
        raise ValueError("incompatible Stage 1 reference, snapshot or observation bins")
    frozen = dict(name=case.name, input=snapshot, model_sha256=_hash(snapshot["model"]),
                  policy={"negative_mass_limit": case.negative_mass_limit})
    src = [Path(__file__), ROOT/"validation/pde_monotone.py", ROOT/"validation/pde_unsplit.py",
           ROOT/"validation/pde_qualification.py", ROOT/"validation/pde_volatility_moments.py"]
    signature = _hash(dict(input=FROZEN_HASH,
                           sources={p.name: sha256(p.read_bytes()).hexdigest() for p in src},
                           package=metadata["source_sha256"], versions=metadata["versions"]))
    output.mkdir(exist_ok=True, parents=True)
    run_dir = output/"runs"
    run_dir.mkdir(exist_ok=True)
    runs = {}
    for name, spec in designs(records).items():
        file = run_dir/f"{name}.json"
        if not file.exists():
            free = _available_bytes()
            if free is not None and free < 2.5*2**30:
                raise MemoryError(f"{name}: less than 2.5 GiB available before assembly")
            print(f"Starting {name} {len(spec[0])} x {len(spec[1])}", flush=True)
            with get_context("spawn").Pool(1) as pool:
                _, elapsed, peak = pool.apply(_worker, ((frozen, name, spec, run_dir, signature),))
            print(f"Finished {name} in {elapsed:.1f} s, peak {peak/2**30:.2f} GiB", flush=True)
        run = json.loads(file.read_bytes())
        if not (run["signature"] == signature and run["input_sha256"] == FROZEN_HASH
                and run["grid_sha256"] == _hash(dict(x=spec[0], sigma=spec[1]))
                and run["steps"] == spec[2] and run["name"] == name):
            raise ValueError(f"existing record disagrees with current inputs or source: {file}")
        runs[name] = run
    from validation.pde_attribution_report import write_report
    return write_report(case, records, ref, runs, metadata, output)
