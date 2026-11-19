"""Compatibility imports for ``stochvolmodels.models.rough_logsv.split_simulation``.

The canonical implementation owns the definitions and numerical behavior.
Existing import paths retain the same objects without deprecation warnings.
"""

from stochvolmodels.models.rough_logsv.split_simulation import (
    np as np,
    Enum as Enum,
    Tuple as Tuple,
    expm as expm,
    njit as njit,
    objmode as objmode,
    batch_expA as batch_expA,
    batch_invA as batch_invA,
    drift_ode_solve as drift_ode_solve,
    drift_ode_solve2 as drift_ode_solve2,
    drift_ode_solve2_f32 as drift_ode_solve2_f32,
    drift_ode_solve3 as drift_ode_solve3,
    diffus_sde_solve_f64 as diffus_sde_solve_f64,
    drift_diffus_strand_f64 as drift_diffus_strand_f64,
    log_spot_full_solve2_f64 as log_spot_full_solve2_f64,
    log_spot_full_combined_f64 as log_spot_full_combined_f64,
    diffus_sde_solve_f32 as diffus_sde_solve_f32,
    drift_diffus_strand_f32 as drift_diffus_strand_f32,
    log_spot_full_solve2_f32 as log_spot_full_solve2_f32,
    log_spot_full_combined_f32 as log_spot_full_combined_f32,
    log_spot_full_combined as log_spot_full_combined,
    NamedTuple as NamedTuple,
    _valid_factor_paths as _valid_factor_paths,
    _drift_rk4_checked as _drift_rk4_checked,
    drift_ode_solve_exp_implicit as drift_ode_solve_exp_implicit,
    _split_volatility_move as _split_volatility_move,
    split_step_f64 as split_step_f64,
    split_step_combined_f64 as split_step_combined_f64,
    SplitStepPaths as SplitStepPaths,
    simulate_split_step as simulate_split_step,
)
