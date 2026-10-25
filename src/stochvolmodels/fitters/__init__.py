"""Provider-independent volatility-model fitting and diagnostic utilities.

Exports combine approximate LogSV smile fitting with immutable fit and surface
reports. Diagnostics observe supplied model outputs in explicit quote units;
optional provider conversion is isolated in the adapters subpackage.
"""

from stochvolmodels.fitters.diagnostics import DiagnosticReport, diagnose_fit
from stochvolmodels.fitters.surface_checks import (
    SurfaceEvaluator,
    check_call_slice,
    check_calendar,
    compare_surface_reports,
    diagnose_surface,
)
from stochvolmodels.fitters.logsv_smile import (
    ATM_VOL,
    BETA,
    VOLVOL,
    calc_logsv_atm_fit,
    calc_logsv_ivols,
    calc_logsv_ivols_partials,
    calc_logsv_pdf,
    calc_logsv_pdf_core,
    fit_logsv_ivols,
    generate_grid_option_prices_from_slice,
    get_pdf_delta_space,
    get_vols_delta_space,
    infer_strikes_from_deltas,
)

__all__ = [
    'DiagnosticReport',
    'SurfaceEvaluator',
    'diagnose_fit',
    'check_call_slice',
    'check_calendar',
    'diagnose_surface',
    'compare_surface_reports',
    'ATM_VOL',
    'BETA',
    'VOLVOL',
    'calc_logsv_atm_fit',
    'calc_logsv_ivols',
    'calc_logsv_ivols_partials',
    'calc_logsv_pdf',
    'calc_logsv_pdf_core',
    'fit_logsv_ivols',
    'generate_grid_option_prices_from_slice',
    'get_pdf_delta_space',
    'get_vols_delta_space',
    'infer_strikes_from_deltas',
]
