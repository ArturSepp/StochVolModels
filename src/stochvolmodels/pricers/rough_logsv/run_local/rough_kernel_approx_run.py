"""Compatibility entry point for the rough model's local development runner.

Imports resolve to the canonical runner, preserving its globals and source path.
Executing the former module path dispatches the canonical runner exactly once.
"""
if __name__ == "__main__":
    import runpy

    runpy.run_module(
        "stochvolmodels.models.rough_logsv.run_local.rough_kernel_approx_run", run_name="__main__"
    )
else:
    import sys
    from stochvolmodels.models.rough_logsv.run_local import rough_kernel_approx_run as _implementation

    sys.modules[__name__] = _implementation
