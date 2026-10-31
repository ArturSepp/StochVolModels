"""Compatibility entry point; implementation moved to ``papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.empirics.plot_data``."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.empirics.plot_data", run_name="__main__")
else:
    from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.empirics.plot_data import *  # noqa: F401,F403
