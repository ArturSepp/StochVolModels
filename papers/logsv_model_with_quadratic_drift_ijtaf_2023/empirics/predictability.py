"""Compatibility entry point; implementation moved to ``papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.empirics.predictability``."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.empirics.predictability", run_name="__main__")
else:
    from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.empirics.predictability import *  # noqa: F401,F403
