"""Configuration that must accompany the tests when they ship in a wheel."""


def pytest_configure(config) -> None:
    """Register package-local markers outside the source checkout."""
    markers = (
        "slow: numerical regression or simulation tests excluded from the fast CI lane",
        "optional_integration: provider or optional-extra contract that skips on a core install",
        "repository_only: documentation, example, or paper contract absent from the wheel",
        "paper_replication: long repository paper-reproduction gate run outside the core suite",
    )
    for marker in markers:
        config.addinivalue_line("markers", marker)
