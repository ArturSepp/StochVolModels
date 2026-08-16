"""Configuration that must accompany the tests when they ship in a wheel."""


def pytest_configure(config) -> None:
    """Register package-local markers outside the source checkout."""
    config.addinivalue_line(
        "markers",
        "slow: numerical regression or simulation tests excluded from the fast CI lane",
    )
