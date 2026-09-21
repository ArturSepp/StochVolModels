"""The wheel test inventory follows source files rather than a fixed module count."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from zipfile import ZipFile

import pytest


@pytest.fixture
def checker(tmp_path, monkeypatch):
    """Load the repository-only checker against a controlled source inventory."""
    script = Path(__file__).resolve().parents[3] / "scripts/check_wheel_contents.py"
    if not script.is_file():
        pytest.skip("Repository wheel builder is not part of an installed wheel")
    spec = spec_from_file_location("wheel_checker", script)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source_tests"
    source.mkdir()
    for index in range(34):
        (source / f"test_feature_{index}.py").write_text("# source inventory\n")
    monkeypatch.setattr(module, "SOURCE_TEST_ROOT", source, raising=False)
    return module


def make_wheel(tmp_path, checker, *, omit=False, extra=False):
    """Create a minimal archive containing the independently enumerated files."""
    names = set(checker.REQUIRED_RUNTIME_FILES)
    names.update(f"stochvolmodels/tests/test_feature_{index}.py" for index in range(34))
    names.add(
        "stochvolmodels/tests/test_rough_logsv_pricer_regression/"
        "test_rough_logsv_pricer_pricing_regression.npz"
    )
    if omit:
        names.remove("stochvolmodels/tests/test_feature_0.py")
    if extra:
        names.add("stochvolmodels/tests/test_untracked_feature.py")
    wheel = tmp_path / "stochvolmodels-0.0.0-py3-none-any.whl"
    with ZipFile(wheel, "w") as archive:
        for name in names:
            archive.writestr(name, b"")
    return wheel


def test_new_source_test_does_not_require_updating_a_magic_count(tmp_path, checker):
    checker.check_wheel(make_wheel(tmp_path, checker))


def test_missing_source_test_is_rejected_even_at_the_old_count(tmp_path, checker):
    with pytest.raises(AssertionError, match="test module inventory"):
        checker.check_wheel(make_wheel(tmp_path, checker, omit=True))


def test_unexpected_test_module_is_rejected(tmp_path, checker):
    with pytest.raises(AssertionError, match="test module inventory"):
        checker.check_wheel(make_wheel(tmp_path, checker, extra=True))
