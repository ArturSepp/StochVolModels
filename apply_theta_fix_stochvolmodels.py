"""
apply the bsm theta fix to StochVolModels: run from the repository root
patches stochvolmodels/pricers/analytic/bsm.py, bumps pyproject.toml 1.2.1 -> 1.2.2,
and inserts the CHANGELOG entry. every replacement is exact-match and asserted,
so the script aborts without writing anything if the working copy differs from HEAD.
"""
# packages
from pathlib import Path

BSM = Path('stochvolmodels/pricers/analytic/bsm.py')
PYPROJECT = Path('pyproject.toml')
CHANGELOG = Path('CHANGELOG.md')

OLD_CALL = "            theta = -forward * npdf(d1)*vol/(0.5*np.sqrt(ttm)) - discount_rate*discfactor*strike*ncdf(d2)"
NEW_CALL = "            theta = -discfactor*forward * npdf(d1)*vol/(2.0*np.sqrt(ttm)) - discount_rate*discfactor*strike*ncdf(d2)"
OLD_PUT = "            theta = -forward * npdf(d1)*vol/(0.5*np.sqrt(ttm)) + discount_rate*discfactor*strike*ncdf(-d2)"
NEW_PUT = "            theta = -discfactor*forward * npdf(d1)*vol/(2.0*np.sqrt(ttm)) + discount_rate*discfactor*strike*ncdf(-d2)"

OLD_VERSION = 'version = "1.2.1"'
NEW_VERSION = 'version = "1.2.2"'

CHANGELOG_ANCHOR = "## [1.2.1] - 2026-07-21"
CHANGELOG_ENTRY = """## [1.2.2] - 2026-07-22

### Fixed
- `compute_bsm_vanilla_theta` (and `compute_bsm_vanilla_theta_vector`): the volatility-decay
  term was 4x too large -- `vol/(0.5*sqrt(ttm))` instead of `vol/(2*sqrt(ttm))` -- and omitted
  the leading `discfactor`, so theta was wrong in every regime for both calls and puts.
  Reported by @gaoflow in ArturSepp/VanillaOptionPricers#1.


"""


def apply_edit(path: Path, replacements: list) -> None:
    src = path.read_text(encoding='utf-8')
    for old, new in replacements:
        n = src.count(old)
        if n != 1:
            raise ValueError(f"{path}: expected pattern exactly once, got {n}: {old[:70]!r}")
    for old, new in replacements:
        src = src.replace(old, new)
    path.write_text(src, encoding='utf-8', newline='\n')
    print(f"patched {path}")


def main() -> None:
    for p in (BSM, PYPROJECT, CHANGELOG):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found: run from the StochVolModels repository root")
    # validate all files before writing any of them
    edits = [
        (BSM, [(OLD_CALL, NEW_CALL), (OLD_PUT, NEW_PUT)]),
        (PYPROJECT, [(OLD_VERSION, NEW_VERSION)]),
        (CHANGELOG, [(CHANGELOG_ANCHOR, CHANGELOG_ENTRY + CHANGELOG_ANCHOR)]),
    ]
    for path, replacements in edits:
        src = path.read_text(encoding='utf-8')
        for old, _ in replacements:
            n = src.count(old)
            if n != 1:
                raise ValueError(f"{path}: expected pattern exactly once, got {n}: {old[:70]!r}")
    for path, replacements in edits:
        apply_edit(path, replacements)
    print("done: bsm.py fixed, version 1.2.1 -> 1.2.2, CHANGELOG entry inserted")


if __name__ == '__main__':
    main()
