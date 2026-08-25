# Volatility-book production

This directory contains repository-only chapter analytics. The B1A production contract provides
one bounded, deterministic smoke rollup for the three accepted chapter pipelines. It does not
change package APIs or chapter numerics.

## Run the rollup

Install the repository environment from the tracked lock first. The research extra resolves
option-chain-analytics 5.2.0 exactly from uv.lock:

~~~console
uv sync --locked --extra research
~~~

From the repository root, the one production command is:

~~~console
python -m volatility_book.run_book_production --profile smoke
~~~

On a fresh output tree, the command validates the pinned inputs and computes all three stages in
contract order. On a subsequent invocation, the same command validates and reuses every complete
stage; it does not recompute merely because outputs already exist. A missing, incomplete, stale,
or invalid stage is regenerated selectively while other valid stages remain reused.

To deliberately replace every contract-owned stage after the same path checks:

~~~console
python -m volatility_book.run_book_production --profile smoke --force
~~~

The force flag requests all-stage recomputation. Both selective regeneration and force are limited
to the output directories declared in book_production_contract.json. If
STOCHVOLMODELS_BOOK_FORBID_RECOMPUTE=1, any fresh, stale, or forced run that would invoke
analytics fails before mutation; a complete validated run may still be reused.

The independent contract check is:

~~~console
python -m volatility_book.verify_book_production --profile smoke
~~~

## Deterministic graph

The stages are independent numerically and execute in this fixed order:

| Stage | Chapter profile | Contract-owned output |
| --- | --- | --- |
| discrete_d3_smoke | smoke | outputs/volatility_book/ch_discrete_vol/book_production/smoke |
| regime_t3_smoke | smoke | outputs/volatility_book/ch_lognormal_sv_risk_premia/book_production/smoke |
| student_t3_canonical | canonical | outputs/volatility_book/ch_tdist_risk_premia/book_production/smoke |

The Student chapter intentionally uses canonical: that deterministic terminal-law chapter has no
reduced profile. The discrete stage intentionally stops at the acceptance manifest's D3.1 smoke
gate. Discrete Round 2 requires full and clean-reference Round-1 provenance, and Round 3 consumes
the full upstream chain; those exact-tag/full-budget obligations are not a fresh-clone
reduced-workload smoke path.

## Inputs and outputs

The tracked contract hash-pins the three chapter acceptance manifests and uv.lock using SHA-256
over UTF-8 text normalized to LF. This makes the pins identical on LF and CRLF checkouts. The
contract also records option-chain-analytics 5.2.0 as the required locked version. Chapter
computations use tracked sources, embedded deterministic parameters and seeds, and no market,
network, or licensed analytics inputs.

Generated artifacts remain untracked below the ignored outputs/ tree:

- The discrete stage writes its JSON results, Markdown report, and figures PDF.

- The regime stage writes one validated numerical payload, four PDF/PNG figure pairs, two TeX
  tables, and its chapter artifact manifest.

- The Student stage writes one validated numerical payload, two PDF/PNG figure pairs, one TeX
  table, and its chapter artifact manifest.

- The rollup writes
  outputs/volatility_book/book_production/smoke/execution_manifest.json, an ignored snapshot of
  the tracked contract, and a persistent `.run.lock` control file whose operating-system lock is
  released automatically if a process exits. Numba and Matplotlib caches are redirected into the
  declared `.runtime_cache` directory beside those files, so analytics never write caches into
  tracked source trees. Manifest paths are repository-relative and artifact hashes use SHA-256;
  absolute paths are not part of the portable contract.

## Safety boundaries

Repository-local artifacts are allowed only below outputs/volatility_book, with each child stage
further confined to its chapter's accepted output root. The runner rejects absolute path values
in the contract, path traversal, symlinks and Windows reparse points, the repository root or an
ancestor of it, and tracked chapter locations. It never recomputes a cache-valid stage unless
force is explicit, and it never replaces anything outside the contract-owned directories. The
analytics themselves are offline; environment creation may contact the configured Python package
index to obtain locked dependencies.

This unit excludes the discrete Round 2/3 chain, the regime canonical workload, master-book TeX or
PDF assembly, external calibration or data acquisition, package and chapter numerical changes,
version changes, tags, releases, publication, and deployment.
