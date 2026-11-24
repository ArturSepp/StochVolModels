# stochvolmodels.pde_solvers

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](../../../CITATION.cff).

Discretise forward and backward equations on one-dimensional and tensor grids. Models supply coefficients, products and pricers supply terminal or initial data and interpret the result.

[Parent package guide](../README.md).

Read each row as importer → imported SVM module. Names are relative to stochvolmodels. Lazy marks on-demand imports and export lookups, typing marks type-checking edges. The last column gives directly used optional packages, test tools or checkout namespaces. A dash means none.

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Discretise forward and backward equations on one-dimensional and tensor grids. | [`pde_solvers.adi`](adi.py), [`pde_solvers.boundaries`](boundaries.py), [`pde_solvers.coefficients`](coefficients.py), [`pde_solvers.diagnostics`](diagnostics.py), [`pde_solvers.grids`](grids.py), [`pde_solvers.jumps`](jumps.py), [`pde_solvers.operators_1d`](operators_1d.py), [`pde_solvers.solver_1d`](solver_1d.py), [`pde_solvers.solver_2d`](solver_2d.py) | [pde_solvers.py](../../../examples/docs/pde_solvers.py) | — |
| [`adi.py`](adi.py) | Alternating-direction time stepping for split two-dimensional PDEs. | [`pde_solvers.operators_1d`](operators_1d.py), [`pde_solvers.operators_2d`](operators_2d.py), [`pde_solvers.solver_1d`](solver_1d.py) | [test_pde_adi.py](../tests/test_pde_adi.py) | — |
| [`boundaries.py`](boundaries.py) | Typed immutable boundary specifications for PDE operators. | [`pde_solvers.grids`](grids.py) | [test_pde_adi.py](../tests/test_pde_adi.py) | — |
| [`coefficients.py`](coefficients.py) | Validated generic PDE coefficient containers and provider protocols. | Typing [`pde_solvers.grids`](grids.py) | [test_pde_adi.py](../tests/test_pde_adi.py) | — |
| [`diagnostics.py`](diagnostics.py) | Per-step observability for one-dimensional PDE propagation. | [`pde_solvers.boundaries`](boundaries.py), [`pde_solvers.grids`](grids.py), [`pde_solvers.operators_1d`](operators_1d.py) | [test_pde_solver_1d.py](../tests/test_pde_solver_1d.py) | — |
| [`grids.py`](grids.py) | Immutable one- and two-dimensional grids for finite-difference PDE solvers. | — | [test_pde_grids.py](../tests/test_pde_grids.py) | — |
| [`jumps.py`](jumps.py) | Weighted-adjoint deterministic translations and compound-Poisson jump steps. | [`pde_solvers.grids`](grids.py), [`pde_solvers.operators_1d`](operators_1d.py) | [test_pde_jumps.py](../tests/test_pde_jumps.py) | — |
| [`operators_1d.py`](operators_1d.py) | Generic one-dimensional backward generators and weighted forward adjoints. | [`pde_solvers.boundaries`](boundaries.py), [`pde_solvers.coefficients`](coefficients.py), [`pde_solvers.grids`](grids.py), [`pde_solvers.tridiagonal`](tridiagonal.py) | [test_pde_operators_1d.py](../tests/test_pde_operators_1d.py) | — |
| [`operators_2d.py`](operators_2d.py) | Split two-dimensional PDE generators on nonuniform tensor grids. | [`pde_solvers.boundaries`](boundaries.py), [`pde_solvers.coefficients`](coefficients.py), [`pde_solvers.grids`](grids.py), [`pde_solvers.operators_1d`](operators_1d.py), [`pde_solvers.solver_1d`](solver_1d.py), [`pde_solvers.tridiagonal`](tridiagonal.py) | [test_pde_operators_2d.py](../tests/test_pde_operators_2d.py) | — |
| [`poisson.py`](poisson.py) | Conservative forward translations along the first axis of a tensor density grid. | [`pde_solvers.grids`](grids.py) | [test_pde_adi.py](../tests/test_pde_adi.py) | — |
| [`solver_1d.py`](solver_1d.py) | Theta time stepping for generic one-dimensional forward and backward PDEs. | [`pde_solvers.boundaries`](boundaries.py), [`pde_solvers.coefficients`](coefficients.py), [`pde_solvers.diagnostics`](diagnostics.py), [`pde_solvers.grids`](grids.py), [`pde_solvers.operators_1d`](operators_1d.py), [`pde_solvers.tridiagonal`](tridiagonal.py) | [test_pde_solver_1d.py](../tests/test_pde_solver_1d.py) | — |
| [`solver_2d.py`](solver_2d.py) | ADI propagation for generic two-dimensional forward and backward PDEs. | [`pde_solvers.adi`](adi.py), [`pde_solvers.boundaries`](boundaries.py), [`pde_solvers.coefficients`](coefficients.py), [`pde_solvers.diagnostics`](diagnostics.py), [`pde_solvers.grids`](grids.py), [`pde_solvers.operators_1d`](operators_1d.py), [`pde_solvers.operators_2d`](operators_2d.py), [`pde_solvers.poisson`](poisson.py), [`pde_solvers.solver_1d`](solver_1d.py) | [test_pde_adi.py](../tests/test_pde_adi.py) | — |
| [`tridiagonal.py`](tridiagonal.py) | Inspectable tridiagonal algebra used by implicit PDE line solves. | — | [test_pde_grids.py](../tests/test_pde_grids.py) | — |
