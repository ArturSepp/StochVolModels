"""Independent real-space density check for both panels of the IL paper's Figure 8."""

import numpy as np

from stochvolmodels import LogSVPricer, LogSvParams
from stochvolmodels.models.logsv import LogSvModel
from stochvolmodels.models.logsv.coefficients import LogSvPdeCoefficients
from stochvolmodels.pde_solvers import (
    ADIConfig,
    BoundaryPair1D,
    BoundarySet2D,
    Grid1D,
    PDE2DADISolver,
    PDEDirection,
    TensorGrid2D,
)
from il_payoffs import il_payoff
from replicate_figures import P0, TTM, figure8_inputs, logsv_premium_curve


def _mesh(params, nx, ny, *, widen=False):
    """Widen the tails with extra nodes, preserving the central mesh for the domain check."""
    x = Grid1D.concentrated(-1.4, 1.4, nx, 0.0, 0.12)
    y = Grid1D.concentrated(0.005, 2.0, ny, params.sigma0, 0.1)
    if widen:
        extra = np.arange(1, 13)
        x = Grid1D(
            np.r_[
                x.nodes[0] - extra[::-1] * x.spacings[0],
                x.nodes,
                x.nodes[-1] + extra * x.spacings[-1],
            ]
        )
        y = Grid1D(np.r_[0.0, y.nodes, y.nodes[-1] + extra * y.spacings[-1]])
    return TensorGrid2D(x, y)


def _project(params, chain, ranges, grid, steps):
    """Project both panels from one undiscounted terminal law with zero carry."""
    natural = BoundaryPair1D.natural()
    law = PDE2DADISolver(ADIConfig(theta=0.5)).solve(
        grid,
        LogSvPdeCoefficients(LogSvModel(params)),
        BoundarySet2D(natural, natural),
        PDEDirection.FORWARD,
        np.linspace(0.0, TTM, steps + 1),
        grid.project_delta(0.0, params.sigma0),
    )
    mass = grid.marginalize_y(law.values) * grid.x.quadrature_weights
    relative = np.exp(grid.x.nodes)
    spots = P0 * relative
    # The IL payoff is the borrowed LP's P&L in units of the initial notional.
    apr = [
        -np.dot(
            mass, il_payoff(spots, P0, P0 * np.exp(-r), P0 * np.exp(r), borrowed=True, notional=1.0)
        )
        / TTM
        for r in ranges
    ]
    strikes, kinds = chain.strikes_ttms[0], chain.optiontypes_ttms[0]
    f, d = chain.forwards[0], chain.discfactors[0]
    intrinsic = (f * relative[:, None] - strikes[None, :]) * np.where(kinds == "C", 1.0, -1.0)
    prices = d * (mass @ np.maximum(intrinsic, 0.0))
    ivols = chain.compute_model_ivols_from_chain_data([prices])[0]
    if not np.all(np.isfinite(ivols)):
        raise FloatingPointError(
            f"Figure 8 IV inversion failed at K/F={strikes[~np.isfinite(ivols)] / f}"
        )
    total_mass, forward = float(mass.sum()), float(mass @ relative)
    assert abs(total_mass - 1.0) < 1e-10 and abs(forward - 1.0) < 5e-4
    return dict(
        shape=grid.shape,
        log_forward_bounds=grid.x.nodes[[0, -1]].tolist(),
        volatility_bounds=grid.y.nodes[[0, -1]].tolist(),
        time_steps=steps,
        theta=0.5,
        protection_apr=np.asarray(apr).tolist(),
        vanilla_prices=prices.tolist(),
        implied_vols=ivols.tolist(),
        mass=total_mass,
        normalized_forward=forward,
        terminal_negative_mass=law.diagnostics[-1].negative_mass,
        peak_negative_mass=max(step.negative_mass for step in law.diagnostics),
    )


def compare_figure8(quadrature):
    """Return the original series and compare all ranges and strikes on four numerical grids."""
    chain, ranges, parameters = figure8_inputs()
    strikes = chain.strikes_ttms[0]
    center = (strikes >= 0.8 - 1e-14) & (strikes <= 1.2 + 1e-14)
    paper_pricer = LogSVPricer()
    cases = []
    for params in parameters:
        paper_apr = logsv_premium_curve(params, ranges)
        raw, vols = paper_pricer.compute_chain_prices_with_vols(chain, params)
        reference_prices, reference_iv = raw[0], vols[0]
        fine = _mesh(params, 101, 61)
        configurations = {
            "fine": (fine, 80),
            "coarse": (_mesh(params, 51, 31), 40),
            "time_refined": (fine, 160),
            "wide_domain": (_mesh(params, 101, 61, widen=True), 80),
        }
        results = {}
        for name, (grid, steps) in configurations.items():
            row = _project(params, chain, ranges, grid, steps)
            apr_error = 100.0 * (row["protection_apr"] - paper_apr)
            iv_error = 100.0 * (row["implied_vols"] - reference_iv)
            price_error = 1e4 * (row["vanilla_prices"] - reference_prices)
            if name != "coarse":
                assert np.all(np.abs(apr_error) < 0.08), (name, params.volvol, apr_error)
                assert np.all(np.abs(iv_error) < 1.1), (name, params.volvol, iv_error)
                assert np.max(np.abs(iv_error[center])) < 0.15, (name, params.volvol, iv_error)
            row.update(
                apr_difference_pp=apr_error.tolist(),
                iv_difference_pp=iv_error.tolist(),
                price_difference_bp=price_error.tolist(),
            )
            results[name] = row
        cases.append(
            dict(
                params={
                    k: getattr(params, k)
                    for k in ("sigma0", "theta", "kappa1", "kappa2", "beta", "volvol")
                },
                transform_apr=paper_apr.tolist(),
                transform_ivs=reference_iv.tolist(),
                transform_prices=reference_prices.tolist(),
                grids=results,
            )
        )

    # In the fixed-volatility limit a separate AMM-inventory quadrature prices the same ranges.
    sigma = 0.5
    constant = LogSvParams(
        sigma0=sigma, theta=sigma, beta=0.0, volvol=0.0, kappa1=2.21, kappa2=2.18
    )
    grid = TensorGrid2D(Grid1D.concentrated(-1.4, 1.4, 201, 0.0, 0.12), Grid1D.uniform(0.0, 1.0, 5))
    projected = _project(constant, chain, ranges, grid, 160)
    black = np.array(
        [
            -quadrature(P0, P0, P0 * np.exp(-r), P0 * np.exp(r), TTM, sigma, True) / TTM
            for r in ranges
        ]
    )
    gap = 100.0 * (projected["protection_apr"] - black)
    assert np.max(np.abs(gap)) < 0.05, gap
    return dict(
        spot=P0,
        maturity=TTM,
        discount=1.0,
        notional=1.0,
        borrowed=True,
        strikes_over_forward=(strikes / chain.forwards[0]).tolist(),
        option_types=chain.optiontypes_ttms[0].tolist(),
        log_range=ranges.tolist(),
        cases=cases,
        constant_vol_check=dict(
            volatility=sigma,
            quadrature_apr=black.tolist(),
            density=projected,
            apr_difference_pp=gap.tolist(),
        ),
    )
