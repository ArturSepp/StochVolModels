"""Run the correlated heat equation and one-week LogSV grid refinements."""

from enum import Enum

from examples.docs.pde_solvers import heat_2d, short_expiry_refinement


class Locals(Enum):
    """Correlated diffusion and short-expiry LogSV examples."""

    GAUSSIAN_MOMENTS = 1
    SHORT_EXPIRY = 2


def run_local(local: Locals) -> None:
    """Execute a canonical solver example and report its numerical diagnostics."""
    if local == Locals.GAUSSIAN_MOMENTS:
        last = heat_2d().diagnostics[-1]
        print(
            f"2D heat density at t={last.time_end:g}: mass {last.mass:.8f}, "
            f"variances ({last.variance_x:.6f}, {last.variance_y:.6f}), covariance {last.covariance_xy:.6f}"
        )
    elif local == Locals.SHORT_EXPIRY:
        for row in short_expiry_refinement():
            print(
                f"{row['model']} {row['nx']} x {row['ny']}, {row['steps']} steps: "
                f"maximum price error {row['price_error_bp']:.3f} bp of discounted forward, "
                f"negative mass {row['negative_mass']:.2e}"
            )
    else:
        raise ValueError(local)


if __name__ == "__main__":
    for case in Locals:
        run_local(case)
