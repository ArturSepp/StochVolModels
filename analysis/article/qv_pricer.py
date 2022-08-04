# built
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

# internal
import utils.mgf_pricer as mgfp
from generic.config import VariableType
import pricers.logsv.affine_expansion as afe
from pricers.logsv.affine_expansion import ExpansionOrder
from pricers.logsv.logsv_params import LogSvParams
from utils.bsm_pricer import model_chain_prices_to_bsm_ivols


class UnitTests(Enum):
    QV_SLICE_PRICER = 1
    COMPARE_WITH_MC = 2


def run_unit_test(unit_test: UnitTests):

    params = LogSvParams(sigma0=1.0,
                         theta=1.0,
                         kappa1=3.0,
                         kappa2=3.0,
                         beta=0.15,
                         volvol=1.85)

    strikes = np.linspace(0.5, 2.0, 19)
    optiontypes = np.full(strikes.shape, 'C')
    expansion_order = ExpansionOrder.SECOND
    variable_type = VariableType.Q_VAR
    phi_grid, psi_grid, theta_grid = mgfp.get_transform_var_grid(variable_type=variable_type, is_spot_measure=True)

    if unit_test == UnitTests.QV_SLICE_PRICER:
        ttm = 1.0
        forward = 1.0
        a_grid = afe.compute_logsv_a_mgf_grid(phi_grid=phi_grid,
                                              psi_grid=psi_grid,
                                              theta_grid=theta_grid,
                                              ttm=ttm,
                                              params=params,
                                              is_analytic=False,
                                              expansion_order=expansion_order,
                                              is_stiff_solver=False)
        qvar_options = mgfp.slice_qvar_pricer_with_a_grid(log_mgf_grid=a_grid,
                                                          psi_grid=psi_grid,
                                                          ttm=ttm,
                                                          forward=forward,
                                                          strikes=strikes,
                                                          optiontypes=optiontypes,
                                                          y=params.sigma0 - params.theta,
                                                          expansion_order=expansion_order)

        bsm_ivols = model_chain_prices_to_bsm_ivols(ttms=np.array([ttm]),
                                                    forwards=np.array([forward]),
                                                    strikes_ttms=(strikes,),
                                                    optiontypes_ttms=(optiontypes,),
                                                    model_prices_ttms=(qvar_options,))

        print(qvar_options)
        print(bsm_ivols)

    elif unit_test == UnitTests.COMPARE_WITH_MC:
        ttms = np.array([0.12, 0.25, 0.5, 1.0])
        forwards = np.array([1.0, 1.0, 1.0, 1.0])
        discfactors = np.ones_like(forwards)
        fig = mgfp.plot_comp_mgf_with_mc(ttms=ttms,
                                         forwards=forwards,
                                         discfactors=discfactors,
                                         strikes_ttms=(strikes, strikes, strikes, strikes),
                                         optiontypes_ttms=(optiontypes, optiontypes, optiontypes, optiontypes),
                                         params=params,
                                         variable_type=VariableType.Q_VAR,
                                         is_stiff_solver=False,
                                         is_log_strike_xaxis=False,
                                         is_add_first_order=True,
                                         is_spot_measure=True)
        fig.savefig("..//..//draft//figures//model_vs_mc_qvar_logsv.pdf")

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.COMPARE_WITH_MC

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
