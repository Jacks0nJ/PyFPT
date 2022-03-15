import numpy as np
from scipy import optimize

from .He3 import He3
from .He4 import He4
from .He6 import He6

from .mean_N_sto_limit import mean_N_sto_limit
from .variance_N_sto_limit import variance_N_sto_limit
from .skewness_N_sto_limit import skewness_N_sto_limit
from .kurtosis_N_sto_limit import kurtosis_N_sto_limit


# Using the Gram–Charlier A series
# https://en.wikipedia.org/wiki/Edgeworth_series to approximate when we expect
# classical deviation from a gaussian. This is done by finding x such that the
# higher order terms of the edgeworth expanion are
# nu is the amount pf deviation from a Gaussian.
def gaussian_deviation_sto_limit(V, V_dif, V_ddif, phi_i, phi_end, nu=1,
                                 x_interval=None):

    mean = mean_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)
    std = variance_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)**0.5
    skewness = skewness_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)
    kurtosis = kurtosis_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)

    def higher_order_egdeworth_term(y):
        norm_y = (y-mean)/std
        skew_term = np.divide(skewness*He3(norm_y), 6)
        kurtosis_term = np.divide(kurtosis*He4(norm_y), 24)
        skew_squared_term =\
            np.divide(He6(norm_y)*skewness**2, 72)
        return (skew_term+kurtosis_term+skew_squared_term)-nu

    if x_interval is None:
        sol = optimize.root_scalar(higher_order_egdeworth_term,
                                   method='brentq',
                                   bracket=[mean, mean+10000*std])
    else:
        sol = optimize.root_scalar(higher_order_egdeworth_term,
                                   method='brentq', bracket=x_interval)

    return sol.root  # The root is the position of when deviation occurs
