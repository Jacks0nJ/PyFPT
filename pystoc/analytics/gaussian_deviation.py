import numpy as np
from scipy import optimize

from .He3 import He3
from .He4 import He4
from .He6 import He6


# Using the Gram–Charlier A series
# https://en.wikipedia.org/wiki/Edgeworth_series to approximate when we expect
# classical deviation from a gaussian. This is done by finding x such that the
# higher order terms of the edgeworth expanion are
# nu is the amount pf deviation from a Gaussian.
def gaussian_deviation(mean, std, third_cumulant, fourth_cumulant, nu=1,
                       x_interval=None):

    def higher_order_egdeworth_term(y):
        norm_y = (y-mean)/std
        skew_term = np.divide(third_cumulant*He3(norm_y), 6*std**3)
        kurtosis_term = np.divide(fourth_cumulant*He4(norm_y), 24*std**4)
        skew_squared_term =\
            np.divide(He6(norm_y)*third_cumulant**2, 72*std**6)
        return (skew_term+kurtosis_term+skew_squared_term)-nu

    if x_interval is None:
        sol = optimize.root_scalar(higher_order_egdeworth_term,
                                   method='brentq',
                                   bracket=[mean, mean+10000*std])
    else:
        sol = optimize.root_scalar(higher_order_egdeworth_term,
                                   method='brentq', bracket=x_interval)

    return sol.root  # The root is the position of when deviation occurs
