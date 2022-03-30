'''
Gaussian Deviation
------------------
This module calculates the point of deviation from Gaussian behaviour for the
first-passage times in the number of e-folds for a provided threshold value
using the `Edgeworth series`_ in low diffusion limit
and the relations for the central moments given in `Vennin-Starobinsky 2015`_.
This is calculated by using root finding to find the point at which the higher
order terms of the Edgeworth series first equals the threshold.

.. _Edgeworth series: https://en.wikipedia.org/wiki/Edgeworth_series
.. _Vennin-Starobinsky 2015: https://arxiv.org/abs/1506.04732
'''


import numpy as np
from scipy import optimize

from .mean_N_sto_limit import mean_N_sto_limit
from .variance_N_sto_limit import variance_N_sto_limit
from .skewness_N_sto_limit import skewness_N_sto_limit
from .kurtosis_N_sto_limit import kurtosis_N_sto_limit


# Using the Gram–Charlier A series
# https://en.wikipedia.org/wiki/Edgeworth_series to approximate when we expect
# classical deviation from a gaussian. This is done by finding x such that the
# higher order terms of the edgeworth expanion are
# nu is the amount pf deviation from a Gaussian.
def gaussian_deviation_sto_limit(V, V_dif, V_ddif, phi_i, phi_end, nu=1.,
                                 phi_interval=None):
    """Returns the skewness of the number of e-folds.

    Parameters
    ----------
    V : function
        The potential.
    V_dif : function
        The potential's first derivative.
    V_ddif : function
        The potential second derivative.
    phi_i : float
        The initial scalar field value.
    nu : float, optional
        The decimal threshold of the deviation from Gaussianity. Defaults to 1
    phi_interval : list, optional.
        The field interval which contains the root. Defaults to between 0 and
        10000 standard deviations from the mean.

    Returns
    -------
    deviation_point : float
        The field value at which the deviation occurs.

    """
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

    if phi_interval is None:
        sol = optimize.root_scalar(higher_order_egdeworth_term,
                                   method='brentq',
                                   bracket=[mean, mean+10000*std])
    else:
        sol = optimize.root_scalar(higher_order_egdeworth_term,
                                   method='brentq', bracket=phi_interval)
    # The root is the position of when deviation occurs
    deviation_point = sol.root
    return deviation_point


# This is the "probabilist's Hermite polynomial", which is different to the
# "physicist's Hermite polynomials" used by SciPy
def He3(y):
    He3 = y**3-3*y
    return He3


# This is the "probabilist's Hermite polynomial", which is different to the
# "physicist's Hermite polynomials" used by SciPy
def He4(y):
    He4 = y**4-6*y+3
    return He4


# This is the "probabilist's Hermite polynomial", which is different to the
# "physicist's Hermite polynomials" used by SciPy
def He6(y):
    He6 = y**6-15*y**4+45*y**2-15
    return He6
