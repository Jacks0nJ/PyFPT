'''
Gaussian Deviation
------------------
This module calculates the point of deviation from Gaussian behaviour for the
first-passage times in the number of e-folds for a provided threshold value
using the `Edgeworth series`_ in low diffusion limit
and the relations for the central moments given in `Vennin--Starobinsky 2015`_.
This is calculated by using root finding to find the point at which the higher
order terms of the Edgeworth series first equal the threshold.

.. _Edgeworth series: https://en.wikipedia.org/wiki/Edgeworth_series
.. _Vennin--Starobinsky 2015: https://arxiv.org/abs/1506.04732
'''


import numpy as np
from scipy import optimize

from .mean_efolds import mean_efolds
from .variance_efolds import variance_efolds
from .skewness_efolds import skewness_efolds
from .kurtosis_efolds import kurtosis_efolds


# Using the Gram–Charlier A series
# https://en.wikipedia.org/wiki/Edgeworth_series to approximate when we expect
# classical deviation from a gaussian. This is done by finding x such that the
# higher order terms of the edgeworth expanion are
# nu is the amount pf deviation from a Gaussian.
def gaussian_deviation(potential, potential_dif, potential_ddif, phi_in,
                       phi_end, nu=1., phi_interval=None):
    """Returns the skewness of the number of e-folds.

    Parameters
    ----------
    potential : function
        The potential.
    potential_dif : function
        The potential's first derivative.
    potential_ddif : function
        The potential's second derivative.
    phi_in : float
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
    mean =\
        mean_efolds(potential, potential_dif, potential_ddif, phi_in, phi_end)
    std =\
        variance_efolds(potential, potential_dif, potential_ddif, phi_in,
                        phi_end)**0.5
    skewness =\
        skewness_efolds(potential, potential_dif, potential_ddif, phi_in,
                        phi_end)
    kurtosis =\
        kurtosis_efolds(potential, potential_dif, potential_ddif, phi_in,
                        phi_end)

    def higher_order_egdeworth_term(y):
        norm_y = (y-mean)/std
        skew_term = np.divide(skewness*hermite_poly3(norm_y), 6)
        kurtosis_term = np.divide(kurtosis*hermite_poly4(norm_y), 24)
        skew_squared_term =\
            np.divide(hermite_poly6(norm_y)*skewness**2, 72)
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
def hermite_poly3(y):
    hermite_poly3 = y**3-3*y
    return hermite_poly3


# This is the "probabilist's Hermite polynomial", which is different to the
# "physicist's Hermite polynomials" used by SciPy
def hermite_poly4(y):
    hermite_poly4 = y**4-6*y+3
    return hermite_poly4


# This is the "probabilist's Hermite polynomial", which is different to the
# "physicist's Hermite polynomials" used by SciPy
def hermite_poly6(y):
    hermite_poly6 = y**6-15*y**4+45*y**2-15
    return hermite_poly6
