'''
Edgeworth PDF
-------------
This module returns the `Edgeworth series`_ probability density function (PDF)
for first-passage times in the low-diffusion limit, using the results from
`Vennin--Starobinsky 2015`_ to calculate the required moments, as a function.

.. _Edgeworth series: https://en.wikipedia.org/wiki/Edgeworth_series
.. _Vennin--Starobinsky 2015: https://arxiv.org/abs/1506.04732
'''
import numpy as np

from .mean_efolds import mean_efolds
from .variance_efolds import variance_efolds
from .skewness_efolds import skewness_efolds
from .kurtosis_efolds import kurtosis_efolds

pi = np.pi


# This returns a function which returns the Edgeworth expansion
def edgeworth_pdf(potential, potential_dif, potential_ddif, phi_in, phi_end):
    """ Returns the Edgeworth expansion in the low-diffusion limit.

    Parameters
    ----------
    potential : function
        The potential.
    potential_dif : function
        The potential's first derivative.
    potential_ddif : function
        The potential's second derivative.
    phi_in : float
        The initial field value.
    phi_end : float
        The end scalar field value.

    Returns
    -------
    edgeworth_function : function
        The Edgeworth expansion for the probability density function at the
        provided e-fold values, i.e. a function of ``(N)``.

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

    def edgeworth_function(efolds):
        norm_efolds = (efolds-mean)/std

        skew_term = np.divide(skewness*hermite_poly3(norm_efolds), 6)
        kurtosis_term = np.divide(kurtosis*hermite_poly4(norm_efolds), 24)
        skew_squared_term =\
            np.divide(hermite_poly6(norm_efolds)*skewness**2, 72)

        gaussian = np.divide(np.exp(-0.5*norm_efolds**2), std*(2*pi)**0.5)
        return gaussian*(1+skew_term+kurtosis_term+skew_squared_term)

    return edgeworth_function


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
