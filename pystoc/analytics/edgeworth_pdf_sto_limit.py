'''
Edgeworth PDF
-------------
This module returns the `Edgeworth series`_ probability density function (PDF)
for first-passage times in the low-diffusion limit, using the results from
`Vennin-Starobinsky 2015`_ to calcuate the required moments, as a function.

.. _Edgeworth series: https://en.wikipedia.org/wiki/Edgeworth_series
.. _Vennin-Starobinsky 2015: https://arxiv.org/abs/1506.04732
'''
import numpy as np

from .mean_N_sto_limit import mean_N_sto_limit
from .variance_N_sto_limit import variance_N_sto_limit
from .skewness_N_sto_limit import skewness_N_sto_limit
from .kurtosis_N_sto_limit import kurtosis_N_sto_limit

PI = np.pi


# This returns a function which returns the Edgeworth expansion
def edgeworth_pdf_sto_limit(V, V_dif, V_ddif, phi_i, phi_end):
    """ Returns the Edgeworth expansion in the low-diffusion limit.

    Parameters
    ----------
    V : function
        The potential
    V_dif : function
        The potential's first derivative
    V_ddif : function
        The potential second derivative
    phi_i : float
        The initial field value
    phi_end : float
        The end scalar field value.

    Returns
    -------
    edgeworth_function : function
        The Edgeworth expansion.

    """
    mean = mean_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)
    std = variance_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)**0.5
    skewness = skewness_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)
    kurtosis = kurtosis_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)

    def edgeworth_function(N):
        norm_N = (N-mean)/std

        skew_term = np.divide(skewness*He3(norm_N), 6)
        kurtosis_term = np.divide(kurtosis*He4(norm_N), 24)
        skew_squared_term = np.divide(He6(norm_N)*skewness**2, 72)

        gaussian = np.divide(np.exp(-0.5*norm_N**2), std*(2*PI)**0.5)
        return gaussian*(1+skew_term+kurtosis_term+skew_squared_term)

    return edgeworth_function


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
