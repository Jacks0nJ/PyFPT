'''
Gaussian PDF
-------------
This module returns the Gaussian probability density function (PDF)
for first-passage times in the low-diffusion limit, using the results from
`Vennin--Starobinsky 2015`_ to calcuate the required moments, as a function.

.. _Vennin--Starobinsky 2015: https://arxiv.org/abs/1506.04732
'''
import numpy as np

from .mean_N_sto_limit import mean_N_sto_limit
from .variance_N_sto_limit import variance_N_sto_limit

PI = np.pi


# This returns a function which returns the Edgeworth expansion
def gaussian_pdf_sto_limit(V, V_dif, V_ddif, phi_i, phi_end):
    """ Returns the Gaussian approximation in the low-diffusion limit.

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
    gaussian_function : function
        The Gaussian approximation.

    """
    mean = mean_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)
    std = variance_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)**0.5

    def gaussian_function(N):
        norm_N = (N-mean)/std

        gaussian = np.divide(np.exp(-0.5*norm_N**2), std*(2*PI)**0.5)
        return gaussian

    return gaussian_function
