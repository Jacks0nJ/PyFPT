'''
Gaussian PDF
-------------
This module returns the Gaussian probability density function (PDF)
for first-passage times in the low-diffusion limit, using the results from
`Vennin--Starobinsky 2015`_ to calculate the required moments, as a function.

.. _Vennin--Starobinsky 2015: https://arxiv.org/abs/1506.04732
'''
import numpy as np

from .mean_efolds import mean_efolds
from .variance_efolds import variance_efolds

pi = np.pi


# This returns a function which returns the Edgeworth expansion
def gaussian_pdf(potential, potential_dif, potential_ddif, phi_in, phi_end):
    """Returns the Gaussian approximation in the low-diffusion limit.

    Parameters
    ----------
    potential : function
        The potential.
    potential_dif : function
        The potential's first derivative.
    potential_ddif : function
        The potential's second derivative/
    phi_in : float
        The initial field value.
    phi_end : float
        The end scalar field value.

    Returns
    -------
    gaussian_function : function
        The Gaussian approximation for the probability density function at the
        provided e-fold values, i.e. a function of ``(N)``.

    """
    mean =\
        mean_efolds(potential, potential_dif, potential_ddif, phi_in, phi_end)
    std =\
        variance_efolds(potential, potential_dif, potential_ddif, phi_in,
                        phi_end)**0.5

    def gaussian_function(efolds):
        norm_efolds = (efolds-mean)/std

        gaussian = np.divide(np.exp(-0.5*norm_efolds**2), std*(2*pi)**0.5)
        return gaussian

    return gaussian_function
