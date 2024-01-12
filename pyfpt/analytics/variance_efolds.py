'''
Variance of the Number of e-folds
---------------------------------
This module calculates the variance of the number of e-folds in low diffusion
limit using equation 3.35 from `Vennin--Starobinsky 2015`_.

.. _Vennin--Starobinsky 2015: https://arxiv.org/abs/1506.04732
'''

import numpy as np
from scipy import integrate

from .reduced_potential import reduced_potential
from .reduced_potential_diff import reduced_potential_diff
from .reduced_potential_ddiff import reduced_potential_ddiff

planck_mass = 1


# Equation 3.35 in Vennin 2015
def variance_efolds(potential, potential_dif, potential_ddif, phi_in, phi_end):
    """Returns the variance of the number of e-folds.

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
    phi_end : float
        The end scalar field value.

    Returns
    -------
    var_efolds : float
        the variance of the number of e-folds.

    """
    v_func = reduced_potential(potential)
    v_dif_func = reduced_potential_diff(potential_dif)
    v_ddif_func = reduced_potential_ddiff(potential_ddif)

    def integrand_calculator(phi):
        # Pre calculating values
        v = v_func(phi)
        v_dif = v_dif_func(phi)
        v_ddif = v_ddif_func(phi)
        non_classical = 6*v-np.divide(5*(v**2)*v_ddif, v_dif**2)
        constant_factor = 2/(planck_mass**4)

        integrand = constant_factor*np.divide(v**4, v_dif**3)*(1+non_classical)
        return integrand
    var_efolds, er = integrate.quad(integrand_calculator, phi_end, phi_in)
    return var_efolds
