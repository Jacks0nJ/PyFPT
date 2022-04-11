'''
Variance of the Number of e-folds
---------------------------------
This module calculates the variance of the number of e-folds in low diffusion
limit using equation 3.35 from `Vennin-Starobinsky 2015`_.

.. _Vennin-Starobinsky 2015: https://arxiv.org/abs/1506.04732
'''

import numpy as np
from scipy import integrate

from .reduced_potential import reduced_potential
from .reduced_potential_diff import reduced_potential_diff
from .reduced_potential_ddiff import reduced_potential_ddiff

M_PL = 1


# Equation 3.35 in Vennin 2015
def variance_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end):
    """Returns the variance of the number of e-folds.

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
    phi_end : float
        The end scalar field value.

    Returns
    -------
    var_N : float
        the variance of the number of e-folds.

    """
    v_func = reduced_potential(V)
    V_dif_func = reduced_potential_diff(V_dif)
    V_ddif_func = reduced_potential_ddiff(V_ddif)

    def integrand_calculator(phi):
        # Pre calculating values
        v = v_func(phi)
        V_dif = V_dif_func(phi)
        V_ddif = V_ddif_func(phi)
        non_classical = 6*v-np.divide(5*(v**2)*V_ddif, V_dif**2)
        constant_factor = 2/(M_PL**4)

        integrand = constant_factor*np.divide(v**4, V_dif**3)*(1+non_classical)
        return integrand
    var_N, er = integrate.quad(integrand_calculator, phi_end, phi_i)
    return var_N
