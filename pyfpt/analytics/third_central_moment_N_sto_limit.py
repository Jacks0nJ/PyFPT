'''
Third Central Moment of the Number of e-folds
---------------------------------------------
This module calculates the third central moment of the number of e-folds in
low diffusion limit using equation 3.37 from `Vennin-Starobinsky 2015`_.

.. _Vennin-Starobinsky 2015: https://arxiv.org/abs/1506.04732
'''


import numpy as np
from scipy import integrate

from .reduced_potential import reduced_potential
from .reduced_potential_diff import reduced_potential_diff
from .reduced_potential_ddiff import reduced_potential_ddiff

M_PL = 1


# Equation 3.37 in Vennin 2015
def third_central_moment_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end):
    """Returns the third central moment of the number of e-folds.

    Parameters
    ----------
    V : function
        The potential
    V_dif : function
        The potential's first derivative
    V_ddif : function
        The potential second derivative
    phi_i : float
        The initial scalar field value
    phi_end : float
        The end scalar field value

    Returns
    -------
    third_moment_N : float
        the third central moment of the number of e-folds.

    """
    v_func = reduced_potential(V)
    V_dif_func = reduced_potential_diff(V_dif)
    V_ddif_func = reduced_potential_ddiff(V_ddif)

    def integrand_calculator(phi):
        # Pre calculating values
        v = v_func(phi)
        V_dif = V_dif_func(phi)
        V_ddif = V_ddif_func(phi)
        non_classical = 14*v-np.divide(11*(v**2)*V_ddif, V_dif**2)
        constant_factor = 12/(M_PL**6)

        integrand = constant_factor*np.divide(v**7, V_dif**5)*(1+non_classical)
        return integrand
    third_moment_N, er = integrate.quad(integrand_calculator, phi_end, phi_i)
    return third_moment_N
