'''
Fourth Central Moment of the Number of e-folds
----------------------------------------------
This module calculates the fourth central moment of the number of e-folds in
low diffusion limit using equation 3.40 from `Vennin--Starobinsky 2015`_.

.. _Vennin--Starobinsky 2015: https://arxiv.org/abs/1506.04732
'''


import numpy as np
from scipy import integrate

from .reduced_potential import reduced_potential
from .reduced_potential_diff import reduced_potential_diff
from .variance_efolds import variance_efolds

planck_mass = 1


# This is done using Vincent's calculations he gave me
def fourth_central_moment_efolds(potential, potential_dif, potential_ddif,
                                 phi_in, phi_end):
    """Returns the fourth central moment of the number of e-folds.

    Parameters
    ----------
    potential : function
        The potential
    potential_dif : function
        The potential's first derivative
    potential_ddif : function
        The potential second derivative
    phi_in : float
        The initial scalar field value
    phi_end : float
        The end scalar field value

    Returns
    -------
    fourth_moment_efolds : float
        the fourth central moment of the number of e-folds.

    """
    v_func = reduced_potential(potential)
    v_dif_func = reduced_potential_diff(potential_dif)

    def integrand_calculator(phi):
        # Pre calculating values
        v = v_func(phi)
        v_dif = v_dif_func(phi)
        constant_factor = 120/(planck_mass**8)

        integrand = constant_factor*np.divide(v**10, v_dif**7)
        return integrand
    non_guassian, er = integrate.quad(integrand_calculator, phi_end, phi_in)
    # As Vincent's method explicitly calculates the excess kurtosis, need to
    # add Wick's theorem term
    gaussian_4th_moment =\
        3*variance_efolds(potential, potential_dif, potential_ddif, phi_in,
                          phi_end)**2

    fourth_moment_efolds = gaussian_4th_moment+non_guassian

    return fourth_moment_efolds
