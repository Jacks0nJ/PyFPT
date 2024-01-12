'''
Reduced Potential Second Derivative
-----------------------------------
This module reduces the passed potential second derivative function to its
dimensionless form given in equation 2.1 of `Vennin--Starobinsky 2015`_.

.. _Vennin--Starobinsky 2015: https://arxiv.org/abs/1506.04732
'''


import numpy as np
pi = np.pi
planck_mass = 1.0


def reduced_potential_ddiff(potential_ddiff):
    """Returns the reduced potential second derivative as a function

    Parameters
    ----------
    potential_ddif : function
        The potential's second derivative.

    Returns
    -------
    v_ddif : function
        the reduced potential second derivative.

    """
    def v_ddiff(phi):
        v_ddiff_value =\
            np.divide(potential_ddiff(phi), 24*(pi**2)*(planck_mass**4))
        return v_ddiff_value
    return v_ddiff
