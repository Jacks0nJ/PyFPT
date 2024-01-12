'''
Reduced Potential
-----------------
This module reduces the passed potential function to its dimensionless form
given in equation 2.1 of `Vennin--Starobinsky 2015`_.

.. _Vennin--Starobinsky 2015: https://arxiv.org/abs/1506.04732
'''


import numpy as np
pi = np.pi
planck_mass = 1.0


def reduced_potential(potential):
    """Returns the reduced potential as a function

    Parameters
    ----------
    potential : function
        The potential.

    Returns
    -------
    v : function
        the reduced potential.

    """
    def v(phi):
        v_value = potential(phi)/(24*(pi**2)*(planck_mass**4))
        return v_value
    return v
