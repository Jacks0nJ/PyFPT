'''
Reduced Potential
-----------------
This module reduces the passed potential function to its dimensionless form
given in equation 2.1 of `Vennin--Starobinsky 2015`_.

.. _Vennin--Starobinsky 2015: https://arxiv.org/abs/1506.04732
'''


import numpy as np
PI = np.pi
M_PL = 1.0


def reduced_potential(V):
    """Returns the reduced potential as a function

    Parameters
    ----------
    V : function
        The potential

    Returns
    -------
    v : function
        the reduced potential

    """
    def v(phi):
        v_value = V(phi)/(24*(PI**2)*(M_PL**4))
        return v_value
    return v
