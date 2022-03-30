'''
Reduced Potential Derivative
----------------------------
This module reduces the passed potential derivative function to its
dimensionless form given in equation 2.1 of `Vennin-Starobinsky 2015`_.

.. _Vennin-Starobinsky 2015: https://arxiv.org/abs/1506.04732
'''

import numpy as np
PI = np.pi
M_PL = 1.0


def reduced_potential_diff(V_diff):
    """Returns the reduced potential derivative as a function

    Parameters
    ----------
    V_diff : function
        The potential's second derivative

    Returns
    -------
    v_diff : function
        the reduced potential derivative

    """
    def v_diff(phi):
        v_diff_value = V_diff(phi)/(24*(PI**2)*(M_PL**4))
        return v_diff_value
    return v_diff
