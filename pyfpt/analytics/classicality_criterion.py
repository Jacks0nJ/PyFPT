'''
Classicality Criterion
----------------------
This module uses the classicality criterion (equation 3.27) from
`Vennin--Starobinsky 2015`_ to see if the inflation model investigated will
have dynamics which will deviate strongly from classical prediction. If the
returned :math:`{\\eta}` parameter is of order unity, stochastic effects
dominate.

.. _Vennin--Starobinsky 2015: https://arxiv.org/abs/1506.04732
'''

from .reduced_potential import reduced_potential
from .reduced_potential_diff import reduced_potential_diff
from .reduced_potential_ddiff import reduced_potential_ddiff


def classicality_criterion(V, V_dif, V_ddif, phi_i):
    """ Returns eta for the provided potential.

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

    Returns
    -------
    eta : float
        the :math:`{\\eta}` parameter.

    """

    v_func = reduced_potential(V)
    V_dif_func = reduced_potential_diff(V_dif)
    V_ddif_func = reduced_potential_ddiff(V_ddif)

    v = v_func(phi_i)
    V_dif = V_dif_func(phi_i)
    V_ddif = V_ddif_func(phi_i)

    eta = abs(2*v - (V_ddif*v**2)/(V_dif**2))
    return eta
