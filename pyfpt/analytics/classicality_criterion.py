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


def classicality_criterion(potential, potential_dif, potential_ddif, phi_in):
    """ Returns eta for the provided potential.

    Parameters
    ----------
    potential : function
        The potential
    potential_dif : function
        The potential's first derivative
    potential_ddif : function
        The potential second derivative
    phi_in : float
        The initial field value

    Returns
    -------
    eta : float
        the :math:`{\\eta}` parameter.

    """

    v_func = reduced_potential(potential)
    v_dif_func = reduced_potential_diff(potential_dif)
    v_ddif_func = reduced_potential_ddiff(potential_ddif)

    v = v_func(phi_in)
    v_dif = v_dif_func(phi_in)
    v_ddif = v_ddif_func(phi_in)

    eta = abs(2*v - (v_ddif*v**2)/(v_dif**2))
    return eta
