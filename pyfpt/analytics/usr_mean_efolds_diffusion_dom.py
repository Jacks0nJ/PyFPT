'''
USR Inflation Mean e-folds Diffusion Domination
-----------------------------------------------
This module calculates the mean number of e-folds in the diffusion dominated
limit of ultra-slow-roll (USR) inflation in a flat potential at next-to-leading
order using equation 4.14 from `Pattison et al 2022`_.

.. _Pattison et al 2022: https://arxiv.org/abs/2101.05741
'''

import numpy as np


def usr_mean_efolds_diffusion_dom(x, y, mu):
    """ Returns the mean number of e-folds is USR inflation in diffusion
    domination.

    Parameters
    ----------
    x : float
        The rescaled and dimensionaless field value. Must be equal to or
        smaller than 1.
    y: float
        The rescaled and dimensionaless field velocity. Diffusion domination is
        only true when this is less than 1.
    mu: float
        The dimensionaless effective flat potential width.
    Returns
    -------
    usr_mean_efolds : float
        the mean number of e-folds.

    """
    # Check if the user has input correct values
    if isinstance(x, float) is True or isinstance(x, int) is True:
        if x > 1.:
            raise ValueError("Field value x must be equal to or less than 1")
    else:
        if any(x > 1.):
            raise ValueError("Field value x must be equal to or less than 1")
    if isinstance(y, float) is True or isinstance(y, int) is True:
        if y > 1.:
            print("WARNING: this approximation is only valid for y<1")
    else:
        if any(y > 1.):
            print("WARNING: this approximation is only valid for y<1")

    s_three = 3**0.5
    leading_oder = x*(1-x/2)*mu**2
    cosh_term = np.divide(np.cosh(s_three*mu*(1-x)), np.cosh(s_three*mu))
    tanh_term = np.divide(np.sinh(s_three*mu*x),
                          s_three*mu*np.cosh(s_three*mu))
    return leading_oder + y*(x - 1 + cosh_term - tanh_term)*mu**2
