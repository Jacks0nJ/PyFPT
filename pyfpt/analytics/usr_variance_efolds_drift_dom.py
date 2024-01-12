'''
USR Inflation Variance e-folds Drift Domination
-------------------------------------------
This module calculates the the varaince of number of e-folds in the drift
dominated (low diffusion) limit of ultra-slow-roll inflation in a flat
potential at next-to-leading order using equation 3.10 from
`Pattison et al 2022`_.

.. _Pattison et al 2022: https://arxiv.org/abs/2101.05741
'''

import numpy as np


def usr_variance_efolds_drift_dom(x, y, mu):
    """ Returns the varaince of number of e-folds is USR inflation in drift
    domination.

    Parameters
    ----------
    x : float
        The rescaled and dimensionaless field value. Must be equal to or
        smaller than 1.
    y: float
        The rescaled and dimensionaless field velocity. Drift domination is
        only true when this is greater than 1.
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
        if y < 1.:
            raise ValueError("Velocity y must be equal to or greater than 1")
    else:
        if any(y < 1.):
            raise ValueError("Velocity y must be equal to or greater than 1")

    mean_cls = -np.log(1-x/y)/3
    usr_var_efolds = mean_cls*np.divide(2, (3*mu*(y-x))**2)
    return usr_var_efolds
