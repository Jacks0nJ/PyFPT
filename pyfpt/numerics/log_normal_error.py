'''
Lognormal Error
---------------------
This module calculates the error of the estimation of the probability density
of the target distribution from the sample distribution using the lognormal
method. This calculation is taken from `Zhou--Gao 1997`_.

.. _Zhou--Gao 1997: https://pubmed.ncbi.nlm.nih.gov/9131765/
'''


import numpy as np


def log_normal_error(weights, z_alpha=1):
    """Returns the errors on the estimation of the probability density using
    the lognormal method.

    Parameters
    ----------
    weights: numpy.ndarray
        Associated weights to the first-passage time data. Must be a one-to-one
        correspondance between them.
        bin edges used if a sequence.
    num_runs : int or float, optional
        The numer of standard errors in the returned quantity. Defaults to 1
        standard error.
    Returns
    -------
    errors : numpy.ndarray
        The asymmetric error, as lower and upper bounds.
    """
    log_w = np.log(weights)
    log_var = np.var(log_w, ddof=1)  # unbiased variance
    log_mean = np.mean(log_w)
    n = len(weights)
    log_err = z_alpha*np.sqrt(log_var/n+(log_var**2)/(2*n-2))
    upper_err = n*np.exp(log_mean+log_var/2)*(np.exp(log_err)-1)
    lower_err = n*np.exp(log_mean+log_var/2)*(1-np.exp(-log_err))
    return np.array([lower_err, upper_err])
