'''
Lognormal Mean
--------------
This module estimates the mean of a lognormal distribution using the maxiumum
liklihood method from `Shen--Brown--Zhi 2006`_.

.. _Shen--Brown--Zhi 2006: https://pubmed.ncbi.nlm.nih.gov/16345103/
'''

import numpy as np


# The data provided needs to be raw data - the log is then taken in the
# function uses the maximum likelihood
# Shen 2006 Statist. Med. 2006; 25:3023–3038
def log_normal_mean(weights):
    """Returns the height of the histogram bar.

    Parameters
    ----------
    weights: numpy.ndarray
        The distribution of weights whose mean is desired.
    Returns
    -------
    mean : float
        The lognormal estimate for the mean of the provided array.
    """
    weights_log = np.log(weights)
    weights_log_mean = np.mean(weights_log)
    weights_log_std = np.std(weights_log, ddof=1)

    mean = np.exp(weights_log_mean+0.5*weights_log_std**2)
    return mean
