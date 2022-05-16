'''
Lognormal Height
----------------
This module estimates the unnormalised histogram bar height for a bin of
first-passage times of weighted data, assuming the weights are drwan from an
underlying lognormal distribution.
'''


from .log_normal_mean import log_normal_mean


def log_normal_height(weights):
    """Returns the unnoramlised height of the histogram bar.

    Parameters
    ----------
    weights: numpy.ndarray
        The distribution of weights whose height is desired.
    Returns
    -------
    height : float
        Lognormal estimate for the histogram bar height of this bin.
    """
    height = len(weights)*log_normal_mean(weights)
    return height
