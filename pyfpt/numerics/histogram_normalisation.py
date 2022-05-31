'''
Histogram Normalisation
-------------------------------
This module calculates the histogram normalisation using the formula
``num_runs*bin_width``. Therefore, the total area of the histogram may not be
1. Instead, each bin is normalised.
'''
from numpy import diff


# Returns the normalisation factor for a histogram, including one with weights
def histogram_normalisation(bins, num_runs):
    """Returns histogram normalisation. If evenly spaced bins are used, then a
    scalar is returned. Otherwise, the correct normalisation for each bin is
    returned.

    Parameters
    ----------
    bins : int or sequence of scalars
        Either the number of evenly spaced bins used in the histogram or the
        bin edges used if a sequence.
    num_runs : int
        The number of simulation runs used in the histogram.
    Returns
    -------
    normalisation : float or sequence of scalars
        If ``bins`` was an int, then the normalisation as a float is returned.
        If ``bins`` was a sequence, then the normalisation per bin is returned.
    """
    if isinstance(num_runs, int) is False:
        raise ValueError('num_runs must be an integer')

    normalisation = num_runs*diff(bins)
    return normalisation
