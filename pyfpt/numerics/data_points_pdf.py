'''
Probability Density of the Data
-------------------------------
This module post processes the first-passage times and weights to estimate the
probability density of the target distribution.
'''


import numpy as np


from .histogram_normalisation import histogram_normalisation
from .data_in_histogram_bins import data_in_histogram_bins
from .jackknife_errors import\
    jackknife_errors
from .log_normal_height import log_normal_height
from .log_normal_error import log_normal_error
from .lognormality_check import lognormality_check


def data_points_pdf(data, weights, estimator,
                    bins=50, min_bin_size=400, num_sub_samples=20,
                    display=True):
    """Returns the (truncated) histogram bin centres, heights and errors, using
    the provided estimator method.

    Parameters
    ----------
    data : numpy.ndarray
        Input first-passage time data.
    weights : numpy.ndarray
        Associated weights to the first-passage time data. Must be a one-to-one
        correspondence between them.
    estimator : string
        The estimator used to reconstruct the target distribution probability
        density from the importance sample. If ``'lognormal'``, it assumes the
        weights in each bin follow a lognomral distribution. If ``'naive'``, no
        assumption is made but more runs are required for convergence.
    bins : int or list, optional
        If bins is an integer, it defines the number equal width bins for the
        first-passage times. If bins is a list or numpy array, it defines the
        bin edges, including the left edge of the first bin and the right edge
        of the last bin. The widths can vary. Defaults to 50 evenly spaced
        bins.
    min_bin_size : int, optional
        The minimum number of runs per bin included in the data analysis.
        If a bin has less than this number, it is truncated. Defaults to 400.
    num_sub_samples : int, optional
        The number of subsamples used in jackknife estimation of the errors
        used for the ``'naive'`` estimator. Defaults to 20 when ``estimator``
        is ``'naive'``.
    display : bool, optional
        If True, p-value plots of both the real data, and the theoretical
        expectation if the underlying distribution is truly lognormal, are
        displayed using ``fpt.numerics.lognormality_check`` if a p-value is
        below the specified threshold.
    Returns
    -------
    bin_centres : numpy.ndarray
        The centres of the histogram bins (after truncation of underfilled
        bins).
    heights : numpy.ndarray
        The heights of the normalised histogram bars (after truncation of
        underfilled bins).
    errors : numpy.ndarray
        The errors in estimating the heights (after truncation of
        underfilled bins).
    num_runs_used : int
        The number of runs used (after truncation of underfilled bins).
    bins : numpy.ndarray
        The untruncated bin edges.

    """
    num_runs = len(data)

    # If the number of bins used has been specified
    if isinstance(bins, int) is True:
        num_bins = bins
        # Want raw heights of histogram bars
        heights_raw, bins =\
            np.histogram(data, num_bins, weights=weights)
    # If the bins have been specified
    else:
        num_bins = len(bins)-1  # as bins is the bin edges, so plus 1
        # Want raw heights of histogram bars
        heights_raw, bins =\
            np.histogram(data, bins=bins, weights=weights)

    # Calculate the normalisation of the histogram
    histogram_norm =\
        histogram_normalisation(bins, num_runs)

    # Need to know the data and weights in each bin to estimate the errors
    data_in_bins, weights_in_bins =\
        data_in_histogram_bins(data, weights, bins)

    # Predictions need the bin centre to make comparison
    bin_centres = np.array([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)])

    # Removing underfilled bins if needed
    if isinstance(min_bin_size, int) is True:
        # Then loop through to find where length is greater than min_bin_size
        filled_bins = []
        num_runs_used = len(data)

        for i in range(len(bins)-1):
            data_in_bin = data_in_bins[:, i]
            data_in_bin = data_in_bin[data_in_bin > 0]
            # First, remove if empty
            if len(data_in_bin) == 0:
                filled_bins.append(False)
            # If there is enough data in this bin
            elif len(data_in_bin) >= min_bin_size:
                filled_bins.append(True)
            # Don't include under filled tail bins
            else:
                filled_bins.append(False)
                # Reflect in number of succesful simulatios
                num_runs_used -= len(data_in_bin)
        bin_centres = bin_centres[filled_bins]
    else:
        print('min_bin_size is not an integrer, defaulting to 0')

    # Now estimating the probability density function
    if estimator == 'naive':
        heights = heights_raw/histogram_norm
        errors = jackknife_errors(data, weights, bins, num_sub_samples)
        if isinstance(min_bin_size, int) is True:
            heights = heights[filled_bins]
            errors = errors[filled_bins]

        # For consistancy with the lognormal method, need to provide both upp
        # and lower errors. As the errors are symmetrical, simply copy them.
        errors = np.tile(errors, (2, 1))

    elif estimator == 'lognormal':

        heights_est = np.zeros(num_bins)
        # The errors for the log-normal case are asymmetric
        errors_est = np.zeros((2, num_bins))
        for i in range(num_bins):
            w = weights_in_bins[:, i]
            # Only calculate filled bins
            if filled_bins[i] is True or\
                (np.any([w > 0]) is True and isinstance(min_bin_size, int)
                 is False):
                w = w[w > 0]
                heights_est[i] = log_normal_height(w)
                errors_est[:, i] = log_normal_error(w)

        # Include only filled values
        # Remember to normalise errors as well
        heights = heights_est/histogram_norm
        heights = heights[errors_est[0, :] > 0]

        # The errors are a 2D array, so need to slice correctly
        errors = errors_est/histogram_norm
        errors = errors[:, errors_est[0, :] > 0]

        # Checking p-values if lognormal was used.
        # Need to provide only weights of the filled bins
        _ = lognormality_check(bin_centres, weights_in_bins[:, filled_bins],
                               display=display)

    else:
        raise ValueError('Not valid estimator method')

    return bin_centres, heights, errors, num_runs_used, bins
