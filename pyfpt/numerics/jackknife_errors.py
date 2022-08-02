'''
Jackknife Errors
----------------
This module calculates the errors of the histogram bars by using a
simplified jackknife resampling method. The data is sub-sampled into many
histograms with the same bins. This way a distribution of heights, for each
bin, can be made. The standard deviation of the distribution of heights for as
bin, by the central limit theorem, then gives error when divided root of the
number sub samples.
'''


import numpy as np

from .histogram_normalisation import histogram_normalisation
from .data_in_histogram_bins import data_in_histogram_bins


def jackknife_errors(data_input, weights_input, bins, num_sub_samps):
    """Returns the jackknife resampling errors for the estimation of histogram
    bar height, for the provided weighted data and bin edges.

    Parameters
    ----------
    data : numpy.ndarray
        Input first-passage time data.
    weights: numpy.ndarray
        Associated weights to the first-passage time data. Must be a one-to-one
        correspondence between them.
    bins : sequence
        Defines the bin edges of the histogram, including the left edge of the
        first bin and the right edge of the last bin. The widths can vary.
    num_sub_samples : int
        The number of subsamples used in jackknife estimation of the errors
        used for the ``'naive'`` estimator. Must divide into number of data
        points with no remainder.
    -------
    errors : numpy.ndarray
        The jackknife errors.
    """
    # Make an array of random indexs
    indx = np.arange(0, len(data_input), 1)
    np.random.shuffle(indx)
    # This allows the data to be randomised and keep the weights matched
    data = data_input[indx]
    weights = weights_input[indx]
    num_bins = len(bins)-1  # bins is an array of the side, so one less

    height_array = np.zeros((num_bins, num_sub_samps))  # Storage

    # Need to check that the data can be evenly divided
    if len(data)/num_sub_samps == 0:
        pass

    else:
        # If the data canot be evenly divided, must truncate it to work. This
        # will remove data randomly
        overspill = len(data) % num_sub_samps
        data = data[:-overspill]
        weights = weights[:-overspill]
        print("Data which could not be evenly divided " +
              "into the subsamples given. Randomly truncating to fit.")

    # Next organise into subsamples
    data =\
        np.reshape(data, (int(data.shape[0]/num_sub_samps), num_sub_samps))
    weights =\
        np.reshape(weights, (int(weights.shape[0]/num_sub_samps),
                             num_sub_samps))

    # Find the heights of the histograms, for each sample
    for i in range(num_sub_samps):
        # We need to manually calculate the bin height, as numpy's scheme does
        # not have the precision when weights are very small.
        _, ws = data_in_histogram_bins(data[:, i], weights[:, i], bins)
        height_raw = np.sum(ws, axis=0)
        norm = histogram_normalisation(bins, len(data[:, i]))
        height_array[:, i] = height_raw/norm

    # To store the errors
    errors = np.zeros(num_bins)

    # Need the sqaure root, as the error in jackknife scales with it.s
    sqrt_sub_samples = np.sqrt(num_sub_samps)

    # Now can find the standard deviation of the heights, as the bins are the
    # same. Then divide by the sqaure root of the number of samples by
    # jackknife and you have the error
    for j in range(num_bins):
        bars = height_array[j, :]
        if np.any([np.abs(bars) > 0]):
            # Used to be just np.std(bars)
            errors[j] = np.std(bars[bars > 0])/sqrt_sub_samples
        else:
            errors[j] = 0  # Remove any empty histogram bars

    return errors
