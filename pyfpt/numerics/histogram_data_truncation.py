'''
Histogram Data Truncation
-------------------------
This module truncates the first-passage time data (and its associated weights)
above the specified threshold. The main purpose is to truncate runs which
exceeded the maximum time.
'''

import numpy as np


# Trucates data above a certain threshold. Has options for both weights and if
# a the trucation needs to be rounded up to a certain value.
def histogram_data_truncation(data, threshold, weights=0,
                              num_sub_samples=None):
    """Returns truncated first-passage time data and the associated weights
    if provided.

    Parameters
    ----------
    data : numpy.ndarray
        Input first-passage time data.
    threshold : scalar
        ``data`` below the threshold will be kept, above it will be truncated.
    weights: numpy.ndarray, optional
        Associated weights to the first-passage time data. Must be a one-to-one
        correspondence between them.
    num_sub_samples : int, optional
        The number of subsamples if the naive estimator error estimation is
        used. This means the truncated data will always be an integer multiple
        of ``num_sub_samples``, such that jackknife resampling can be done.
    -------
    truncated_data : numpy.ndarray
        The truncated ``data``.
    truncated_data : numpy.ndarray, optional
        The truncated ``weights``, if provided.
    """
    # There are weird errors if I use None for a null input of weights, so
    # integer
    if isinstance(weights, int):
        if num_sub_samples is None:
            # Simplest case, just truncate data above threshold
            truncated_data = data[data < threshold]
        elif isinstance(num_sub_samples, int):
            data = np.sort(data)
            num_above_threshold = len(data[data > threshold])
            # Want to remove a full subsamples worth, so can be subdivided
            # later. So rounding the number removed up.
            rounded =\
                round(num_above_threshold/num_sub_samples)+1
            # Removing a full subsample
            rounded_num_above_threshold = rounded*num_sub_samples
            truncated_data = data[:-rounded_num_above_threshold]
        return truncated_data

    else:
        if num_sub_samples is None:
            # Simplest case again, but also truncated in weights. Need to
            # define the logic first, so the weights are not truncated based on
            # already truncated data.
            data_remove_logic = data < threshold
            truncated_data = data[data_remove_logic]
            truncated_weights = weights[data_remove_logic]
        elif isinstance(num_sub_samples, int):
            # Sort in order of increasing Ns
            sort_idx = np.argsort(data)
            data = data[sort_idx]
            weights = weights[sort_idx]
            num_above_threshold = len(data[data > threshold])
            # Want to remove a full subsamples worth, so can be subdivided
            # later. So rounding the number removed up.
            if num_above_threshold > 0:
                rounded = round(num_above_threshold/num_sub_samples)+1
                rounded_num_above_threshold = rounded*num_sub_samples
                truncated_data = data[:-rounded_num_above_threshold]
                truncated_weights = weights[:-rounded_num_above_threshold]
                return data[:-rounded_num_above_threshold], \
                    weights[:-rounded_num_above_threshold]
            else:
                # If none above the thresold, retuurn all of the data.
                truncated_data = data
                truncated_weights = weights
        else:
            raise ValueError('num_sub_samples must be integer')
        return truncated_data, truncated_weights
