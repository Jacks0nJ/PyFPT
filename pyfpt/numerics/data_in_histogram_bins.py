'''
Data in Histogram Bins
----------------------
This module subdivides the first-passage time data and its associated weights
according to the first-passage time bins used in the estimation of the target
distribution.
'''

import numpy as np


def data_in_histogram_bins(data, weights, bin_edges):
    """Returns first-passage time data and the associated weights in columns
    corresponding to the provided bins edges. The number of rows corresponds
    to the largest bin, with empty elements filled with zeros.

    Parameters
    ----------
    data : numpy.ndarray
        Input first-passage time data.
    weights: numpy.ndarray
        Associated weights to the first-passage time data. Must be a one-to-one
        correspondence between them.
    bin_edges : sequence of scalars
        The bin edges of the histogram used in the estimation of the target
        distribution.
    Returns
    -------
    data_columned : numpy.ndarray
        The data separated into columns, with the data in each column
        corresponding to a particular bin.
    weights_columned : numpy.ndarray
        The weights separated into columns, with the weights in each column
        corresponding to the associated data in a particular bin.
    """
    # Create empty arrays large enough to store all of the data, as it is
    # possible all of the data is in 1 bins.
    data_columned = np.zeros([len(data), len(bin_edges)-1])
    weights_columned = np.zeros([len(data), len(bin_edges)-1])

    # Keep track of the largest bin, so excess zero rows can be removed
    largest_bin = 0

    # This loops through all of the bins, finding the data in that bin and
    # stores it. Stops before last bin.
    for i in range(len(bin_edges)-2):
        # Find the data in this bin using slicing
        data_logic = (data >= bin_edges[i]) & (data < bin_edges[i+1])
        data_slice = data[data_logic]

        # Stores this data in the corresponding coloumn
        data_columned[0:len(data_slice), i] = data_slice

        # Do the same for the weights
        weights_columned[0:len(data_slice), i] = weights[data_logic]

        # See if this bin has more data than any of the previous
        if len(data_slice) > largest_bin:
            largest_bin = len(data_slice)

    # The final bin also includes the last value, so has an equals in less than
    data_logic = (data >= bin_edges[len(bin_edges)-2]) &\
        (data <= bin_edges[len(bin_edges)-1])
    data_slice = data[data_logic]
    # Store this last bin
    data_columned[0:len(data_slice), len(bin_edges)-2] = data_slice
    weights_columned[0:len(data_slice), len(bin_edges)-2] = weights[data_logic]

    # Now remove excess zero rows using knowledge of largest bin
    data_columned = data_columned[:largest_bin, :]
    weights_columned = weights_columned[:largest_bin, :]
    return data_columned, weights_columned
