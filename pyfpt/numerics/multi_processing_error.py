'''
Multiprocessing Error
---------------------
This module alerts the user to a possible multiprocessing error. This occurs
when the data from different cores is incorrectly combined, with weights not
corresponding to the data.
'''


import numpy as np


# This function tests if a multiprocessing error has occured. This is when the
# data from the different cores becomes mixed, and the weights and N are not
# correct
def multi_processing_error(data, weights):
    """Alerts the user to possible multiprocessing error if the data is
    and log of the weights is sufficiently uncorrelated.

    Parameters
    ----------
    data : numpy.ndarray
        Input first-passage time data.
    weights: numpy.ndarray
        Associated weights to the first-passage time data. Must be a one-to-one
        correspondance between them.
    """
    # Checking if multipprocessing error occured, by looking at correlation
    pearson_corr = np.corrcoef(data, np.log10(weights))
    pearson_corr = pearson_corr[0, 1]

    if abs(pearson_corr) < 0.55:  # Data is uncorrelated
        print('Possible multiprocessing error occured, see documentation')
