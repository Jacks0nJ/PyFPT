'''
Save Data to File
-----------------
This module saves the raw first-passage time data and its associated weights
(if provided) to a comma separated value file using pandas in the same
directory as where PyFPT is run from.
'''

import pandas as pd
import numpy as np


def save_data_to_file(data, x_in, num_runs, weights=None,
                      extra_label=None):
    """Saves the provided data and the associated weights to a file, titled
    "IS_data_x_in_<x_in>_iterations_<num_runs>(_<extra_label>).csv"
    The first-passage time data is stored as "FPTs" and the associated weights
    as "ws".

    Parameters
    ----------
    data : list or numpy.ndarray
        Input first-passage time data.
    x_in : float or numpy.ndarray
        The initial position value. If multi-dimensional, first value is used.
    num_runs : int
        The number of simulation runs.
    weights: list or numpy.ndarray, optional
        Associated weights to the first-passage time data. Must be a one-to-one
        correspondance between them. Defaults to ``None``.
    extra_label: string, optional
        Optional extra string to label file.
    """
    data_dict_raw = {}
    data_dict_raw['FPTs'] = data
    if isinstance(weights, np.ndarray) is True or isinstance(weights, list):
        data_dict_raw['ws'] = weights

    data_pandas_raw = pd.DataFrame(data_dict_raw)

    if isinstance(x_in, np.ndarray) is True:
        x_in = x_in[0]

    raw_file_name = 'IS_data_x_in_' + ('%s' % float('%.3g' % x_in)) +\
        '_iterations_' + str(num_runs)

    if isinstance(extra_label, str) is True:
        raw_file_name += extra_label + '.csv'
    else:
        raw_file_name += '.csv'
    # Saving to a directory for the language used

    data_pandas_raw.to_csv(raw_file_name)

    print('Saved data to file '+raw_file_name)
