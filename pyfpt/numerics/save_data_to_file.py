'''
Save Data to File
-----------------
This module saves the raw first-passage time data and its associated weights to
a comma seperated value file using pandas in the same directory as where PyFPT
is run from.
'''

import pandas as pd


def save_data_to_file(data, weights, x_in, num_runs, bias, extra_label=None):
    """Saves the provided data and the associated weights to a file, titled
    "IS_data_x_in_<x_in>_iterations_<num_runs>_bias_<bias>(<extra_label>).csv"

    Parameters
    ----------
    data : numpy.ndarray
        Input first-passage time data.
    weights: numpy.ndarray
        Associated weights to the first-passage time data. Must be a one-to-one
        correspondance between them.
    x_in : float
        The initial position value.
    num_runs : int
        The number of simulation runs.
    bias : float
        The coefficent of the diffusion used define the bias.
    extra_label: string, optional
        Optional extra string to label file.
    """
    data_dict_raw = {}
    data_dict_raw['FPTs'] = data
    data_dict_raw['ws'] = weights

    data_pandas_raw = pd.DataFrame(data_dict_raw)

    raw_file_name = 'IS_data_x_in_' + ('%s' % float('%.3g' % x_in)) +\
        '_iterations_' + str(num_runs) + '_bias_' +\
        ('%s' % float('%.3g' % bias))
    if isinstance(extra_label, str) is True:
        raw_file_name += extra_label + '.csv'
    else:
        raw_file_name += '.csv'
    # Saving to a directory for the language used

    data_pandas_raw.to_csv(raw_file_name)

    print('Saved data to file '+raw_file_name)
