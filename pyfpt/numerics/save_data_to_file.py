'''
Save Data to File
-----------------
This module saves the raw first-passage time data and its associated weights to
a comma seperated value file using pandas in the same directory as where PyFPT
is run from.
'''

import pandas as pd


def save_data_to_file(data, weights, phi_i, num_runs, bias):
    """Saves the provided data and the associated weights to a file, titled
    "IS_data_phi_i_<phi_i>_iterations_<num_runs>_bias_<bias>.csv"

    Parameters
    ----------
    data : numpy.ndarray
        Input first-passage time data.
    weights: numpy.ndarray
        Associated weights to the first-passage time data. Must be a one-to-one
        correspondance between them.
    phi_i : float
        The initial scalar field value.
    num_runs : int
        The number of simulation runs.
    bias : float
        The coefficent of the diffusion used define the bias.
    """
    data_dict_raw = {}
    data_dict_raw['N'] = data
    data_dict_raw['w'] = weights

    data_pandas_raw = pd.DataFrame(data_dict_raw)

    raw_file_name = 'IS_data_phi_i_' + ('%s' % float('%.3g' % phi_i)) +\
        '_iterations_' + str(num_runs) + '_bias_' +\
        ('%s' % float('%.3g' % bias)) + '.csv'
    # Saving to a directory for the language used

    data_pandas_raw.to_csv(raw_file_name)

    print('Saved data to file '+raw_file_name)
