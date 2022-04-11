import pandas as pd


def save_data_to_file(sim_N_dist, w_values, phi_i, num_sims, bias=0):
    data_dict_raw = {}
    data_dict_raw['N'] = sim_N_dist
    if bias > 0:
        data_dict_raw['w'] = w_values

    data_pandas_raw = pd.DataFrame(data_dict_raw)

    raw_file_name = 'IS_data_phi_i_' + ('%s' % float('%.3g' % phi_i)) +\
        '_iterations_' + str(num_sims) + '_bias_' +\
        ('%s' % float('%.3g' % bias)) + '.csv'
    # Saving to a directory for the language used

    data_pandas_raw.to_csv(raw_file_name)

    print('Saved data to file '+raw_file_name)
