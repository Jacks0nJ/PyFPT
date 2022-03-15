import numpy as np


# Returns the data used in histogram bars as columns.
def histogram_data_in_bins(data, weights, bins):
    data_columned = np.zeros([len(data), len(bins)-1])
    weights_columned = np.zeros([len(data), len(bins)-1])
    # The bins have the same range until the end
    for i in range(len(bins)-2):
        data_logic = (data >= bins[i]) & (data < bins[i+1])
        data_slice = data[data_logic]
        data_columned[0:len(data_slice), i] = data_slice
        weights_columned[0:len(data_slice), i] = weights[data_logic]
    # The final bin also includes the last value, so has an equals in less than
    data_logic = (data >= bins[len(bins)-2]) & (data <= bins[len(bins)-1])
    data_slice = data[data_logic]
    data_columned[0:len(data_slice), len(bins)-2] = data_slice
    weights_columned[0:len(data_slice), len(bins)-2] = weights[data_logic]
    return data_columned, weights_columned
