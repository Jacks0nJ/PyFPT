import numpy as np


# Trucates data above a certain threshold. Has options for both weights and if
# a the trucation needs to be rounded up to a certain value.
def histogram_data_truncation(data, threshold, weights=0,
                              num_sub_samples=None):
    if isinstance(weights, int):
        if num_sub_samples is None:
            return data[data < threshold]
        elif isinstance(num_sub_samples, int):
            data = np.sort(data)
            num_above_threshold = len(data[data > threshold])
            # Want to remove a full subsamples worth
            rounded_num_above_threshold =\
                round(num_above_threshold/num_sub_samples)+1
            return data[:-rounded_num_above_threshold]

    else:
        if num_sub_samples is None:
            data_remove_logic = data < threshold
            return data[data_remove_logic], weights[data_remove_logic]
        elif isinstance(num_sub_samples, int):
            # Sort in order of increasing Ns
            sort_idx = np.argsort(data)
            data = data[sort_idx]
            weights = weights[sort_idx]
            num_above_threshold = len(data[data > threshold])
            # Want to remove a full subsamples worth
            if num_above_threshold > 0:
                rounded = round(num_above_threshold/num_sub_samples)+1
                rounded_num_above_threshold = rounded*num_sub_samples
                return data[:-rounded_num_above_threshold],\
                    weights[:-rounded_num_above_threshold]
            else:
                return data, weights
        else:
            raise ValueError('num_sub_samples must be integer')
