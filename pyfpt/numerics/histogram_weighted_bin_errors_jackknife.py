import numpy as np

from .histogram_normalisation import histogram_normalisation


# Method for calculating the errors of the histogram bars by using
# the simplified jackknife analysis. The data is sub-sampled into many
# histograms with the same bins. This way a distribution for the different
# heights can be done. Takes the bins used as general argument
# Arguments must be numpy arrays
def histogram_weighted_bin_errors_jackknife(data_input, weights_input, bins,
                                            num_sub_samps):
    # Make an array of random indexs
    indx = np.arange(0, len(data_input), 1)
    np.random.shuffle(indx)
    # This allows the data to be randomised and keep the weights matched
    data = data_input[indx]
    weights = weights_input[indx]
    num_bins = len(bins)-1  # bins is an array of the side, so one less

    height_array = np.zeros((num_bins, num_sub_samps))  # Storage

    # Next organise into subsamples
    data =\
        np.reshape(data, (int(data.shape[0]/num_sub_samps), num_sub_samps))
    weights =\
        np.reshape(weights, (int(weights.shape[0]/num_sub_samps),
                             num_sub_samps))

    # Find the heights of the histograms, for each sample
    for i in range(num_sub_samps):
        heights_raw, _ =\
            np.histogram(data[:, i], bins, weights=weights[:, i])
        norm = histogram_normalisation(bins, len(data[:, i]))
        height_array[:, i] = heights_raw/norm

    error = np.zeros(num_bins)
    sqrt_sub_samples = np.sqrt(num_sub_samps)

    # Now can find the standard deviation of the heights, as the bins are the
    # same. Then divide by the sqaure root of the number of samples by
    # jackknife and you have the error
    for j in range(num_bins):
        bars = height_array[j, :]
        if np.any([bars > 0]):
            # Used to be just np.std(bars)
            error[j] = np.std(bars[bars > 0])/sqrt_sub_samples
        else:
            error[j] = 0  # Remove any empty histogram bars

    return error
