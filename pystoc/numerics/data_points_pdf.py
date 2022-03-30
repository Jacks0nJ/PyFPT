import matplotlib.pyplot as plt
import numpy as np


from .histogram_analytical_normalisation import\
    histogram_analytical_normalisation
from .histogram_data_in_bins import histogram_data_in_bins
from .histogram_weighted_bin_errors_jackknife import\
    histogram_weighted_bin_errors_jackknife
from .log_normal_height import log_normal_height
from .log_normal_errors import log_normal_errors
from .lognormality_check import lognormality_check


def data_points_pdf(Ns, ws, num_sub_samples, reconstruction,
                    min_bin_size=None, bins=50, num_sims=None):
    # If no number of simulations argument is passed.
    if isinstance(num_sims, int) is not True:
        num_sims = len(Ns)

    # If the number of bins used has been specified
    if isinstance(bins, int) is True:
        num_bins = bins
        # Want raw heights of histogram bars
        heights_raw, bins, _ =\
            plt.hist(Ns, num_bins, weights=ws)
        plt.clf()
    # If the bins have been specified
    else:
        num_bins = len(bins)-1  # as bins is the bin edges, so plus 1
        # Want raw heights of histogram bars
        heights_raw, bins, _ =\
            plt.hist(Ns, bins=bins, weights=ws)
        plt.clf()

    analytical_norm =\
        histogram_analytical_normalisation(bins, num_sims)

    data_in_bins, weights_in_bins =\
        histogram_data_in_bins(Ns, ws, bins)

    # Predictions need the bin centre to make comparison
    bin_centres = np.array([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)])

    # Removing underfilled bins if needed
    if isinstance(min_bin_size, int) is True:
        # Then loop through to find where length is greater than min_bin_size
        filled_bins = []
        num_sims_used = len(Ns)
        for i in range(len(bins)-1):
            data_in_bin = data_in_bins[:, i]
            data_in_bin = data_in_bin[data_in_bin > 0]
            # First, remove if empty
            if len(data_in_bin) == 0:
                filled_bins.append(False)
            # If there is enough data in this bin
            elif len(data_in_bin) >= min_bin_size:
                filled_bins.append(True)
            # Don't include under filled tail bins
            else:
                filled_bins.append(False)
                # Reflect in number of succesful simulatios
                num_sims_used -= len(data_in_bin)
        bin_centres_uncut = bin_centres
        bin_centres = bin_centres[filled_bins]

    if reconstruction == 'naive':
        heights = heights_raw/analytical_norm
        errors = histogram_weighted_bin_errors_jackknife(Ns, ws, bins,
                                                         num_sub_samples)
        if isinstance(min_bin_size, int) is True:
            heights = heights[filled_bins]
            errors = errors[filled_bins]
    elif reconstruction == 'lognormal':

        heights_est = np.zeros(num_bins)
        # The errors for the log-normal case are asymmetric
        errors_est = np.zeros((2, num_bins))
        for i in range(num_bins):
            w = weights_in_bins[:, i]
            # Only calculate filled bins
            if filled_bins[i] is True or\
                (np.any([w > 0]) is True and isinstance(min_bin_size, int)
                 is False):
                w = w[w > 0]
                heights_est[i] =\
                    log_normal_height(w, position=bin_centres_uncut[i])
                errors_est[:, i] = log_normal_errors(w)

        # Include only filled values
        # Remember to normalise errors as well
        heights = heights_est[errors_est[0, :] > 0]/analytical_norm
        # The errors are a 2D array, so need to slice correctly
        errors = errors_est[:, errors_est[0, :] > 0]/analytical_norm

        # Checking p-values if lognormal was used
        lognormality_check(bin_centres, weights_in_bins, filled_bins,
                           num_bins)

    else:
        raise ValueError('Not valid reconstrcution method')

    return bin_centres, heights, errors, num_sims_used, bins
