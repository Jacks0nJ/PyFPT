import numpy as np


# This function tests if a multiprocessing error has occured. This is when the
# data from the different cores becomes mixed, and the weights and N are not
# correct
def multi_processing_error(sim_N_dist, w_values):
    # Checking if multipprocessing error occured, by looking at correlation
    pearson_corr = np.corrcoef(sim_N_dist, np.log10(w_values))
    pearson_corr = pearson_corr[0, 1]

    if pearson_corr > -0.55:  # Data is uncorrelated
        print('Possible multiprocessing error occured')
