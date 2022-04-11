from numpy import diff


# Returns the normalisation factor for a histogram, including one with weights
def histogram_analytical_normalisation(bins, num_sims):
    return num_sims*diff(bins)
