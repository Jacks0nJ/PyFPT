import numpy as np

from .He3 import He3
from .He4 import He4
from .He6 import He6

PI = np.pi


# IMPORTANT - I've added a new term to this series
# Using the Gram–Charlier A series to approximate an arbitary pdf with a
# Gaussian distribution, plus two higher order terms from the 3rd and 4th
# cumulants. There are issues with convergence and error,
# see https://en.wikipedia.org/wiki/Edgeworth_series
def edgeworth_pdf(x, mean, std, skewness, kurtosis):
    norm_x = (x-mean)/std

    skew_term = np.divide(skewness*He3(norm_x), 6)
    kurtosis_term = np.divide(kurtosis*He4(norm_x), 24)
    skew_squared_term = np.divide(He6(norm_x)*skewness**2, 72)

    gaussian = np.divide(np.exp(-0.5*norm_x**2), std*(2*PI)**0.5)

    return gaussian*(1+skew_term+kurtosis_term+skew_squared_term)
