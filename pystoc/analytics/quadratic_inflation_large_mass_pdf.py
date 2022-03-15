import numpy as np

from .quadratic_inflation_characteristic_function import\
    quadratic_inflation_characteristic_function
from .continuous_ft import continuous_ft

PI = np.pi
M_PL = 1


def quadratic_inflation_large_mass_pdf(bin_centres, phi_i, phi_end, V):
    def chi(t):
        return quadratic_inflation_characteristic_function(t, phi_i, phi_end,
                                                           V)

    # Use integral symmetric to simplfy to only do the positive half,
    # then double.
    # Remember they use a different fourier 2pi convention to be,
    # hence the extra one.
    v = V(phi_i)/(24*(PI**2)*(M_PL**4))

    # Stolen from Chris' quadratic code, no idea why this is a thing!
    if v < 10:
        t0max = 1000.
    if v < 0.1:
        t0max = 6000.
    if v < 0.04:
        t0max = 10.**7
    PDF_analytical_test =\
        np.array([2*continuous_ft(N, chi, component='real', t_lower=0,
                                  t_upper=t0max)/(2*PI)**0.5 for N in
                  bin_centres])
    return PDF_analytical_test
