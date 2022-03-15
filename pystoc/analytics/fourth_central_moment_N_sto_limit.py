import numpy as np
from scipy import integrate

from .reduced_potential import reduced_potential
from .reduced_potential_diff import reduced_potential_diff
from .variance_N_sto_limit import variance_N_sto_limit

M_PL = 1


# This is done using Vincent's calculations he gave me
def fourth_central_moment_N_sto_limit(V, V_dif, V_ddif, phi_int, phi_end):

    v_func = reduced_potential(V)
    V_dif_func = reduced_potential_diff(V_dif)

    def integrand_calculator(phi):
        # Pre calculating values
        v = v_func(phi)
        V_dif = V_dif_func(phi)
        constant_factor = 120/(M_PL**8)

        integrand = constant_factor*np.divide(v**10, V_dif**7)
        return integrand
    non_guassian, er = integrate.quad(integrand_calculator, phi_end, phi_int)
    # As Vincent's method explicitly calculates the excess kurtosis, need to
    # add Wick's theorem term
    gaussian_4th_moment =\
        3*variance_N_sto_limit(V, V_dif, V_ddif, phi_int, phi_end)**2

    return gaussian_4th_moment+non_guassian
