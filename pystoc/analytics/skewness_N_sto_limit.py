import numpy as np
from scipy import integrate

from .reduced_potential import reduced_potential
from .reduced_potential_diff import reduced_potential_diff
from .reduced_potential_ddiff import reduced_potential_ddiff
from .variance_N_sto_limit import variance_N_sto_limit

M_PL = 1


# Equation 3.37 in Vennin 2015, then divded by sigma^3 to make skewness
def skewness_N_sto_limit(V, V_dif, V_ddif, phi_int, phi_end):
    v_func = reduced_potential(V)
    V_dif_func = reduced_potential_diff(V_dif)
    V_ddif_func = reduced_potential_ddiff(V_ddif)

    def integrand_calculator(phi):
        # Pre calculating values
        v = v_func(phi)
        V_dif = V_dif_func(phi)
        V_ddif = V_ddif_func(phi)
        non_classical = 14*v-np.divide(11*(v**2)*V_ddif, V_dif**2)
        constant_factor = 12/(M_PL**6)

        integrand = constant_factor*np.divide(v**7, V_dif**5)*(1+non_classical)
        return integrand
    skewness_value, er = integrate.quad(integrand_calculator, phi_end, phi_int)
    # Now normalise by the variance
    skewness_value = skewness_value/variance_N_sto_limit(V, V_dif, V_ddif,
                                                         phi_int, phi_end)**1.5
    return skewness_value
