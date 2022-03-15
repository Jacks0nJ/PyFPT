import numpy as np
from scipy import integrate

from .reduced_potential import reduced_potential
from .reduced_potential_diff import reduced_potential_diff
from .reduced_potential_ddiff import reduced_potential_ddiff

M_PL = 1


# Equation 3.28 in Vennin 2015
def mean_N_sto_limit(V, V_dif, V_ddif, phi_int, phi_end):
    v_func = reduced_potential(V)
    V_dif_func = reduced_potential_diff(V_dif)
    V_ddif_func = reduced_potential_ddiff(V_ddif)

    def integrand_calculator(phi):
        # Pre calculating values
        v = v_func(phi)
        V_dif = V_dif_func(phi)
        V_ddif = V_ddif_func(phi)
        non_classical = v-np.divide((v**2)*V_ddif, V_dif**2)
        constant_factor = 1/(M_PL**2)

        integrand = constant_factor*np.divide(v, V_dif)*(1+non_classical)
        return integrand

    mean_N, er = integrate.quad(integrand_calculator, phi_end, phi_int)

    return mean_N
