import numpy as np
from scipy import integrate

from .reduced_potential import reduced_potential
from .reduced_potential_diff import reduced_potential_diff
from .reduced_potential_ddiff import reduced_potential_ddiff

M_PL = 1


# Equation 3.35 in Vennin 2015
def variance_N_sto_limit(V, V_dif, V_ddif, phi_int, phi_end):
    v_func = reduced_potential(V)
    V_dif_func = reduced_potential_diff(V_dif)
    V_ddif_func = reduced_potential_ddiff(V_ddif)

    def integrand_calculator(phi):
        # Pre calculating values
        v = v_func(phi)
        V_dif = V_dif_func(phi)
        V_ddif = V_ddif_func(phi)
        non_classical = 6*v-np.divide(5*(v**2)*V_ddif, V_dif**2)
        constant_factor = 2/(M_PL**4)

        integrand = constant_factor*np.divide(v**4, V_dif**3)*(1+non_classical)
        return integrand
    d_N_sq_value, er = integrate.quad(integrand_calculator, phi_end, phi_int)
    return d_N_sq_value
