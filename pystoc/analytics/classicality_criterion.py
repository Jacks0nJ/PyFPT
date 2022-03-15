import numpy as np

from .reduced_potential import reduced_potential
from .reduced_potential_diff import reduced_potential_diff
from .reduced_potential_ddiff import reduced_potential_ddiff


# Equation 3.27 in Vennin 2015
def classicality_criterion(V, V_dif, V_ddif, phi_int):
    v_func = reduced_potential(V)
    V_dif_func = reduced_potential_diff(V_dif)
    V_ddif_func = reduced_potential_ddiff(V_ddif)

    v = v_func(phi_int)
    V_dif = V_dif_func(phi_int)
    V_ddif = V_ddif_func(phi_int)

    eta = np.abs(2*v - np.divide(V_ddif*v**2, V_dif**2))
    return eta
