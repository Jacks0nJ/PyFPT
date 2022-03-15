import numpy as np
PI = np.pi
M_PL = 1.0


def reduced_potential_diff(V_dif):
    def v_dif(phi):
        v_dif_value = np.divide(V_dif(phi), 24*(PI**2)*(M_PL**4))
        return v_dif_value
    return v_dif
