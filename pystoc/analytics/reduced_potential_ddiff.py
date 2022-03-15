import numpy as np
PI = np.pi
M_PL = 1.0


def reduced_potential_ddiff(V_ddif):
    def v_ddif(phi):
        v_ddif_value = np.divide(V_ddif(phi), 24*(PI**2)*(M_PL**4))
        return v_ddif_value
    return v_ddif
