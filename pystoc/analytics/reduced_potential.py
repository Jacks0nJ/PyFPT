import numpy as np
PI = np.pi
M_PL = 1.0


def reduced_potential(V):
    def v(phi):
        v_value = np.divide(V(phi), 24*(PI**2)*(M_PL**4))
        return v_value
    return v
