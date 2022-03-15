import numpy as np
from mpmath import hyp1f1, nstr

PI = np.pi
M_PL = 1


def quadratic_inflation_characteristic_function(t, phi, phi_end, V):
    v_0 = V(M_PL)/(24*(M_PL**4)*(PI**2))
    v = V(phi)/(24*(M_PL**4)*(PI**2))
    v_end = V(phi_end)/(24*(M_PL**4)*(PI**2))
    alpha = np.sqrt(np.complex(1, -np.divide(4*t, v_0)))
    term_1 = (V(phi)/V(phi_end))**(0.25 - 0.25*alpha)
    num = hyp1f1(-0.25 + 0.25*alpha, 1 + 0.5*alpha, -1/v)
    denom = hyp1f1(-0.25 + 0.25*alpha, 1 + 0.5*alpha, -1/v_end)
    chi_mp = (term_1*num)/denom
    chi = np.complex(float(nstr(chi_mp.real, n=12)), float(nstr(chi_mp.imag,
                                                                n=12)))
    return chi
