import numpy as np
from scipy import integrate

PI = np.pi


# This is very inefficent, but is accurate. This follows the standard
# conventions, where the forward operation is negative in the exponential.
# THIS WILL NOT WORK FOR COMPLEX EXPONENTIAL!
def continuous_ft(w, func, component=None, t_lower=-np.inf, t_upper=np.inf):
    def integrand_real(t):
        return (np.exp(np.complex(0, -w*t))*func(t)).real

    def integrand_imaginary(t):
        return (np.exp(np.complex(0, -w*t))*func(t)).imag

    if component == 'real':
        real_component, _ = integrate.quad(integrand_real, t_lower, t_upper,
                                           limit=400)
        return real_component/np.sqrt(2*PI)
    elif component == 'imag':
        img_component, _ = integrate.quad(integrand_imaginary, t_lower,
                                          t_upper, limit=400)
        return -img_component/np.sqrt(2*PI)
    else:
        real_component, _ = integrate.quad(integrand_real, t_lower, t_upper,
                                           limit=400)
        img_component, _ = integrate.quad(integrand_imaginary, t_lower,
                                          t_upper, limit=400)
        return np.complex(real_component, img_component)/np.sqrt(2*PI)
