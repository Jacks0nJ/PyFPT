'''
Quadratic Inflation Multi-Field PDF
-----------------------------------
This module calculates probability density function (PDF) for first-passage
time of the number of e-folds for quadratic inflation for multiple radially
symmetric fields in the large mass case. The large mass case corresponds to
diffusion domination. This is done using the results of `Pattison et al 2017`_
and `Assadullahi et al 2016`_. A UV cutoff at infinity is assumed.

.. _Pattison et al 2017: https://arxiv.org/abs/1707.00537
.. _Assadullahi et al 2016: https://arxiv.org/abs/1604.04502
'''


import numpy as np
from mpmath import hyp1f1, nstr
from scipy import integrate


pi = np.pi
planck_mass = 1


def quadratic_inflation_multi_field_pdf(efolds, v0, n, r_in, r_end=2**0.5,
                                        t0max=None):
    """ Returns PDF of quadratic inflation in the large mass multiple radially
    symmetric field case.

    Parameters
    ----------
    efolds : list or np.ndarray
        The first-passage times where the PDF is to be calculated.
    v0 : float
        The reduced potential value at radial coordinate r=1. Acts like an
        effective mass term, i.e. is the protionality constant between the
        potential and the radial field/coordinate.
    n : int
        The number of fields.
    r_in : float
        The initial field value.
    r_end : float, optional
        The end scalar field value. Defaults to a value such that the first
        slow-roll parameter is 1.
    t0_max: int, optional
        The maximum value for dummy variable t used in the numerical
        integration, as a proxy for infinity.

    Returns
    -------
    pdf : np.ndarray
        The probability density function at the provided e-fold values.

    """
    def potential(r):
        return v0*r**2

    def chi(t):
        return characteristic_function(t, n, r_in, r_end, potential)

    # Use integral symmetric to simplfy to only do the positive half,
    # then double.
    # Remember they use a different fourier 2pi convention to be,
    # hence the extra one.
    v = potential(r_in)

    # Stolen from Chris' quadratic code, no idea why this is a thing!
    if t0max is None:
        if v < 0.04:
            t0max = 10.**7
        elif v <= 0.5:
            t0max = 6000.
        else:
            t0max = 1000.
    pdf = [2*continuous_ft(efolds_value, chi, 0, t0max, component='real') /
           (2*pi)**0.5 for efolds_value in efolds]
    return pdf


def characteristic_function(t, n, r, r_end, potential):
    v_0 = potential(1)
    v = potential(r)
    v_end = potential(r_end)
    alpha = np.sqrt(np.complex((2-n)**2, -np.divide(4*t, v_0)))
    term_1 = (potential(r)/potential(r_end))**(0.25*(2-n) - 0.25*alpha)
    num = hyp1f1(-0.25*(2-n) + 0.25*alpha, 1 + 0.5*alpha, -1/v)
    denom = hyp1f1(-0.25*(2-n) + 0.25*alpha, 1 + 0.5*alpha, -1/v_end)
    chi_mp = (term_1*num)/denom
    chi = np.complex(float(nstr(chi_mp.real, n=12)), float(nstr(chi_mp.imag,
                                                                n=12)))
    return chi


# This is very inefficent, but is accurate. This follows the standard
# conventions, where the forward operation is negative in the exponential.
def continuous_ft(w, func, t_lower, t_upper, component=None):

    def integrand_real(t):
        return (np.exp(np.complex(0, -w*t))*func(t)).real

    def integrand_imaginary(t):
        return (np.exp(np.complex(0, -w*t))*func(t)).imag

    if component == 'real':
        real_component, _ = integrate.quad(integrand_real, t_lower, t_upper,
                                           limit=400)
        return real_component/np.sqrt(2*pi)
    elif component == 'imag':
        img_component, _ = integrate.quad(integrand_imaginary, t_lower,
                                          t_upper, limit=400)
        return -img_component/np.sqrt(2*pi)
    else:
        real_component, _ = integrate.quad(integrand_real, t_lower, t_upper,
                                           limit=400)
        img_component, _ = integrate.quad(integrand_imaginary, t_lower,
                                          t_upper, limit=400)
        return np.complex(real_component, img_component)/np.sqrt(2*pi)
