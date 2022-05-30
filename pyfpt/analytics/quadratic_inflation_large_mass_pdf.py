'''
Quadratic Inflation Large Mass PDF
----------------------------------
This module calculates probability density function (PDF) for first-passage
time of the number of e-folds for quadratic inflation in the large mass case.
The large mass case corresponds to diffusion domination. This is done using the
results of `Pattison et al 2017`_, and therefore assumes a UV cutoff at
infinity.

.. _Pattison et al 2017: https://arxiv.org/abs/1707.00537
'''


import numpy as np
from mpmath import hyp1f1, nstr
from scipy import integrate


pi = np.pi
planck_mass = 1


def quadratic_inflation_large_mass_pdf(efolds, m, phi_in, phi_end=2**0.5):
    """ Returns PDF of quadratic inflation in the large mass case.

    Parameters
    ----------
    efolds : list
        The first-passage times where the PDF is to be calculated.
    m : float
        The mass of quadratic inflation potential.
    phi_in : float
        The initial field value.
    phi_end : float, optional
        The end scalar field value. Defaults to a value such that the first
        slow-roll parameter is 1.

    Returns
    -------
    pdf : list
        The probability density function at the provided efold values.

    """
    def potential(phi):
        return 0.5*(m*phi)**2

    def chi(t):
        return quadratic_inflation_characteristic_function(t, phi_in, phi_end,
                                                           potential)

    # Use integral symmetric to simplfy to only do the positive half,
    # then double.
    # Remember they use a different fourier 2pi convention to be,
    # hence the extra one.
    v0 = (m**2)/(48*pi**2)
    v = v0*phi_in**2

    # Stolen from Chris' quadratic code, no idea why this is a thing!
    if v < 10:
        t0max = 1000.
    if v < 0.1:
        t0max = 6000.
    if v < 0.04:
        t0max = 10.**7
    pdf = [2*continuous_ft(efolds_value, chi, 0, t0max, component='real') /
           (2*pi)**0.5 for efolds_value in efolds]
    return pdf


def quadratic_inflation_characteristic_function(t, phi, phi_end, potential):
    v_0 = potential(planck_mass)/(24*(planck_mass**4)*(pi**2))
    v = potential(phi)/(24*(planck_mass**4)*(pi**2))
    v_end = potential(phi_end)/(24*(planck_mass**4)*(pi**2))
    alpha = np.sqrt(np.complex(1, -np.divide(4*t, v_0)))
    term_1 = (potential(phi)/potential(phi_end))**(0.25 - 0.25*alpha)
    num = hyp1f1(-0.25 + 0.25*alpha, 1 + 0.5*alpha, -1/v)
    denom = hyp1f1(-0.25 + 0.25*alpha, 1 + 0.5*alpha, -1/v_end)
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
