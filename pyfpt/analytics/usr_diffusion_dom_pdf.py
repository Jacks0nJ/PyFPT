'''
USR Inflation Diffusion Domination PDF
--------------------------------------
This module returns a function which calculates the  probability density
function (PDF) for first-passage time of the number of e-folds for
ulta-slow-roll (USR) inflation in a flat potential for diffusion domination.
This is done using the results of `Pattison et al 2022`_, where the notation
used is also defined.

.. _Pattison et al 2022: https://arxiv.org/abs/2101.05741
'''

import numpy as np


planck_mass = 1

pi = np.pi


def usr_diffusion_dom_pdf(x, y, mu, n=1000):
    """ Returns PDF of USR inflation in a flat potential for diffusion
    domination.

    Parameters
    ----------
    x : float
        The rescaled and dimensionaless field value. Must be equal to or
        smaller than 1.
    y: float
        The rescaled and dimensionaless field velocity. Diffusion domination is
        only true when this is less than 1.
    mu: float
        The dimensionaless effective flat potential width.
    n: int, optional
        The number of terms used in Eq. (4.23). Defaults to 100
    Returns
    -------
    usr_pdf_function : function
        The probability density function at the provided e-fold values, i.e.
        a function of ``(N)``.

    """
    # Check if the user has input correct values
    if isinstance(x, float) is True or isinstance(x, int) is True:
        if x > 1.:
            raise ValueError("Field value x must be equal to or less than 1")
    else:
        if any(x > 1.):
            raise ValueError("Field value x must be equal to or less than 1")
    if isinstance(y, float) is True or isinstance(y, int) is True:
        if y > 1.:
            print("WARNING: this approximation is only valid for y<1")
    else:
        if any(y > 1.):
            print("WARNING: this approximation is only valid for y<1")
    # Needs to a complex array as omega can be complex
    n_array = np.arange(n, dtype=complex)

    poles_LO = leading_poles(mu, n_array)
    residuals_LO = leading_residue(x, y, mu, n_array)
    residuals_NLO = next_to_leading_residue(x, y, mu, n_array)

    def usr_pdf_function(N):
        pdf = 0.
        for i in range(n):
            pdf += (residuals_LO[i] +
                    residuals_NLO[i]*np.exp(-3*N))*np.exp(-poles_LO[i]*N)
        return pdf.real

    return usr_pdf_function


# Eq. 4.18 of arXiv:2101.05741v1
def leading_poles(mu, n):
    return (pi*(n+0.5)/mu)**2


# Eq. 4.19 of arXiv:2101.05741v1
def leading_residue(x, y, mu, n):
    n_plus_half = n+0.5
    omega = np.sqrt((n_plus_half*pi)**2-3*mu**2)

    overal_factor = np.divide(2*n_plus_half*pi, mu**2)

    sin_term = np.sin(n_plus_half*pi*x)
    cos_term = -y*n_plus_half*pi*np.cos(n_plus_half*pi*x)

    curly_brackets = omega*np.cos(omega*(1 - x)) -\
        ((-1)**n)*n_plus_half*pi*np.sin(omega*x)
    curly_bracket_factor = y*np.divide(n_plus_half*pi, omega*np.cos(omega))

    return overal_factor*(sin_term + cos_term +
                          curly_bracket_factor*curly_brackets)


# Eq. 4.22 of arXiv:2101.05741v1
def next_to_leading_residue(x, y, mu, n):
    n_plus_half = n+0.5
    omega = np.sqrt((n_plus_half*pi)**2+3*mu**2)

    overal_factor = np.divide(np.sin(n_plus_half*pi*x)*2*y*(-1)**n,
                              np.cos(omega)*mu**2)

    return overal_factor*(-omega**2 +
                          pi*n_plus_half*omega*np.sin(omega)*(-1)**n)
