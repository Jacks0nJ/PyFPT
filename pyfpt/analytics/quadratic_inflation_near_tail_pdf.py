'''
Quadratic Inflation Near Tail PDF
----------------------------------
This module calculates the near tail of the probability density function (PDF)
for first-passage time of the number of e-folds for quadratic inflation in the
large mass case. The large mass case corresponds to diffusion domination. This
is done using the results of appendix in `<insert our paper>`_, and therefore
assumes UV cutoff at infinity.

.. _<insert our paper>: https://arxiv.org/abs/1707.00537
'''

import numpy as np
from scipy import integrate

PI = np.pi
M_PL = 1


def quadratic_inflation_near_tail_pdf(N, m, phi_i, phi_end=2**0.5,
                                      numerical_integration=False):
    """ Returns PDF of quadratic inflation for the near tail.

    Parameters
    ----------
    N : list
        The first-passage times where the PDF is to be calculated.
    V : function
        The potential.
    phi_i : float
        The initial field value.
    phi_end : float, optional
        The end scalar field value. Defaults to value such that the first
        slow-roll parameter is 1.
    numerical_integration : bool, optional
        If numerical integration is used.

    Returns
    -------
    pdf : list
        The probability density function at N values.

    """
    v0 = (m**2)/(48*PI**2)
    v = v0*phi_i**2
    N_cl = 0.25*phi_i**2-0.25*phi_end**2
    ve = v0*phi_end**2
    if numerical_integration is False:

        # Calculating the terms individually for clarity
        constant = (np.sqrt(2)*PI**2)/(128*v0**2)
        exp = np.exp(-0.25*v0*N)

        frac_expo_i = np.divide(PI**2, 16*v0*(N+N_cl+1))
        fraction_i = np.divide(np.exp(frac_expo_i-1/v)*v**1.5, (N+N_cl+1)**3)

        frac_expo_end = np.divide(PI**2, 16*v0*(N-N_cl+1))
        fraction_end = np.divide(np.exp(frac_expo_end-1/ve)*ve**1.5,
                                 (N-N_cl+1)**3)

        pdf = constant*exp*(fraction_i - fraction_end)

    elif numerical_integration is True:
        a1 = 0.25*v0*N + 0.0625*(v+ve)
        a2 = 0.25*v0*N + 0.0625*(3*ve-v)

        def G(x, a):
            return np.exp(0.25*PI*x-a*x**2)*x**(5/2)

        Ga1_int, _ = integrate.quad(G, 3, np.infty, args=(a1))
        Ga2_int, _ = integrate.quad(G, 3, np.infty, args=(a2))

        first_term = np.exp(-1/v)*Ga1_int*v**1.5
        second_term = np.exp(-1/ve)*Ga2_int*ve**1.5
        pdf = v0*np.exp(-0.25*v0*N)*(first_term-second_term)/(32*PI)
    return pdf
