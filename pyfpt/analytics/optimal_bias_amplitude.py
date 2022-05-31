'''
Optimal Bias Amplitude
---------------------------------
This module returns a constant (known as the bias amplitude) which when
multiplied by the functional form of the bias used, and in the drift-dominated
limited, results in a duration of inflation equal to the provided target. This
is used to 'tune' how far into the tail of the probability density of the
number of e-folds is investigated by the numerical code, by setting the mean.
'''

from scipy.integrate import quad
from scipy.optimize import root_scalar


def optimal_bias_amplitude(N_target, phi_in, phi_end, potential,
                           potential_diff, bias_function=None,
                           planck_mass=1):
    """Returns bias amplitude for the provided target number of e-folds
    ``N_target`` and bias.

    Parameters
    ----------
    N_target : scalar
        The number of e-folds of interest which are to be investigated- the
        target.
        The potential second derivative.
    phi_in : float
        The initial scalar field value.
        simulated.
    phi_end : float
        The end scalar field value.
    bias_function : function, optional
        The functional form of the bias used. The default is to use the
        diffusion amplitude.
    planck_mass : scalar, optional
        The Planck mass used in the calculations. The standard procedure is to
        set it to 1. The default is 1.
    Returns
    -------
    bias_amplitude : scalar
        the bias amplitude which for the used functional form of the bias, in
        the drift dominated limit gives the target number of e-folds.
    """

    pi = 3.141592653589793
    if bias_function is None:
        def efolds(A):
            def integrand(phi):
                H_squared = potential(phi)/(3*planck_mass**2)
                H = H_squared**0.5
                classical_drift = potential_diff(phi)/(3*H_squared)
                return (classical_drift-A*H/(2*pi))**-1

            integral, _ = quad(integrand, phi_end, phi_in)
            return N_target - integral

        def efolds_derivative(A):
            def integrand(phi):
                H_squared = potential(phi)/(3*planck_mass**2)
                H = H_squared**0.5
                classical_drift = potential_diff(phi)/(3*H_squared)
                return -H/(2*pi)*(classical_drift-A*H/(2*pi))**-2

            integral, _ = quad(integrand, phi_end, phi_in)
            return integral
    # If bias is a function
    elif callable(bias_function):
        def efolds(A):
            def integrand(phi):
                classical_drift = (planck_mass**2)*potential_diff(phi) /\
                    potential(phi)
                return (classical_drift-A*bias_function(phi))**-1

            integral, _ = quad(integrand, phi_end, phi_in)
            return N_target - integral

        def efolds_derivative(A):
            def integrand(phi):
                classical_drift = (planck_mass**2)*potential_diff(phi) /\
                    potential(phi)
                return -bias_function(phi) *\
                    (classical_drift-A*bias_function(phi))**-2

            integral, _ = quad(integrand, phi_end, phi_in)
            return integral
    else:
        raise ValueError('Argument bias_function must be a function')

    sol = root_scalar(efolds, x0=0, fprime=efolds_derivative,
                      method='newton')
    bias_ampltude = sol.root
    return bias_ampltude
