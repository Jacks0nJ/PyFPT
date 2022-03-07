
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 09:52:31 2021
    This is to be a repository of standard functions used in inflationary
    cosmology, such as scalar field density and slow roll parameters. This uses
    N as the time varible.

    It base takes H as an argument, but the 'ind' suffix means that the
    function can directly calculate H within the enviroment. This allows H to
    be used as a constraint equation.
@author: user
"""

import numpy as np
from scipy import integrate
from scipy import optimize
from mpmath import hyp1f1, nstr
# import scipy as sp
# import math
# M_PL = 2.435363*10**18 old value
M_PL = 1.0
PI = np.pi  # so can just use PI as needed
EULER_GAMMA = 0.5772156649

'''
Functions from Vennin 2015
'''


def reduced_potential(V):
    def v(phi):
        v_value = np.divide(V(phi), 24*(PI**2)*(M_PL**4))
        return v_value
    return v


def reduced_potential_diff(V_dif):
    def v_dif(phi):
        v_dif_value = np.divide(V_dif(phi), 24*(PI**2)*(M_PL**4))
        return v_dif_value
    return v_dif


def reduced_potential_ddiff(V_ddif):
    def v_ddif(phi):
        v_ddif_value = np.divide(V_ddif(phi), 24*(PI**2)*(M_PL**4))
        return v_ddif_value
    return v_ddif


# Equation 3.27 in Vennin 2015
def classicality_criterion(V, V_dif, V_ddif, phi_int):
    v_func = reduced_potential(V)
    V_dif_func = reduced_potential_diff(V_dif)
    V_ddif_func = reduced_potential_ddiff(V_ddif)

    v = v_func(phi_int)
    V_dif = V_dif_func(phi_int)
    V_ddif = V_ddif_func(phi_int)

    eta = np.abs(2*v - np.divide(V_ddif*v**2, V_dif**2))
    return eta


# Equation 3.28 in Vennin 2015
def mean_N_sto_limit(V, V_dif, V_ddif, phi_int, phi_end):
    v_func = reduced_potential(V)
    V_dif_func = reduced_potential_diff(V_dif)
    V_ddif_func = reduced_potential_ddiff(V_ddif)

    def integrand_calculator(phi):
        # Pre calculating values
        v = v_func(phi)
        V_dif = V_dif_func(phi)
        V_ddif = V_ddif_func(phi)
        non_classical = v-np.divide((v**2)*V_ddif, V_dif**2)
        constant_factor = 1/(M_PL**2)

        integrand = constant_factor*np.divide(v, V_dif)*(1+non_classical)
        return integrand

    mean_N, er = integrate.quad(integrand_calculator, phi_end, phi_int)

    return mean_N


# Equation 3.35 in Vennin 2015
def variance_N_sto_limit(V, V_dif, V_ddif, phi_int, phi_end):
    v_func = reduced_potential(V)
    V_dif_func = reduced_potential_diff(V_dif)
    V_ddif_func = reduced_potential_ddiff(V_ddif)

    def integrand_calculator(phi):
        # Pre calculating values
        v = v_func(phi)
        V_dif = V_dif_func(phi)
        V_ddif = V_ddif_func(phi)
        non_classical = 6*v-np.divide(5*(v**2)*V_ddif, V_dif**2)
        constant_factor = 2/(M_PL**4)

        integrand = constant_factor*np.divide(v**4, V_dif**3)*(1+non_classical)
        return integrand
    d_N_sq_value, er = integrate.quad(integrand_calculator, phi_end, phi_int)
    return d_N_sq_value


# Equation 3.37 in Vennin 2015
def third_central_moment_N_sto_limit(V, V_dif, V_ddif, phi_int, phi_end):
    v_func = reduced_potential(V)
    V_dif_func = reduced_potential_diff(V_dif)
    V_ddif_func = reduced_potential_ddiff(V_ddif)

    def integrand_calculator(phi):
        # Pre calculating values
        v = v_func(phi)
        V_dif = V_dif_func(phi)
        V_ddif = V_ddif_func(phi)
        non_classical = 14*v-np.divide(11*(v**2)*V_ddif, V_dif**2)
        constant_factor = 12/(M_PL**6)

        integrand = constant_factor*np.divide(v**7, V_dif**5)*(1+non_classical)
        return integrand
    third_moment, er = integrate.quad(integrand_calculator, phi_end, phi_int)
    return third_moment


# This is done using Vincent's calculations he gave me
def fourth_central_moment_N_sto_limit(V, V_dif, V_ddif, phi_int, phi_end):

    v_func = reduced_potential(V)
    V_dif_func = reduced_potential_diff(V_dif)

    def integrand_calculator(phi):
        # Pre calculating values
        v = v_func(phi)
        V_dif = V_dif_func(phi)
        constant_factor = 120/(M_PL**8)

        integrand = constant_factor*np.divide(v**10, V_dif**7)
        return integrand
    non_guassian, er = integrate.quad(integrand_calculator, phi_end, phi_int)
    # As Vincent's method explicitly calculates the excess kurtosis, need to
    # add Wick's theorem term
    gaussian_4th_moment =\
        3*variance_N_sto_limit(V, V_dif, V_ddif, phi_int, phi_end)**2

    return gaussian_4th_moment+non_guassian


# Equation 3.37 in Vennin 2015, then divded by sigma^3 to make skewness
def skewness_N_sto_limit(V, V_dif, V_ddif, phi_int, phi_end):
    v_func = reduced_potential(V)
    V_dif_func = reduced_potential_diff(V_dif)
    V_ddif_func = reduced_potential_ddiff(V_ddif)

    def integrand_calculator(phi):
        # Pre calculating values
        v = v_func(phi)
        V_dif = V_dif_func(phi)
        V_ddif = V_ddif_func(phi)
        non_classical = 14*v-np.divide(11*(v**2)*V_ddif, V_dif**2)
        constant_factor = 12/(M_PL**6)

        integrand = constant_factor*np.divide(v**7, V_dif**5)*(1+non_classical)
        return integrand
    skewness_value, er = integrate.quad(integrand_calculator, phi_end, phi_int)
    # Now normalise by the variance
    skewness_value = skewness_value/variance_N_sto_limit(V, V_dif, V_ddif,
                                                         phi_int, phi_end)**1.5
    return skewness_value


# Using the standard relation between the central moments and the kurtosis.
# Fisher is an optional argument
def kurtosis_N_sto_limit(V, V_dif, V_ddif, phi_int, phi_end, Fisher=True):
    # The excess kurtosis over the expected Gaussian amount
    fourth_moment = fourth_central_moment_N_sto_limit(V, V_dif, V_ddif,
                                                      phi_int, phi_end)
    var = variance_N_sto_limit(V, V_dif, V_ddif, phi_int, phi_end)
    if Fisher is True:
        return fourth_moment/var**2-3
    else:
        return fourth_moment/var**2


def skewness_N_chaotic_inflation(m, phi_i):
    skew = np.divide((m**4)*(phi_i**10), 61440*(PI**4)*(M_PL**14))
    return skew


'''
Analytical Results for Stochastic m^2*phi^2 inflation
'''


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


def chaotic_inflation_characteristic_function(t, phi, phi_end, V):
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


def large_mass_pdf(bin_centres, phi_i, phi_end, V):
    def chi(t):
        return chaotic_inflation_characteristic_function(t, phi_i, phi_end,
                                                         V)

    # Use integral symmetric to simplfy to only do the positive half,
    # then double.
    # Remember they use a different fourier 2pi convention to be,
    # hence the extra one.
    v = V(phi_i)/(24*(PI**2)*(M_PL**4))

    # Stolen from Chris' quadratic code, no idea why this is a thing!
    if v < 10:
        t0max = 1000.
    if v < 0.1:
        t0max = 6000.
    if v < 0.04:
        t0max = 10.**7
    PDF_analytical_test =\
        np.array([2*continuous_ft(N, chi, component='real', t_lower=0,
                                  t_upper=t0max)/(2*PI)**0.5 for N in
                  bin_centres])
    return PDF_analytical_test


'''
Edgeworth
'''


def He3(y):
    return y**3-3*y


def He4(y):
    return y**4-6*y+3


def He6(y):
    return y**6-15*y**4+45*y**2-15


# IMPORTANT - I've added a new term to this series
# Using the Gram–Charlier A series to approximate an arbitary pdf with a
# Gaussian distribution, plus two higher order terms from the 3rd and 4th
# cumulants. There are issues with convergence and error,
# see https://en.wikipedia.org/wiki/Edgeworth_series
def edgeworth_pdf(x, mean, std, skewness, kurtosis):
    norm_x = (x-mean)/std

    skew_term = np.divide(skewness*He3(norm_x), 6)
    kurtosis_term = np.divide(kurtosis*He4(norm_x), 24)
    skew_squared_term = np.divide(He6(norm_x)*skewness**2, 72)

    gaussian = np.divide(np.exp(-0.5*norm_x**2), std*(2*PI)**0.5)

    return gaussian*(1+skew_term+kurtosis_term+skew_squared_term)


# This returns a function which returns the Edgeworth expansion
def edgeworth_pdf_sto_limit(V, V_dif, V_ddif, phi_i, phi_end):

    mean = mean_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)
    std = variance_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)**0.5
    skewness = skewness_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)
    kurtosis = kurtosis_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)

    def edgeworth_function(N):
        return edgeworth_pdf(N, mean, std, skewness, kurtosis)

    return edgeworth_function


# Using the Gram–Charlier A series
# https://en.wikipedia.org/wiki/Edgeworth_series to approximate when we expect
# classical deviation from a gaussian. This is done by finding x such that the
# higher order terms of the edgeworth expanion are
# nu is the amount pf deviation from a Gaussian.
def gaussian_deviation(mean, std, third_cumulant, fourth_cumulant, nu=1,
                       x_interval=None):

    def higher_order_egdeworth_term(y):
        norm_y = (y-mean)/std
        skew_term = np.divide(third_cumulant*He3(norm_y), 6*std**3)
        kurtosis_term = np.divide(fourth_cumulant*He4(norm_y), 24*std**4)
        skew_squared_term =\
            np.divide(He6(norm_y)*third_cumulant**2, 72*std**6)
        return (skew_term+kurtosis_term+skew_squared_term)-nu

    if x_interval is None:
        sol = optimize.root_scalar(higher_order_egdeworth_term,
                                   method='brentq',
                                   bracket=[mean, mean+10000*std])
    else:
        sol = optimize.root_scalar(higher_order_egdeworth_term,
                                   method='brentq', bracket=x_interval)

    return sol.root  # The root is the position of when deviation occurs


# Using the Gram–Charlier A series
# https://en.wikipedia.org/wiki/Edgeworth_series to approximate when we expect
# classical deviation from a gaussian. This is done by finding x such that the
# higher order terms of the edgeworth expanion are
# nu is the amount pf deviation from a Gaussian.
def gaussian_deviation_sto_limit(V, V_dif, V_ddif, phi_i, phi_end, nu=1,
                                 x_interval=None):

    mean = mean_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)
    std = variance_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)**0.5
    skewness = skewness_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)
    kurtosis = kurtosis_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)

    def higher_order_egdeworth_term(y):
        norm_y = (y-mean)/std
        skew_term = np.divide(skewness*He3(norm_y), 6)
        kurtosis_term = np.divide(kurtosis*He4(norm_y), 24)
        skew_squared_term =\
            np.divide(He6(norm_y)*skewness**2, 72)
        return (skew_term+kurtosis_term+skew_squared_term)-nu

    if x_interval is None:
        sol = optimize.root_scalar(higher_order_egdeworth_term,
                                   method='brentq',
                                   bracket=[mean, mean+10000*std])
    else:
        sol = optimize.root_scalar(higher_order_egdeworth_term,
                                   method='brentq', bracket=x_interval)

    return sol.root  # The root is the position of when deviation occurs
