
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
from scipy import stats
from scipy import special
from scipy import optimize
import math
from mpmath import hyp1f1, nstr
#import scipy as sp
#import math
#M_PL = 2.435363*10**18 old value
M_PL = 1.0
PI = np.pi#so can just use PI as needed
EULER_GAMMA = 0.5772156649
##
##
#These assumees the number of e-foldings N as the time varible
##
##

def per_diff(x_correct,x):
    diff = 100*np.divide(np.abs(x_correct-x),x_correct)
    return diff

def hubble_param(V,phi,phi_dN):
    _non_sr_contribution = (1-((phi_dN/M_PL)**2)/6)**(-1)
    H_squared = np.divide(V(phi),3*M_PL**2)*_non_sr_contribution    
    if np.amin(H_squared) > 0:
        H = np.sqrt(H_squared)
    else:
        print(phi_dN)
        raise ValueError('Non positive Hubble parameter') 
    return H

def hubble_param_dN_ind(V,phi,phi_dN):
    H = hubble_param(V,phi,phi_dN)
    H_dN = -(H/2)*(phi_dN/M_PL)**2
    return H_dN

#This is mainly used for propagating in H as a variable
def hubble_param_dN(phi_dN, H):
    H_dN = -np.divide(H*phi_dN**2,2*M_PL**2)
    return H_dN

#This is the acceleration term of the KG equation with N as time
def phi_accel_with_N(V_p,phi,phi_dN,H):
    _accel = -phi_dN*(3-0.5*(phi_dN/M_PL)**2)-\
                np.divide(V_p(phi),H**2)
    return _accel

def hubble_param_sr(V,phi):
    H_squared = np.divide(V(phi), 3*M_PL**2)
    H = np.sqrt(H_squared)
    return H

def phi_dN_sr(V_p, phi, H):
    phi_dN = -np.divide(V_p(phi),3*H**2)
    return phi_dN

def phi_dN_sr_ind(V, V_p, phi):
    H = hubble_param_sr(V,phi)
    phi_dN = -np.divide(V_p(phi),3*H**2)
    return phi_dN


def density_ind(V,phi,phi_dN):
    H = hubble_param(V,phi,phi_dN)
    rho = ((H*phi_dN)**2)/2 + V(phi)
    return rho

def density(V,phi,phi_dN,H):
    rho = ((H*phi_dN)**2)/2 + V(phi)
    return rho

def pressure_ind(V,phi,phi_dN):
    H = hubble_param(V,phi,phi_dN)
    p = ((H*phi_dN)**2)/2 - V(phi)
    return p

def pressure(V,phi,phi_dN,H):
    p = ((H*phi_dN)**2)/2 - V(phi)
    return p

def eos_param_ind(V,phi,phi_dN):
    _p = pressure_ind(V,phi,phi_dN)
    _rho = density_ind(V,phi,phi_dN)
    _w = np.divide(_p,_rho)
    return _w

def eos_param(V,phi,phi_dN,H):
    _p = pressure(V,phi,phi_dN,H)
    _rho = density(V,phi,phi_dN,H)
    _w = np.divide(_p,_rho)
    return _w

def hubble_flow_param_one(phi_dN):
    _epsilon_1 = (0.5)*(phi_dN/M_PL)**2
    return _epsilon_1

def hubble_flow_param_two_ind(V,V_p,phi,phi_dN):
    epsilon_1 = hubble_flow_param_one(V,phi,phi_dN)
    f = f_param_ind(V,V_p,phi,phi_dN)
    _epsilon_2 = 2*epsilon_1 - 6*f
    return _epsilon_2

def hubble_flow_param_two(V,V_p,phi,phi_dN,H):
    epsilon_1 = hubble_flow_param_one(phi_dN)
    f = f_param(V_p,phi,phi_dN,H)
    _epsilon_2 = 2*epsilon_1 - 6*f
    return _epsilon_2

def f_param_ind(V,V_p,phi,phi_dN):
    H = hubble_param(V,phi,phi_dN)
    sr_ration = np.divide(V_p(phi),3*(H**2)*phi_dN)#Extra H due to dN dev
    _f = 1+sr_ration
    return _f

#This is also eta/3, but is kept as it's own calculation to allow comparison
def f_param(V_p,phi,phi_dN,H):
    sr_ration = np.divide(V_p(phi),3*(H**2)*phi_dN)#Extra H due to dN dev
    _f = 1+sr_ration
    return _f

def mu_param_ind(V,V_pp,phi,phi_dN):
    H = hubble_param(V,phi,phi_dN)
    _mu = np.divide(V_pp(phi), 3*H**2)
    return _mu

def mu_param(V_pp,phi,H):
    _mu = np.divide(V_pp(phi), 3*H**2)
    return _mu

#This is also sometimes known as the second slow-roll parameter and is
#alternatively 3*f, but is kept as seperate calculation for testing
def eta_param(V_p,phi,phi_dN,H):
    _phi_accel = phi_accel_with_N(V_p,phi,phi_dN,H)
    _H_dN = hubble_param_dN(phi_dN, H)
    _eta = -np.divide(_H_dN,H)-np.divide(_phi_accel,phi_dN)
    return _eta

def z_param(phi_dN, a):
    _z = phi_dN*a
    return _z

def z_prime_prime_ind(V,V_p,V_pp,phi,phi_dN, a):
    f = f_param_ind(V,V_p,phi,phi_dN)
    e = hubble_flow_param_one(phi_dN)
    mu = mu_param_ind(V,V_pp,phi,phi_dN)
    H = hubble_param(V,phi,phi_dN)
    _z_pp_by_z= (2+5*e-3*mu-12*f*e+2*e**2)*(a*H)**2
    return _z_pp_by_z

def z_prime_prime(V,V_p,V_pp,phi,phi_dN, H, a):
    f = f_param(V_p,phi,phi_dN,H)
    e = hubble_flow_param_one(phi_dN)
    mu = mu_param(V,V_pp,phi,phi_dN,H)
    _z_pp_by_z= (2+5*e-3*mu-12*f*e+2*e**2)*(a*H)**2
    return _z_pp_by_z

def z_prime_prime_by_aH_ind(V,V_p,V_pp,phi,phi_dN):
    f = f_param_ind(V,V_p,phi,phi_dN)
    e = hubble_flow_param_one(phi_dN)
    mu = mu_param_ind(V,V_pp,phi,phi_dN)
    _z_pp_by_aH = 2+5*e-3*mu-12*f*e+2*e**2
    return _z_pp_by_aH


def z_prime_prime_by_aH(V,V_p,V_pp,phi,phi_dN,H):
    f = f_param(V_p,phi,phi_dN,H)
    e = hubble_flow_param_one(phi_dN)
    mu = mu_param(V,V_pp,phi,phi_dN,H)
    _z_pp_by_aH = 2+5*e-3*mu-12*f*e+2*e**2
    return _z_pp_by_aH

#This is a simple method assuming de Sitter background
def vk_mode_accel_conformal_t_DS(vk_vec, k_mode, tau):
    vk_accel = -(k_mode**2-np.divide(2, tau**2))*vk_vec[0]
    return vk_accel

#Remember, N is the e-foldings done since start a_i
def conformal_time_ind(V, phi, phi_dN, N, a_i):
    H = hubble_param(V,phi,phi_dN)
    a = a_i*np.exp(N)
    #This is the de Sitter approx
    tau = -np.divide(1,a*H)
    return tau

#Remember, N is the e-foldings done since start a_i
def conformal_time(H, a):
    #This is the de Sitter approx
    tau = -np.divide(1,a*H)
    return tau

#This simply gives a Bunch Davies vacuum for some k-mode
def bunch_davies_vacuum(tau, k):
    bdv = np.divide( np.exp(np.complex(0,-k*tau)), np.sqrt(2*k) )
    return bdv

#This is the comoving curvature perturbation
def perturbation_R(v, phi_dN, H, a):
    z = z_param(phi_dN, a)
    _R = np.divide(v,z)
    return _R

#This is psi by Valerie's definition
def perturbation_psi(v, a):
    _R = np.divide(v,a)
    return _R


def phi_squared_inflation_N(phi, phi_initial):
    N = (phi_initial**2 - phi**2)/(4*M_PL**2)
    return N

def phi_squared_inflation_N_inverse(N, phi_initial):
    phi = np.sqrt(phi_initial**2 - 4*N*M_PL**2)
    return phi

def phi_sqaured_inflation_power_spectrum(N, m):
    ps = np.divide((N*m)**2, 3*M_PL**2)
    return ps

def phi_squared_inflation_ns(N, N_f):
    n_s = 1 - np.divide(2,N_f-N)
    return n_s

def sr_power_spectrum_potential(V,V_p,phi):
    ps = np.divide(V(phi)**3, 12*(PI*V_p(phi))**2) / (M_PL**6)
    return ps

def sr_power_spectrum(phi_dN, H):
    ps = (np.divide(H, 2*PI*phi_dN))**2
    return ps

def sr_power_spectrum_next_order(V,V_p,phi,phi_dN,H):
    ps_leading_order = sr_power_spectrum(phi_dN, H)
    eta = eta_param(V_p,phi,phi_dN,H)
    epsilon = hubble_flow_param_one(phi_dN)
    num_factor = 2-np.log(2)-EULER_GAMMA
    ps = ( (1+num_factor*(2*epsilon-eta)-epsilon)**2 )*ps_leading_order
    return ps


#This is the ln version of the power spectrum, sometimes a curly P.
def power_spectrum(v, k, phi_dN, H, a):
    R_mod_squared = ( np.abs(perturbation_R(v, phi_dN, H, a)) )**2
    k_cubed = (k**3)/(2*PI**2)
    Delta_k = k_cubed*R_mod_squared
    return Delta_k

def sr_spectral_index(V_p,phi,phi_dN,H):
    _epsilon = hubble_flow_param_one(phi_dN)
    _eta = eta_param(V_p,phi,phi_dN,H)
    _n_s = 1 - 4*_epsilon + 2*_eta
    return _n_s

#This assumes a data set, such that a gradient can be found
#This will have maimum error at the edges
def spectral_index(Delta_squared_vec, k_vec):
    _n_s = 0
    if Delta_squared_vec.shape[0]<3:
        print('Insufficent data to calculate spectral index')
    elif Delta_squared_vec.shape[0] >= 3:
        _n_s = 1 + np.gradient(np.log(Delta_squared_vec), np.log(k_vec))
    else:
        raise ValueError('Unkown spectral index calculation error.')
    return _n_s


'''
Now defing the stochastic functions
'''
def reduced_potential(V):
    def v(phi):
        v_value = np.divide(V(phi), 24*(PI**2)*(M_PL**4))
        return v_value
    return v

def reduced_potential_diff(V_p):
    def v_p(phi):
        v_p_value = np.divide(V_p(phi), 24*(PI**2)*(M_PL**4))
        return v_p_value
    return v_p

def reduced_potential_ddiff(V_pp):
    def v_pp(phi):
        v_pp_value = np.divide(V_pp(phi), 24*(PI**2)*(M_PL**4))
        return v_pp_value
    return v_pp

def average_N(N_vec):
    mean_N = np.mean(N_vec)
    return mean_N


def delta_N_squared(N_vec):
    N_variance = np.mean(N_vec**2) -  average_N(N_vec)**2
    return N_variance

def third_central_moment(N_vec, axis=None):
    if axis==None:
        mean = average_N(N_vec)
        skewness = np.mean((N_vec-mean)**3)
    elif axis==0:
        means = np.mean(N_vec,  axis=0)
        skewness = np.zeros(N_vec.shape[1])
        for i in range(N_vec.shape[1]):
            skewness[i] = np.mean((N_vec[:,i]-means[i])**3)
    return skewness

def fourth_central_moment(N_vec, axis=None):
    if axis==None:
        mean = average_N(N_vec)
        skewness = np.mean((N_vec-mean)**4)
    elif axis==0:
        means = np.mean(N_vec,  axis=0)
        skewness = np.zeros(N_vec.shape[1])
        for i in range(N_vec.shape[1]):
            skewness[i] = np.mean((N_vec[:,i]-means[i])**4)
    return skewness

#From wikipedia
def skewness_std(n):
    return np.sqrt((6*n*(n-1))/((n-2)*(n+1)*(n+3)))
#From wikipedia
def kurtosis_std(n):
    return np.sqrt((24*n*(n-1)**2)/((n-3)*(n-2)*(n+3)*(n+5)))

def jackknife_old(N_vec, num_bins):
    #First make sure data is randomised
    Ns = np.copy(N_vec)
    #np.random.shuffle(Ns)

    #Next organise into bins. Index must int
    Ns = np.reshape(Ns, (int(Ns.shape[0]/num_bins), num_bins))
    #Define the mean and st of the binned values
    mean_dist = np.mean(Ns, axis=0)
    st_dist = np.std(Ns, axis=0)
    #Find the st of these distributions
    mean_dist_st = np.std(mean_dist) 
    st_dist_st = np.std(st_dist) 
    #Find the standard error
    mean_se = np.divide(mean_dist_st, np.sqrt(num_bins))
    st_sr = np.divide(st_dist_st, np.sqrt(num_bins))
    return mean_se, st_sr

#The stat function must accept axis=0 as an argument
def jackknife(N_vec, num_bins, stat, weights = [None, None]):
    if weights[0] == None:
        #First make sure data is randomised
        Ns = np.copy(N_vec)
        np.random.shuffle(Ns)
        #Next organise into bins. Index must int
        Ns = np.reshape(Ns, (int(Ns.shape[0]/num_bins), num_bins))
        #Define the mean and st of the binned values
        dist_of_stat = stat(Ns, axis=0)
        #Find the st of these distributions
        stat_of_dist = np.std(dist_of_stat) 
        #Find the standard error
        st_error = np.divide(stat_of_dist, np.sqrt(num_bins))
    elif len(weights) == len(N_vec):
        #First make sure data is randomised
        Ns = np.copy(N_vec)
        ws = np.copy(weights)
        random_idexs = np.arange(0, len(Ns), 1, dtype = int)
        np.random.shuffle(random_idexs)
        Ns = Ns[random_idexs]
        ws = ws[random_idexs]
    
        #Next organise into bins. Index must int
        Ns = np.reshape(Ns, (int(Ns.shape[0]/num_bins), num_bins))
        ws = np.reshape(ws, (int(ws.shape[0]/num_bins), num_bins))
        
        #Finding the a distribution of this stat using a for loop
        dist_of_stat = np.zeros(num_bins)
        for i in range(num_bins):
            dist_of_stat[i] = stat(Ns[:,i], ws[:,i])
            
        #Find the st of these distributions
        stat_of_dist = np.std(dist_of_stat) 
        #Find the standard error
        st_error = np.divide(stat_of_dist, np.sqrt(num_bins))
    else:
        raise ValueError('Incorrect weight argument')
    return st_error


'''
Functions from Vennin 2015
'''
#Equation 3.20 in Vennin 2015
def ending_probability(phi_i, phi_1, phi_2, V):
    v = reduced_potential(V)
    def integrand(phi):
        return np.exp(-1/v(phi))
    numerator,_ = integrate.quad(integrand, phi_i, phi_2)
    denominator,_ = integrate.quad(integrand, phi_1, phi_2)
    #Probability inflation ends at phi_1
    p_1 = numerator/denominator
    return p_1

#Equation 3.27 in Vennin 2015
def classicality_criterion(V,V_p, V_pp, phi_int):
    v_func = reduced_potential(V)
    v_p_func = reduced_potential_diff(V_p)
    v_pp_func = reduced_potential_ddiff(V_pp)
    
    v = v_func(phi_int)
    v_p = v_p_func(phi_int)
    v_pp = v_pp_func(phi_int)
        

    eta = np.abs(2*v - np.divide(v_pp*v**2, v_p**2))
    return eta


#Equation 3.28 in Vennin 2015
def mean_N_sto_limit(V,V_p, V_pp, phi_int, phi_end):
    v_func = reduced_potential(V)
    v_p_func = reduced_potential_diff(V_p)
    v_pp_func = reduced_potential_ddiff(V_pp)
    
    def integrand_calculator(phi):
        #Pre calculating values
        v = v_func(phi)
        v_p = v_p_func(phi)
        v_pp = v_pp_func(phi)
        non_classical = v-np.divide((v**2)*v_pp, v_p**2)
        constant_factor = 1/(M_PL**2)
        
        integrand = constant_factor*np.divide(v,v_p)*(1+non_classical)
        return integrand
    
    mean_N, er = integrate.quad(integrand_calculator, phi_end, phi_int)
    
    return mean_N
#Equation 3.35 in Vennin 2015
def delta_N_squared_sto_limit(V,V_p, V_pp, phi_int, phi_end):
    v_func = reduced_potential(V)
    v_p_func = reduced_potential_diff(V_p)
    v_pp_func = reduced_potential_ddiff(V_pp)
    
    def integrand_calculator(phi):
        #Pre calculating values
        v = v_func(phi)
        v_p = v_p_func(phi)
        v_pp = v_pp_func(phi)
        non_classical = 6*v-np.divide(5*(v**2)*v_pp, v_p**2)
        constant_factor = 2/(M_PL**4)
        
        integrand = constant_factor*np.divide(v**4,v_p**3)*(1+non_classical)
        return integrand
    d_N_sq_value, er = integrate.quad(integrand_calculator, phi_end, phi_int)
    return d_N_sq_value

#Equation 3.37 in Vennin 2015
def third_central_moment_N_sto_limit(V,V_p, V_pp, phi_int, phi_end):
    v_func = reduced_potential(V)
    v_p_func = reduced_potential_diff(V_p)
    v_pp_func = reduced_potential_ddiff(V_pp)
    
    def integrand_calculator(phi):
        #Pre calculating values
        v = v_func(phi)
        v_p = v_p_func(phi)
        v_pp = v_pp_func(phi)
        non_classical = 14*v-np.divide(11*(v**2)*v_pp, v_p**2)
        constant_factor = 12/(M_PL**6)
        
        integrand = constant_factor*np.divide(v**7,v_p**5)*(1+non_classical)
        return integrand
    third_moment, er = integrate.quad(integrand_calculator, phi_end, phi_int)
    return third_moment

#THIS WAS FOUND NOT TO WORK!!!!!!!!!!!
#Based on Eq. (3.24) of Vennin 2015
#This is very different to the other calculations, as it is not using an
#analytically derived equation and instead is done fully numerically.
#These equations were derived assuming two absorbing boundaries. As this is 
#done numerically, the second absorbing boundary needs to be explictly included
def stochastic_mean_full_old(V,V_p, V_pp, phi_int, phi_end,\
                                      phi_end2 = False, phi_bar = False,\
                                          root_finding = False):
    #If value was not explcitly given, assume the infinte limit is taken
    if phi_end2 == False:
        phi_end2 = 100*phi_int
    if phi_bar == False:
        #assuming v'(phi)>0, see subsection 3.32 of Vennin 2015
        phi_bar = 100*phi_int
    v_func = reduced_potential(V)
    

    
    def N_mean_integral(phi_bar, phi):#phi_bar needs to be found
        #As integration limits are not constant, need to define the limits as 
        #functions, using SciPy's nquad function
    
        #First integrand, for the 2d integration, so 2 arguments
        def integrand(y,x):
            return np.exp(1/v_func(y)- 1/v_func(x))/v_func(y)
        
        #The limits, defined as functions
        
        #The x limits
        def bounds_x():
            return [phi_end, phi]
        
        #The y limits
        def bounds_y(x):
            return [x, phi_bar]
        
        first_integral,error =\
            integrate.nquad(integrand, [bounds_y, bounds_x])
        print('Error for mean is: '+str(error))
        return first_integral
    
    if root_finding == True:
        sol = optimize.root_scalar(N_mean_integral, args=(phi_end2), method='brentq',\
                                   bracket = [phi_end, phi_end2])
        #Use the phi_bar found to find the integral for the phi_int of interest
        phi_bar = sol.root
    return N_mean_integral(phi_bar, phi_int)/M_PL**2

#This is based on footnote 11 of Vennin 2015
def stochastic_mean_full(V,V_p, V_pp, phi_int, phi_end,\
                                      phi_end2 = False, root_finding = False):
    #If value was not explcitly given, assume the infinte limit is taken
    if phi_end2 == False:
        phi_end2 = 100*phi_int

    v_func = reduced_potential(V)
    
    #Need the probability of crossing phi_end rather than phi_end2
    def prob_integrand(z):
        return np.exp(-1/v_func(z))
    #Using Eq. 3.20 of Vennin 2015
    inte_numerator,_ = integrate.quad(prob_integrand, phi_int, phi_end2)
    inte_denominator,_ = integrate.quad(prob_integrand, phi_end, phi_end2)
    p1 = inte_numerator/inte_denominator
    
    def heaviside(z):
        if z>0:
            return 1
        else:
            return 0
    

    def integrand(y,x):
        return np.divide(np.exp(1/v_func(y)- 1/v_func(x)), v_func(y))\
            *(heaviside(x-phi_int)-p1)
    
    #The limits, defined as functions
    
    #The x limits
    def bounds_x(y):
        return [y, phi_end2]
    
    #The y limits
    def bounds_y():
        return [phi_end, phi_end2]
    
    integral,error =\
        integrate.nquad(integrand, [bounds_x, bounds_y])
    return integral/M_PL**2


#This is very different to the other calculations, as it is not using an
#analytically derived equation and instead is done fully numerically.
#These equations were derived assuming two absorbing boundaries. As this is 
#done numerically, the second absorbing boundary needs to be explictly included
def fourth_central_moment_N_sto_limit_old(V,V_p, V_pp, phi_int, phi_end,\
                                      phi_end2 = False, phi_bar = False,\
                                          root_finding = False):
    #If value was not explcitly given, assume the infinte limit is taken
    if phi_end2 == False:
        phi_end2 = 100*phi_int
    if phi_bar == False:
        #assuming v'(phi)>0, see subsection 3.32 of Vennin 2015
        phi_bar = 100*phi_int
    v_func = reduced_potential(V)
    v_p_func = reduced_potential_diff(V_p)
    v_pp_func = reduced_potential_ddiff(V_pp)
    
    #Define these functions to make the code easier to read and be constent 
    #with the notation of Vennin 2015
    
    def f_p(y):
        non_classical = v_func(y)-\
            np.divide(v_pp_func(y)*v_func(y)**2, v_p_func(y)**2)
        return np.divide(v_func(y), v_p_func(y))*(1+non_classical)/M_PL**2
    
    def sigma_3_p(y):
        non_classical = 14*v_func(y)-\
            11*np.divide(v_pp_func(y)*v_func(y)**2, v_p_func(y)**2)
        return 12*np.divide(v_func(y), v_p_func(y))*(1+non_classical)/M_PL**6
    
    def expo(y,x):#Taylor approximation of the exponetial to second order
        return np.exp(-np.divide(v_p_func(x), v_func(x)**2)*(y-x))*(1+\
                    0.5*(-np.divide(v_pp_func(x), v_func(x)**2)+\
                    2*np.divide(v_p_func(x)**2, v_func(x)**3))*(y-x)**2)
    
    def delta_N_4(phi_bar, phi):#phi_bar needs to be found
        #As integration limits are not constant, need to define the limits as 
        #functions, using SciPy's nquad function
    
        #First integrand, for the 2d integration, so 2 arguments
        def integrand1(y,x):
            return 8*f_p(y)*sigma_3_p(y)*expo(y,x)
        
        #First integrand, for the 3d integration, so 3 arguments
        def integrand2(z,y,x):
            v_terms = np.divide(v_func(z)**4, v_p_func(z)**3)*(1+6*v_func(z)-\
                    5*np.divide(v_pp_func(z)*v_func(z)**2, v_p_func(z)**2))
            return 12*(f_p(y)**2)*v_terms*expo(y,x)
        
        #The limits, defined as functions
        
        #The x limits
        def bounds_x():
            return [phi_end, phi]
        
        #The y limits
        def bounds_y(x):
            return [x, phi_bar]
        
        #The y limits
        def bounds_z(y,x):
            return [phi_end, y]
        
        first_integral,error1 =\
            integrate.nquad(integrand1, [bounds_y, bounds_x])
        print(first_integral)
        second_integral,error2 =\
            integrate.nquad(integrand2, [bounds_z, bounds_y, bounds_x])
        print(second_integral)
        print('Error for 4th moment is: '+str(error1+error2))
        return first_integral+second_integral
    if root_finding == True:
        sol = optimize.root_scalar(delta_N_4, args=(phi_end2), method='brentq',\
                                   bracket = [phi_end, phi_end2])
        #Use the phi_bar found to find the integral for the phi_int of interest
        phi_bar = sol.root
    return delta_N_4(phi_bar, phi_int)

#This is done using Vincent's calculations he gave me
def fourth_central_moment_N_sto_limit(V,V_p, V_pp, phi_int, phi_end):

    v_func = reduced_potential(V)
    v_p_func = reduced_potential_diff(V_p)
    
    def integrand_calculator(phi):
        #Pre calculating values
        v = v_func(phi)
        v_p = v_p_func(phi)
        constant_factor = 120/(M_PL**8)
        
        integrand = constant_factor*np.divide(v**10,v_p**7)
        return integrand
    non_guassian, er = integrate.quad(integrand_calculator, phi_end, phi_int)
    #As Vincent's method explicitly calculates the excess kurtosis, need to
    #add Wick's theorem term
    gaussian_4th_moment =\
        3*delta_N_squared_sto_limit(V,V_p, V_pp, phi_int, phi_end)**2
        
    return gaussian_4th_moment+non_guassian

#Equation 3.37 in Vennin 2015, then divded by sigma^3 to make skewness
def skewness_N_sto_limit(V,V_p, V_pp, phi_int, phi_end):
    v_func = reduced_potential(V)
    v_p_func = reduced_potential_diff(V_p)
    v_pp_func = reduced_potential_ddiff(V_pp)
    
    def integrand_calculator(phi):
        #Pre calculating values
        v = v_func(phi)
        v_p = v_p_func(phi)
        v_pp = v_pp_func(phi)
        non_classical = 14*v-np.divide(11*(v**2)*v_pp, v_p**2)
        constant_factor = 12/(M_PL**6)
        
        integrand = constant_factor*np.divide(v**7,v_p**5)*(1+non_classical)
        return integrand
    skewness_value, er = integrate.quad(integrand_calculator, phi_end, phi_int)
    #Now normalise by the variance
    skewness_value = \
        skewness_value/delta_N_squared_sto_limit(V,V_p, V_pp, phi_int, phi_end)**1.5
    return skewness_value 

#Using the standard relation between the central moments and the kurtosis.
#Fisher is an optional argument 
def kurtosis_N_sto_limit(V,V_p, V_pp, phi_int, phi_end, Fisher = True):
    #The excess kurtosis over the expected Gaussian amount
    if Fisher == True:
        return fourth_central_moment_N_sto_limit(V,V_p, V_pp, phi_int, phi_end)/\
            delta_N_squared_sto_limit(V,V_p, V_pp, phi_int, phi_end)**2-3
    else:
        return fourth_central_moment_N_sto_limit(V,V_p, V_pp, phi_int, phi_end)/\
            delta_N_squared_sto_limit(V,V_p, V_pp, phi_int, phi_end)**2

def skewness_N_chaotic_inflation(m, phi_i):
    skew = np.divide((m**4)*(phi_i**10), 61440*(PI**4)*(M_PL**14))
    return skew
    
#Equation 4.4 in Vennin 2015
def power_spectrum_sto_limit(V,V_p, V_pp, phi_int):
    v_func = reduced_potential(V)
    v_p_func = reduced_potential_diff(V_p)
    v_pp_func = reduced_potential_ddiff(V_pp)
    
    v = v_func(phi_int)
    v_p = v_p_func(phi_int)
    v_pp = v_pp_func(phi_int)
    ps_cl = sr_power_spectrum_potential(V, V_p, phi_int)

    ps_sto = ps_cl*(1+5*v -np.divide(4*(v**2)*v_pp, v_p**2))
    return ps_sto

#This only takes numbers as arguments
def quantum_diffusion_mu(delta_phi_well, V_0):
    v_0 = np.divide(V_0, 24*(PI**2)*(M_PL**4))
    mu = np.divide(delta_phi_well, np.sqrt(v_0)*M_PL)
    return mu

#This is only true for a constant potential
def quantum_diffusion_x(phi, phi_end, delta_phi_well):
    return np.divide(phi - phi_end, delta_phi_well)

def quantum_diffusion_mean_N(mu, x):
    qd_mean_N = (mu**2)*x*(1-0.5*x)
    return qd_mean_N


def quantum_diffusion_var_N(mu, x):
    qd_var_N = (mu**4)*(1-(1-x)**4)/6
    return qd_var_N


#This is only true for a constant potential
def quantum_diffusion_N_probability_dist(N, mu, x, n):
    the_sum = 0.0
    for i in range(n):
        first_term = 2*(i+1)-x
        second_term =2*i+x
        expo = -np.divide(mu**2, 4*N)
        the_sum += ((-1)**i)*( first_term*np.exp(expo*first_term**2)\
                              + second_term*np.exp(expo*second_term**2) )
            
    answer = np.divide(mu, 2*(PI**0.5)*(N**1.5))*the_sum
    return answer

#This assumes a 'shallow' and a 'wide' well. See Ezquiaga 2020 Eq. (3.33)
#My tilt = their alpha/M_pl
def quantum_diffusion_N_probability_dist_alt(N, x, delta_phi_well, v_0, n):
    the_sum = 0.0
    mu = np.divide(delta_phi_well, np.sqrt(v_0)*M_PL)
    for i in range(n):
        first_term = (i+0.5)*np.sin(PI*(i+0.5)*x)
        expo_co = np.divide((i+0.5)*PI, mu)**2
        the_sum += first_term*np.exp(-expo_co*N)
            
    P = np.divide(2*PI, mu**2)*the_sum
    return P

#This assumes a 'shallow' and a 'wide' well. See Ezquiaga 2020 Eq. (3.33)
#My tilt = their alpha/M_pl
def quantum_diffusion_N_probability_dist_alt2(N, x, mu, n):
    the_sum = 0.0
    for i in range(n):
        first_term = (i+0.5)*np.sin(PI*(i+0.5)*x)
        expo_co = np.divide((i+0.5)*PI, mu)**2
        the_sum += first_term*np.exp(-expo_co*N)
            
    P = np.divide(2*PI, mu**2)*the_sum
    return P


#This assumes a 'shallow' and a 'wide' well. See Ezquiaga 2020 Eq. (3.33)
#My tilt = their alpha/M_pl
def wide_tilted_well_N_probability_dist(N, x, tilt, delta_phi_well, v_0, n):
    the_sum = 0.0
    mu = np.divide(delta_phi_well, np.sqrt(v_0)*M_PL)
    for i in range(n):
        first_term = i*np.sin(PI*i*x)
        expo_co = np.divide((tilt*M_PL)**2, 4*v_0) + np.divide(i*PI, mu)**2
        the_sum += first_term*np.exp(-expo_co*N)
            
    P = np.divide(2*PI, mu**2)*np.exp((tilt*delta_phi_well*x)/(2*v_0))*the_sum
    return P


#Formula from Ezquiaga 2020 et al for the tilted well case, using 3.17 and 3.19
#This function currently does not work and should not be used in it's current
#state.
def old_wide_tilted_well_N_probability_dist(N, mu, x, tilt, delta_phi_well, n):
    print(delta_phi_well)
    print(((mu*M_PL)**2)/(2*delta_phi_well))
    print(np.exp(((mu*M_PL)**2)/(2*delta_phi_well)))
    def a_n(mu, x, tilt, delta_phi_well, j):#Note, this formula might only work for phi_uv
        return -((-1)**n)*(2*PI/(mu**2))*(j+1)*\
            np.exp(((mu*M_PL)**2)/(2*delta_phi_well)*x)*np.sin(PI*(j+1)*(x-1))
            
    def Lamda_n(mu, x, tilt, delta_phi_well, j):
        return (((M_PL**2)*tilt*mu)/(2*delta_phi_well)) +\
            (1-(delta_phi_well/tilt)*(2/(mu*M_PL))**2)*((j*PI+PI)/mu)**2
            
    P=0
    for i in range(n):
        P += a_n(mu, x, tilt, delta_phi_well, i)*\
            np.exp(-Lamda_n(mu, x, tilt, delta_phi_well, i)*N)
    
    return P

'''
Analytical Results for Stochastic m^2\phi^2 inflation
'''

#This is very inefficent, but is accurate. This follows the standard
#conventions, where the forward operation is negative in the exponential.
#THIS WILL NOT WORK FOR COMPLEX EXPONENTIAL!
def continuous_ft(w,func, component = None, t_lower = -np.inf,\
                          t_upper = np.inf):
    def integrand_real(t):
        return (np.exp(np.complex(0,-w*t))*func(t)).real
    
    def integrand_imaginary(t):
        return (np.exp(np.complex(0,-w*t))*func(t)).imag
    
    if component == 'real':
        real_component,_ = integrate.quad(integrand_real, t_lower, t_upper,limit=400)
        return real_component/np.sqrt(2*PI)
    elif component == 'imag':
        img_component,_ = integrate.quad(integrand_imaginary, t_lower, t_upper,limit=400)
        return -img_component/np.sqrt(2*PI)
    else:
        real_component,_ = integrate.quad(integrand_real, t_lower, t_upper,limit=400)
        img_component,_ = integrate.quad(integrand_imaginary, t_lower, t_upper,limit=400)
        return np.complex(real_component, img_component)/np.sqrt(2*PI)

#This is very inefficent, but is accurate. This follows the standard
#conventions, where the inverse operation is positive in the exponential.
#THIS WILL NOT WORK FOR COMPLEX EXPONENTIAL!
def continuous_inverse_ft(t,func, component = None, t_lower = -np.inf,\
                          t_upper = np.inf):
    def integrand_real(w):
        return (np.exp(np.complex(0,w*t))*func(w)).real
    
    def integrand_imaginary(w):
        return (np.exp(np.complex(0,w*t))*func(w)).imag
    
    if component == 'real':
        real_component,_ = integrate.quad(integrand_real, t_lower, t_upper,limit=400)
        return real_component/np.sqrt(2*PI)
    elif component == 'imag':
        img_component,_ = integrate.quad(integrand_imaginary, t_lower, t_upper,limit=400)
        return img_component/np.sqrt(2*PI)
    else:
        real_component,_ = integrate.quad(integrand_real, t_lower, t_upper,limit=400)
        img_component,_ = integrate.quad(integrand_imaginary, t_lower, t_upper,limit=400)
        return np.complex(real_component, img_component)/np.sqrt(2*PI)
    
#This is legacy code - do not use unless you have to. mpmath is better
def kummer_1f1(a,b,z,tol=10**-10):
    the_sum = 1
    for i in range(999):
        i+=1
        a_term = np.prod(a+np.arange(0,i,1),dtype='complex128')
        b_term = np.prod(b+np.arange(0,i,1),dtype='complex128')
        new_term = (a_term*z**i)/(b_term*math.factorial(i))
        error = np.abs(new_term)/np.abs(the_sum)
        the_sum += new_term
        #If the new term produced a negligible correction
        if i>10 and error<tol:
            break
    return the_sum 
        

def chaotic_inflation_characteristic_function(t, phi, phi_end, V):
    v_0 = V(M_PL)/(24*(M_PL**4)*(PI**2))
    v = V(phi)/(24*(M_PL**4)*(PI**2))
    v_end = V(phi_end)/(24*(M_PL**4)*(PI**2))
    alpha = np.sqrt(np.complex(1,-np.divide(4*t,v_0)))
    term_1 = (V(phi)/V(phi_end))**(0.25 - 0.25*alpha)
    num = hyp1f1(-0.25 + 0.25*alpha, 1 + 0.5*alpha, -1/v)
    denom = hyp1f1(-0.25 + 0.25*alpha, 1 + 0.5*alpha, -1/v_end)
    chi_mp = (term_1*num)/denom
    chi = np.complex(float(nstr(chi_mp.real, n=12)), float(nstr(chi_mp.imag, n=12)))
    return chi

'''
Importance sampling equations
'''

#Pretty sure this version is wrong
def importance_sampling_mean_old(data, weight):
    return np.mean(weight*data)

#Pretty sure this version is wrong
def importance_sampling_var_old(data, weight):
    mean_q = importance_sampling_mean(data, weight)
    return np.mean((data*weight-mean_q)**2)

#Pretty sure this version is wrong
def importance_sampling_skew_old(data, weight):
    mean_q = importance_sampling_mean(data, weight)
    std_q = importance_sampling_st(data, weight)
    return np.mean((np.divide(data*weight - mean_q,std_q))**3)

#Pretty sure this version is wrong
def importance_sampling_kurtosis_old(data, weight, Fisher = True):
    mean_q = importance_sampling_mean(data, weight)
    std_q = importance_sampling_st(data, weight)
    kappa = np.mean((np.divide(data*weight - mean_q,std_q))**4)
    if Fisher == True:#So it is normalised to 0 if a normal distribution
        kappa -= 3
    return kappa


#Works on the whole data set: DOES NOT DO columns
def importance_sampling_mean(data, weight):
    return np.average(data, weights = weight)

#Works on the whole data set: DOES NOT DO columns
def importance_sampling_var(data, weight):
    mean = importance_sampling_mean(data, weight)
    return np.average((data-mean)**2, weights = weight)

def importance_sampling_st(data, weight):
    return importance_sampling_var(data,weight)**0.5

#Works on the whole data set: DOES NOT DO columns
def importance_sampling_skew(data, weight):
    mean = importance_sampling_mean(data, weight)
    std = importance_sampling_st(data, weight)
    return np.average((np.divide(data - mean, std))**3, weights = weight)


#Works on the whole data set: DOES NOT DO columns
def importance_sampling_kurtosis(data, weight, Fisher = True):
    mean = importance_sampling_mean(data, weight)
    std = importance_sampling_st(data, weight)
    kappa = np.average((np.divide(data - mean, std))**4,\
                       weights = weight)
    if Fisher == True:#So it is normalised to 0 if a normal distribution
        kappa -= 3
    return kappa

def imporatnce_sampling_3rd_moment(data, weight):
    mean = importance_sampling_mean(data, weight)
    return np.average((data - mean)**3, weights = weight)

def imporatnce_sampling_4th_moment(data, weight):
    mean = importance_sampling_mean(data, weight)
    return np.average((data - mean)**4, weights = weight)

def importance_sampling_3rd_cumulant(data, weight):
    return imporatnce_sampling_3rd_moment(data, weight)

def importance_sampling_4th_cumulant(data, weight):
    kappa2 = importance_sampling_var(data, weight)
    kappa4 = imporatnce_sampling_4th_moment(data, weight)
    return kappa4-3*kappa2**2


#Using Eq. (48) of arXiv:nucl-th/9809075v1
#Does not take into account the mean
def tail_prob_gaussian(x_star, var):
    return 0.5*(1-special.erf(x_star/np.sqrt(2*var)))


#Estimates the size of a histogram bar using the analytical PDF.
def histogram_analytical_height(bins, analytical_pdf):
    bin_width = np.diff(bins)
    #Integrate over the range to find the probaility. Then divide by the width
    #to estimate the bin height (as it's a probability DENSITY)
    bin_heights =\
        [integrate.quad(analytical_pdf, bins[i], bins[i+1])/bin_width[i]\
         for i in range(len(bin_width))]
    bin_heights = np.array(bin_heights)
    
    return bin_heights[:,0]#Only return heights, not the errors

#For each bin having the same number of data points, this returns the edges of
#said bins
def histogram_same_num_data_bins(data, num_bins):
    if len(data)%num_bins == 0:
        bins = np.zeros(num_bins+1)
        num_in_bins = int(len(data)/num_bins)
        #Define all of the lft hand sides
        bins[0:num_bins] = data[0::num_in_bins]
        #Define thr right hand side
        bins[num_bins] = np.max(data)
    else:
        raise ValueError('Incorrect number of bins for data')
    return bins


#Returns the data used in histogram bars as columns. 
def histogram_data_in_bins(data, weights, bins):
    data_columned  = np.zeros([len(data), len(bins)-1])
    weights_columned  = np.zeros([len(data), len(bins)-1])
    #The bins have the same range until the end
    for i in range(len(bins)-2):
        data_logic = (data>=bins[i]) & (data<bins[i+1])
        data_slice = data[data_logic]
        data_columned[0:len(data_slice),i] = data_slice
        weights_columned[0:len(data_slice),i] = weights[data_logic]
    #The final bin also includes the last value, so has an equals in less than
    data_logic = (data>=bins[len(bins)-2]) & (data<=bins[len(bins)-1])
    data_slice = data[data_logic]
    data_columned[0:len(data_slice),len(bins)-2] = data_slice
    weights_columned[0:len(data_slice),len(bins)-2] = weights[data_logic]
    return data_columned, weights_columned

#Uses the formula defined in
#https://suchideas.com/articles/maths/applied/histogram-errors/
# to find the varaince of the sum. This then defines the standard deviation,
#with the error defined as 2 standard deviations. The errors can optionally be
#normalised
def histogram_weighted_bin_errors(weight_dist, num_data_tot,\
                                  normalisation = 1, num_std = 2):
    num_bins = len(weight_dist[0,:])
    errors = np.zeros(num_bins)
    #As we have normalised the data
    weight_dist = weight_dist/normalisation
    errors  = np.array([np.sum(weight_dist[:,i]**2) -\
                        (np.sum(weight_dist[:,i])**2)/num_data_tot\
                            for i in range(num_bins)])
        
    errors =num_std*np.sqrt(errors)
    return errors

#THIS IS A MESS!!!!!!!!!!!!!!
#Alternative method for calculating the errors of the histogram bars by using
#the simplified jackknife analysis. The data is sub-sampled into many
#histograms with the same bins. This way a distribution for the different
#heights can be done. Takes the bins used as general argument
#Arguments must be numpy arrays
def histogram_weighted_bin_errors_jackknife(data_input, weights_input, bins,\
                                            num_sub_samps, density = True,
                                            analytical_pdf = None,\
                                            log10_errors = False,\
                                            lognormal = False,\
                                            return_log_w_std = False):
    #Make an array of random indexs
    indx = np.arange(0, len(data_input), 1)
    np.random.shuffle(indx)
    #This allows the data to be randomised and keep the weights matched
    data = data_input[indx]
    weights = weights_input[indx]
    num_bins = len(bins)-1#bins is an array of the side, so one less
    if lognormal == True:
        exponent_array = np.zeros((num_bins, num_sub_samps))#Storage
        _ , weights_in_bins_full =\
            histogram_data_in_bins(data_input, weights_input, bins)
        log_normal_height_est = np.zeros(num_bins)
        norm = len(data)*np.diff(bins)[0]
        for i in range(num_bins):
            w = weights_in_bins_full[:,i]
            w = w[w>0]
            log_normal_height_est[i] = log_normal_mean(w)
    else:
        height_array = np.zeros((num_bins, num_sub_samps))#Storage
        

    #Next organise into subsamples
    data =\
        np.reshape(data, (int(data.shape[0]/num_sub_samps), num_sub_samps))
    weights =\
        np.reshape(weights, (int(weights.shape[0]/num_sub_samps), num_sub_samps))
    #Find the heights of the histograms, for each sample
    
    if return_log_w_std == False:
        
        #Use the same analytical normalisation for each histogram
        if analytical_pdf != None:
            norm =\
                histogram_analytical_normalisation(bins, analytical_pdf,\
                                                   int(data.shape[0]/num_sub_samps))
                    
        for i in range(num_sub_samps):
            '''
            height_array[:, i], _ = np.histogram(data[:,i], weights = weights[:,i],\
                                   bins = bins, density = density)
            '''
            _ , ws = histogram_data_in_bins(data[:,i], weights[:,i], bins)
            if lognormal == False:
                heights = np.sum(ws, axis=0)
            elif lognormal == True:
                exponents = np.zeros(ws.shape[1])
                for k in range(ws.shape[1]):
                    w = ws[:,k]
                    #Only do calculation if there is non-zero weights, 
                    #otherwise it defaults to zero
                    if np.any([w>0]) == True:
                        w = w[w>0]
                        w_log = np.log(w)
                        w_log_std = np.std(w_log)
                        w_log_mean = np.mean(w_log)
                        exponents[k] = w_log_mean+0.5*w_log_std**2
                        
            else:
                raise ValueError('Lognormal argument must be boolean')
                
                
            if lognormal == True:
                exponent_array[:, i] = exponents
            elif density == True:
                if analytical_pdf == None:
                    #Using the area of the (expected) histogram to normalise, 
                    #different for each histogram. There is subtly here, as it
                    #needs to the area of the histogram if the data was NOT weighted
                    norm = len(data[:,i])*np.diff(bins)[0]
                #Normalise this histogrram
                height_array[:, i] = heights/norm
            elif density == False:
                height_array[:, i] = heights
            else:
                raise ValueError('Incorrect density argument input')

        if lognormal == True:
            error = np.zeros((2,num_bins))
        else:
            error = np.zeros(num_bins)
        sqrt_sub_samples = np.sqrt(num_sub_samps)
        
        #Now can find the standard deviation of the heights, as the bins are the
        #same. Then divide by the sqaure root of the number of samples by jackknife
        #and you have the error
        for j in range(num_bins):
            if lognormal == True:
                exponent_values = exponent_array[j,:]
                exponent_values = exponent_values[exponent_values!=0]
                exponent_error = np.std(exponent_values)/sqrt_sub_samples
                error[0,j] =\
                    log_normal_height_est[j]*(1-np.exp(-exponent_error))
                error[1,j] =\
                    log_normal_height_est[j]*(np.exp(exponent_error)-1)
            elif log10_errors == False:
                bars = height_array[j,:]
                if np.any([bars>0]):
                    #Used to be just np.std(bars)
                    error[j] = np.std(bars[bars>0])/sqrt_sub_samples
                else:
                    error[j] = 0#Remove any empty histogram bars
            elif log10_errors == True:
                bars = height_array[j,:]
                if np.any([bars>0]):
                    error[j] = np.std(np.log10(bars[bars>0]))/sqrt_sub_samples
                else:
                    error[j] = 0#Remove any empty histogram bars
            else:
                raise ValueError('argument log_errors must be a ')
    
        return error
    
    elif return_log_w_std == True:
        log_w_array = np.zeros((num_bins, num_sub_samps))#Storage
        
        #Use the same analytical normalisation for each histogram
        if analytical_pdf != None:
            norm =\
                histogram_analytical_normalisation(bins, analytical_pdf,\
                                                   int(data.shape[0]/num_sub_samps))
                    
        for i in range(num_sub_samps):
            '''
            height_array[:, i], _ = np.histogram(data[:,i], weights = weights[:,i],\
                                   bins = bins, density = density)
            '''
            _ , ws = histogram_data_in_bins(data[:,i], weights[:,i], bins)
            if lognormal == False:
                heights = np.sum(ws, axis=0)
            elif lognormal == True:
                exponents = np.zeros(ws.shape[1])
                for i in range(ws.shape[1]):
                    w = ws[:,i]
                    #Only do calculation if there is non-zero weights, 
                    #otherwise it defaults to zero
                    if np.any([w>0]) == True:
                        w = w[w>0]
                        w_log = np.log(w)
                        w_log_std = np.std(w_log)
                        w_log_mean = np.mean(w_log)
                        exponents[i] = w_log_mean+0.5*w_log_std**2
            else:
                raise ValueError('Lognormal argument must be boolean')
                
            log_w_array[:, i] = np.std(ws, axis=0)
            
            if lognormal == True:
                exponent_array[:, i] = exponents
            elif density == True:
                if analytical_pdf == None:
                    #Using the area of the (expected) histogram to normalise, 
                    #different for each histogram. There is subtly here, as it
                    #needs to the area of the histogram if the data was NOT weighted
                    norm = len(data[:,i])*np.diff(bins)[0]
                #Normalise this histogrram
                height_array[:, i] = heights/norm
            elif density == False:
                height_array[:, i] = heights
            else:
                raise ValueError('Incorrect density argument input')
        #Now can find the standard deviation of the heights, as the bins are the
        #same. Then divide by the sqaure root of the number of samples by jackknife
        #and you have the error
        
        if lognormal == True:
            error = np.zeros((2,num_bins))
        else:
            error = np.zeros(num_bins)
        w_std_error = np.zeros(num_bins)
        sqrt_sub_samples = np.sqrt(num_sub_samps)
        for j in range(num_bins):
                
            bars = height_array[j,:]
            w_stds = log_w_array[j, :]
            w_std_error[j] = np.std(w_stds[w_stds>0])/sqrt_sub_samples
            
            if lognormal == True:
                exponent_values = exponent_array[j,:]
                exponent_values = exponent_values[exponent_values!=0]
                exponent_error = np.std(exponent_values)/sqrt_sub_samples
                error[0,j] =\
                    log_normal_height_est[j]*(1-np.exp(-exponent_error))
                error[1,j] =\
                    log_normal_height_est[j]*(np.exp(exponent_error)-1)
            elif log10_errors == False:
                if np.any([bars>0]):
                    #Used to be just np.std(bars)
                    error[j] = np.std(bars[bars>0])/sqrt_sub_samples
                else:
                    error[j] = 0#Remove any empty histogram bars
            elif log10_errors == True:
                if np.any([bars>0]):
                    error[j] = np.std(np.log10(bars[bars>0]))/sqrt_sub_samples
                else:
                    error[j] = 0#Remove any empty histogram bars
            else:
                raise ValueError('argument log_errors must be a ')
    
        return error, w_std_error
        

#This works out the error on the hisogram bars, then truncates the data to
#remove the data such that all of the histograms have errors below a threshold
def histogram_truncation(data, weights, num_bins, threshold, side = 'left'):
    #Histogram analysis
    bins = np.linspace(np.min(data), np.max(data), num_bins+1)
    data_bins, weight_bins = histogram_data_in_bins(data, weights, bins)
    #Work out the area of the unweighted histogram to find the area,
    #bar height*width
    height = np.sum(weight_bins, axis=0)
    area_norm = np.sum(height*np.diff(bins))
    height  = height/area_norm
    errors = histogram_weighted_bin_errors(weight_bins, len(data),\
                                               normalisation = area_norm)
    if side == 'left':
        #Find the bin which first has error/height (relative error) greater
        #than the threshild, by blooking at the bin values and taking the
        #right hand side of said bin as the point we trucate data below
        remove_data_below =\
            bins[np.max(np.where((errors/height)[0:int(len(errors)/2)]>\
                                 threshold))+1]
        print(np.max(np.where((errors/height)[0:int(len(errors)/2)]>\
                                 threshold)))
        remove_logic = data>remove_data_below
    elif side == 'right':
        #Find the bin which first has error/height (relative error) greater
        #than the threshild, by looking at the bin values and taking the
        #left hand side of said bin as the point we trucate data above
        remove_data_below =\
            bins[np.max(np.where((errors/height)[int(len(errors)/2):]>\
                                 threshold))-1]
        remove_logic = data<remove_data_below
    data_truncated = data[remove_logic]
    weights_truncated = weights[remove_logic]
    return data_truncated, weights_truncated

#Trucates data above a certain threshold. Has options for both weights and if
#a the trucation needs to be rounded up to a certain value.
def histogram_data_truncation(data, threshold, weights=0,\
                              num_sub_samples = None):
    if isinstance(weights, int):
        if num_sub_samples == None:
            return data[data<threshold]
        elif isinstance(num_sub_samples, int):
            data = np.sort(data)
            num_above_threshold = len(data[data>threshold])
            #Want to remove a full subsamples worth
            rounded_num_above_threshold =\
                round(num_above_threshold/num_sub_samples)+1
            return data[:-rounded_num_above_threshold]
        
    else:
        if num_sub_samples == None:
            data_remove_logic = data<threshold
            return data[data_remove_logic], weights[data_remove_logic]
        elif isinstance(num_sub_samples, int):
            #Sort in order of increasing Ns
            sort_idx = np.argsort(data)
            data = data[sort_idx]
            weights = weights[sort_idx]
            num_above_threshold = len(data[data>threshold])
            #Want to remove a full subsamples worth
            if num_above_threshold>0:
                rounded_num_above_threshold =\
                    (round(num_above_threshold/num_sub_samples)+1)*num_sub_samples
                return data[:-rounded_num_above_threshold],\
                    weights[:-rounded_num_above_threshold]
            else:
                return data, weights
    

    
#This assumes a density based histogram
#This does a left point integral, i.e. if the left side of the bin is above the
#threshold, it is included in the integral.
def histogram_integral(bins, bins_heights, threshold):
    #[:(-1)] as the last bin value is the right hand side of the bin
    bin_logic = bins[:(-1)]>=threshold
    area = (np.diff(bins)[bin_logic])*(bins_heights[bin_logic])
    return area
    
#Returns the normalisation factor for a histogram, including one with weights
def histogram_analytical_normalisation(bins, num_sims):
    return num_sims*np.diff(bins)[0]


#Obsolete!!! Don't use!
#Returns the analytical normallisation factor
def histogram_analytical_normalisation_old(bins, analytical_pdf, num_sims):
    bin_width = np.diff(bins)
    #Integrate over the range to find the probaility. 
    bin_heights =\
        [integrate.quad(analytical_pdf, bins[i], bins[i+1])\
         for i in range(len(bin_width))]
    bin_heights = np.array(bin_heights)
    
    heights_analytical = bin_heights[:,0]#Only heights, not the errors

    area = num_sims*np.sum(heights_analytical*bin_width)
    #Why this additional term works is as yet unknwon
    additional_term,_ = integrate.quad(analytical_pdf, bins[0], bins[-1])
    norm = area/additional_term
    return norm
    
    
'''
Log-normal equations
'''   

#The data provided needs to be raw data - the log is then taken in the function
#Defaults to method proposed by Shen 2006 Statist. Med. 2006; 25:3023–3038
def log_normal_mean(data, method = 'ML', position = None):
    data_log = np.log(data)
    data_log_mean = np.mean(data_log)
    data_log_std = np.std(data_log, ddof = 1)#Unbiased standard deviation
    n = len(data_log)
    if data_log_std**2>=(n+4)/2:
        if isinstance(position, float):
            print('Possible convergance error in Shen2006 method at '+\
                  str(position))
        else:
            print('Possible convergance error in Shen2006 method')
            
    if method == 'naive':
        mean = np.mean(data)
        
    elif method == 'Shen':
        fraction = np.divide((n-1)*n*data_log_std**2,\
                             2*(n+4)*(n-1)+3*n*data_log_std**2)
        mean = np.exp(data_log_mean+fraction)
        
    elif method == 'skewed_log_normal':
        #Using scistat to calculate the cumulants
        cumulant_3 = stats.kstat(data_log, n=3)
        cumulant_4= stats.kstat(data_log, n=4)
        
        amplitude = np.exp(data_log_mean+0.5*data_log_std**2)
        
        #This is the mean of the skewed log-normal distribution.
        #I.e. if we use a Edgeworth expansion rather than a normal distribution
        #I found this, see "Mean of Skewed Log-Normal Distribution.pdf"
        mean = amplitude*(1+cumulant_3/6+cumulant_4/24+\
                          (cumulant_3**2)/72)
        if mean<0 and isinstance(position, float):
            print('skewed log-normal mean error at '+str(position))
            
    elif method == 'skewed_log_normal_numcal':
        #Using scistat to calculate the cumulants
        cumulant_3 = stats.kstat(data_log, n=3)
        cumulant_4 = stats.kstat(data_log, n=4)

        def skewed_log_normal_integrand(lnx):
            return np.exp(lnx)*pdf_gaussian_skew_kurtosis(lnx, data_log_mean,\
                                data_log_std, cumulant_3, cumulant_4)
            
        mean,_ = integrate.quad(skewed_log_normal_integrand, -np.inf, np.inf)
        
    elif method == 'skewed_log_normal_fitting':
        #Using scistat to calculate the cumulants
        cumulant3_guess = stats.kstat(data_log, n=3)
        cumulant4_guess = stats.kstat(data_log, n=4)
        
        def edgeworth_expansion(x, cum3, cum4):
            return pdf_gaussian_skew_kurtosis(x, data_log_mean,\
                                data_log_std, cum3, cum4)

        def skewed_log_normal_integrand(lnx, cum3, cum4):
            return np.exp(lnx)*pdf_gaussian_skew_kurtosis(lnx, data_log_mean,\
                                data_log_std, cum3, cum4)
                
        heights, bins = np.histogram(data_log, bins=50, density=True)
        bin_centres =\
            np.array([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)])

            
        edgeworth_params, cv =\
            optimize.curve_fit(edgeworth_expansion, bin_centres,\
                                 heights,\
                                p0 = (cumulant3_guess, cumulant4_guess))
            
        mean,_ =\
            integrate.quad(skewed_log_normal_integrand, -np.inf, np.inf,\
                           args = (edgeworth_params[0], edgeworth_params[1]))
        
        
    elif method == 'skewed_log_normal2':
        
        #Using Andrew's formula, with sign changed. This is based on the
        #skew normal distribution, from the wikipedia page
        #https://en.wikipedia.org/wiki/Skew_normal_distribution
        data_log_skew = stats.skew(data_log)
        skew_mod = np.abs(data_log_skew)
        delta = np.sign(data_log_skew)*np.sqrt( 0.5*PI*np.divide(skew_mod**(2/3),\
                                    skew_mod**(2/3) + (2+0.5*PI)**(2/3)) )
        omega = data_log_std/np.sqrt(1-2*(delta**2)/PI)
        xi = data_log_mean - omega*delta*np.sqrt(2/PI)
        
        mean = np.exp(xi-0.5*omega**2)*(1+special.erf(omega*delta/(2**0.5)))
            
    elif method == 'skewed_log_normal_numcal2':
        data_log_skew = stats.skew(data_log)
        skew_mod = np.abs(data_log_skew)
        delta = np.sign(data_log_skew)*np.sqrt( 0.5*PI*np.divide(skew_mod**(2/3),\
                                    skew_mod**(2/3) + (2+0.5*PI)**(2/3)) )
        alpha = delta/np.sqrt(1-delta**2)
        omega = data_log_std/np.sqrt(1-2*(delta**2)/PI)
        xi = data_log_mean - omega*delta*np.sqrt(2/PI)
        
        def skewed_log_normal_integrand(lnx):
            return np.exp(lnx)*stats.skewnorm.pdf(lnx, alpha, xi,
            omega)
            
        mean,_ = integrate.quad(skewed_log_normal_integrand, -np.inf, np.inf)
    elif method == 'UMVU':
        def g_term(t, i, n):
            #largest allowed isgamma(171)
            if (0.5*n-0.5+i)<171:
                numerator = special.gamma(0.5*n-0.5)
                denominator = np.math.factorial(i)*special.gamma(0.5*n-0.5+i)
                                
                return (numerator/denominator)*((n-1)/(2*n)*t)**i
            #Have to use 5.11.12 of DLMF
            else:
                ratio = (0.5*n-0.5)**i
                return (ratio*((n-1)/(2*n)*t)**i)/np.math.factorial(i)
        
        i_array = np.arange(0,10)
        g_term_values = [g_term(0.5*n*data_log_std**2,i,n) for i in i_array]
        mean = np.exp(data_log_mean)*np.sum(g_term_values)
    elif method == 'MSE1':
        def g_term(t, i, n):
            #largest allowed isgamma(171)
            if (0.5*n-0.5+i)<171:
                numerator = special.gamma(0.5*n-0.5)
                denominator = np.math.factorial(i)*special.gamma(0.5*n-0.5+i)
                                
                return (numerator/denominator)*((n-1)/(2*n)*t)**i
            #Have to use 5.11.12 of DLMF
            else:
                ratio = (0.5*n-0.5)**i
                return (ratio*((n-1)/(2*n)*t)**i)/np.math.factorial(i)

        cofficent = np.divide(n-3,2*n-2)
        i_array = np.arange(0,10)
        g_term_values =\
            [g_term(cofficent*n*data_log_std**2,i,n) for i in i_array]
        mean = np.exp(data_log_mean)*np.sum(g_term_values)
        
    elif method == 'MSE2':
        def g_term(t, i, n):
            #largest allowed isgamma(171)
            if (0.5*n-0.5+i)<171:
                numerator = special.gamma(0.5*n-0.5)
                denominator = np.math.factorial(i)*special.gamma(0.5*n-0.5+i)
                                
                return (numerator/denominator)*((n-1)/(2*n)*t)**i
            #Have to use 5.11.12 of DLMF
            else:
                ratio = (0.5*n-0.5)**i
                return (ratio*((n-1)/(2*n)*t)**i)/np.math.factorial(i)

        cofficent = np.divide(n-4,2*n-2)
        i_array = np.arange(0,10)
        g_term_values =\
            [g_term(cofficent*n*data_log_std**2,i,n) for i in i_array]
        mean = np.exp(data_log_mean)*np.sum(g_term_values)
    
    else:           
        mean = np.exp(data_log_mean+0.5*data_log_std**2)
        
    return mean
    
    
    
    
'''
Misc
'''
def He3(y):
    return y**3-3*y

def He4(y):
    return y**4-6*y+3

def He6(y):
    return y**6-15*y**4+45*y**2-15

#IMPORTANT - I've added a new term to this series
#Using the Gram–Charlier A series to approximate an arbitary pdf with a 
#Gaussian distribution, plus two higher order terms from the 3rd and 4th
#cumulants. There are issues with convergence and error,
#see https://en.wikipedia.org/wiki/Edgeworth_series
def pdf_gaussian_skew_kurtosis(x, mean, sigma, third_cumulant, fourth_cumulant):
    norm_x = (x-mean)/sigma
    
    skew_term = np.divide(third_cumulant*He3(norm_x), 6*sigma**3)
    kurtosis_term = np.divide(fourth_cumulant*He4(norm_x), 24*sigma**4)
    #This term is NEW!!!!!!!!!!
    skew_squared_term = np.divide(He6(norm_x)*third_cumulant**2, 72*sigma**6)
    
    gaussian = np.divide(np.exp(-0.5*norm_x**2), sigma*(2*PI)**0.5)
    
    return gaussian*(1+skew_term+kurtosis_term+skew_squared_term)


#Using the Gram–Charlier A series
#https://en.wikipedia.org/wiki/Edgeworth_series to approximate when we expect
#classical deviation from a gaussian. This is done by finding x such that the
#higher order terms of the edgeworth expanion are 
#nu is the amount pf deviation from a Gaussian.
def gaussian_deviation(mean, sigma, third_cumulant, fourth_cumulant, nu=1,
                       x_interval = None):
    
    def higher_order_egdeworth_term(y):
        norm_y = (y-mean)/sigma
        skew_term = np.divide(third_cumulant*He3(norm_y), 6*sigma**3)
        kurtosis_term = np.divide(fourth_cumulant*He4(norm_y), 24*sigma**4)
        skew_squared_term =\
            np.divide(He6(norm_y)*third_cumulant**2, 72*sigma**6)
        return (skew_term+kurtosis_term+skew_squared_term)-nu
    
    if x_interval == None:
        sol = optimize.root_scalar(higher_order_egdeworth_term, method='brentq',\
                                       bracket = [mean, mean+10000*sigma])
    else:
        sol = optimize.root_scalar(higher_order_egdeworth_term, method='brentq',\
                                       bracket = x_interval)

    
    return sol.root#The root is the position of when deviation occurs

#The pdf of the exponentially modifed Gaussian, from the sample mean, std 
#and skewness, see
#https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution
# and 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponnorm.html
def expo_mod_gauss_params_guess(m, s, gamma):
    
    def decay_eqn(tau, s, gamma):
        return gamma-np.divide(2*tau**3, (s**2+tau**2)**(3/2))
    
    sol = optimize.root_scalar(decay_eqn, method='brentq',\
                                   bracket = [-10, 2], args = (s, gamma))
    tau = sol.root
    sigma = np.sqrt(s**2-tau**2)
    mean = m-tau
    K = tau/sigma
    
    return mean, sigma, K


#I HAVE NO IDEA WHERE THIS CAME FROM DO NOT USE!!!!!!!!
def vincent_far_tail_fit_old(N,m,phi_i):
    v0 = (m**2)/(48*PI**2)
    ve = v0*2#As inflation ends at phi_end=sqrt(2)
    v = v0*phi_i**2
    
    #Calculating the terms individually for clarity
    constant = (np.sqrt(2)*PI**2)/128
    differance = (np.exp(-1/v)*v**1.5-np.exp(-1/ve)*ve**1.5)/v0**2
    N_dependance = np.divide(np.exp((PI**2)/(16*v0*N)-v0*N/4), N**3)
    log_term = 1/np.log(PI/(2*v0*N))
    print('This is old Vincent;s far tail and is probably wrong!!!!!!!!!!!!')
    
    return constant*differance*N_dependance*log_term
    
#From Vincent's 3.27
def vincent_far_tail_fit(N, m, phi_i, phi_end = 2**0.5):
    v0 = (m**2)/(48*PI**2)
    ve = v0*phi_end**2
    v = v0*phi_i**2
    
    #Calculating the terms individually for clarity
    constant = np.divide(PI**1.5, 2*np.sqrt(v0)*special.gamma(-0.25)**2)
    differance = (np.exp(-1/v)*v**1.5-np.exp(-1/ve)*ve**1.5)
    N_dependance = np.divide(np.exp(-0.25*v0*N),N**1.5)

    return constant*differance*N_dependance

#Taken from Vincent's notes looking at the poles of the chracteristic function
# Eq. 4.30
def vincent_near_tail_fit(N, m, phi_i, phi_end = 2**0.5,\
                          numerical_integration = False):
    v0 = (m**2)/(48*PI**2)
    v = v0*phi_i**2
    N_cl = 0.25*phi_i**2-0.25*phi_end**2
    ve = v0*phi_end**2
    if numerical_integration == False:
        
        #Calculating the terms individually for clarity
        constant = (np.sqrt(2)*PI**2)/(128*v0**2)
        exp = np.exp(-0.25*v0*N)
        
        frac_expo_i = np.divide(PI**2, 16*v0*(N+N_cl+1))
        fraction_i = np.divide(np.exp(frac_expo_i-1/v)*v**1.5, (N+N_cl+1)**3)
        
        frac_expo_end = np.divide(PI**2, 16*v0*(N-N_cl+1))
        fraction_end = np.divide(np.exp(frac_expo_end-1/ve)*ve**1.5,\
                                (N-N_cl+1)**3)
        
        return constant*exp*(fraction_i - fraction_end)
    elif numerical_integration == True:
        a1 = 0.25*v0*N + 0.0625*(v+ve)
        a2 = 0.25*v0*N + 0.0625*(3*ve-v)
        def G(x, a):
            return np.exp(0.25*PI*x-a*x**2)*x**(5/2)
        Ga1_int,_ = integrate.quad(G, 3, np.infty, args = (a1))
        Ga2_int,_ = integrate.quad(G, 3, np.infty, args = (a2))
        
        
        first_term = np.exp(-1/v)*Ga1_int*v**1.5
        second_term = np.exp(-1/ve)*Ga2_int*ve**1.5
        return v0*np.exp(-0.25*v0*N)*(first_term-second_term)/(32*PI)
    else:
        raise ValueError('parameter numerical_integration must be Boolean') 


def vincent_small_v_UV_pole(m, phi_UV, phi_end = None, W_approx = True):
    v0 = (m**2)/(48*PI**2)
    v_UV = v0*phi_UV**2
    if phi_end is None:#As inflation ends at phi_end=sqrt(2)
        v_end = v0*2
    else:
        v_end = v0*phi_end**2

    if W_approx == True:#Use the (x) ~ ln(x) for large x
        fraction1 = np.divide(2*v0, v_UV*v_end)
        fraction2 = np.divide(v_UV*v_end, v_UV-v_end)
        fraction3 = np.divide(v_UV-v_end, (v_end**1.5)*(v_UV**0.5))
        Lambda = fraction1*(1)
        
    elif W_approx == False:#Use the lambert function
        fraction1 = np.divide(2*v0, v_UV-v_end)
        fraction2 = np.divide(v_UV-v_end, (v_end**1.5)*(v_UV**0.5))
        expo = np.exp(1/v_end- v_UV)
        Lambda = v0/4 + fraction1*special.lambertw(2*fraction2*expo)
        Lambda = np.real(Lambda)
        
    return Lambda
    