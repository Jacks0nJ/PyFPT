
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
#import scipy as sp
#import math
#M_PL = 2.435363*10**18 old value
M_PL = 1.0
PI = np.pi#so can just use PI as needed
EULER_GAMMA = 0.5772156649

'''
Functions from Vennin 2015
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

def large_mass_pdf(bin_centres, phi_i, phi_end, V):
    def chi(t):
        return chaotic_inflation_characteristic_function(t,phi_i,\
                                                                  phi_end,V)
    
    
    #Use integral symmetric to simplfy to only do the positive half, then double
    #Remember they use a different fourier 2pi convention to be, hence the extra one
    v = V(phi_i)/(24*(PI**2)*(M_PL**4))
    
    #Stolen from Chris' quadratic code, no idea why this is a thing!
    if v<10:
        t0max = 1000.
    if v<0.1:
        t0max = 6000.
    if v<0.04:
        t0max = 10.**7
    PDF_analytical_test =\
        np.array([2*continuous_ft(N,chi, component = 'real',t_lower = 0,\
                t_upper = t0max)/(2*PI)**0.5 for N in bin_centres])
    return PDF_analytical_test

'''
Importance sampling equations
'''


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


#THIS IS A MESS!!!!!!!!!!!!!!
#Alternative method for calculating the errors of the histogram bars by using
#the simplified jackknife analysis. The data is sub-sampled into many
#histograms with the same bins. This way a distribution for the different
#heights can be done. Takes the bins used as general argument
#Arguments must be numpy arrays
def histogram_weighted_bin_errors_jackknife(data_input, weights_input, bins,\
                                            num_sub_samps, density = True):
    #Make an array of random indexs
    indx = np.arange(0, len(data_input), 1)
    np.random.shuffle(indx)
    #This allows the data to be randomised and keep the weights matched
    data = data_input[indx]
    weights = weights_input[indx]
    num_bins = len(bins)-1#bins is an array of the side, so one less

    height_array = np.zeros((num_bins, num_sub_samps))#Storage
        

    #Next organise into subsamples
    data =\
        np.reshape(data, (int(data.shape[0]/num_sub_samps), num_sub_samps))
    weights =\
        np.reshape(weights, (int(weights.shape[0]/num_sub_samps), num_sub_samps))
    #Find the heights of the histograms, for each sample
    
                
    for i in range(num_sub_samps):
        '''
        height_array[:, i], _ = np.histogram(data[:,i], weights = weights[:,i],\
                               bins = bins, density = density)
        '''
        _ , ws = histogram_data_in_bins(data[:,i], weights[:,i], bins)
        heights = np.sum(ws, axis=0)
        if density == True:
            norm = len(data[:,i])*np.diff(bins)[0]
        else:
            norm=1
        height_array[:, i] = heights/norm


    error = np.zeros(num_bins)
    sqrt_sub_samples = np.sqrt(num_sub_samps)
    
    #Now can find the standard deviation of the heights, as the bins are the
    #same. Then divide by the sqaure root of the number of samples by jackknife
    #and you have the error
    for j in range(num_bins):
        bars = height_array[j,:]
        if np.any([bars>0]):
            #Used to be just np.std(bars)
            error[j] = np.std(bars[bars>0])/sqrt_sub_samples
        else:
            error[j] = 0#Remove any empty histogram bars
            
    return error

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
    

    
#Returns the normalisation factor for a histogram, including one with weights
def histogram_analytical_normalisation(bins, num_sims):
    return num_sims*np.diff(bins)[0]
   
    
'''
Log-normal equations
'''   

#The data provided needs to be raw data - the log is then taken in the function
#uses the maximum likelihood Shen 2006 Statist. Med. 2006; 25:3023–3038
def log_normal_mean(data, position = None):
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
                     
    mean = np.exp(data_log_mean+0.5*data_log_std**2)
        
    return mean
    
def log_normal_height(w, position = None):
    return len(w)*log_normal_mean(w, position = position)

#The most basic way to estimate the error, assuming symmetric errors
#SHOULD THIS USE n-1 or n IN THE STD??????????????????????????
def log_normal_errors(ws, Z_alpha=1):
    log_w = np.log(ws)
    log_var = np.var(log_w, ddof = 1)#unbiased variance
    log_mean = np.mean(log_w)
    n = len(ws)
    log_err = Z_alpha*np.sqrt(log_var/n+(log_var**2)/(2*n-2))
    upper_err = n*np.exp(log_mean+log_var/2)*(np.exp(log_err)-1)
    lower_err = n*np.exp(log_mean+log_var/2)*(1-np.exp(-log_err))
    return np.array([lower_err, upper_err])

    
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

