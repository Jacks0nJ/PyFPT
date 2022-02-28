# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:41:37 2021

@author: user
"""

import numpy as np
import pandas as pd

import scipy.stats as sci_stat
from timeit import default_timer as timer
import multiprocessing as mp
from multiprocessing import Process, Queue
import scipy.optimize


import importance_sampling_sr_cython12 as is_code
import inflation_functions_e_foldings as cosfuncs
import is_data_analysis as isfuncs
import stochastic_inflation_cosmology as sm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mpl_style
plt.style.use(mpl_style.style1)

#M_PL = 2.435363*10**18 old value
M_PL = 1.0# Using units of M_PL
PI = np.pi
#m = 10**(-6)*M_PL#Based on John McDonald's calculations in PHYS461
m = 0.001*M_PL#4*PI*6**0.5

###Intial conditions and tolerances
N_starting = 10#In some sense, this should techically be negative
a_i = 1
phi_end = M_PL*2**0.5
phi_i = M_PL*(4*N_starting+2)**0.5#M_PL*(4*N_starting+2)**0.5
phi_r = 100*phi_i
N_cut_off = 300
N_f = 100
dN = 0.02*m#Assuming std(N) is proportional to m, was dN=0.02m
num_sims = int(200000/mp.cpu_count())
num_sims_used = int(num_sims*mp.cpu_count())
num_bins = 50
num_sub_samples = 20


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

min_bin_size = 400
fit_threshold = 100
#Minimum number of std before exponential tail can be fit
min_tail_start = 4


comp_language = 'Cython'
include_errors = 'yes'
tail_analysis = False
emg_fitting = False#'chi_squared'
method_comarison = False
edgeworth_series = False
manual_norm = True
w_hist = False
save_results = True
save_raw_data = True
log_normal = True
publication_plots = True
contour = False
fontsize = 20
include_std_w_plot = True
save_other_plots = False
count_refs = False
scater_density_plot = True

wind_type = 'diffusion'
bias = 3


if (m == 2 or m==1) and phi_i==phi_r:
    kazuya_pdf = True
    vincent = False
    
elif m>0.6:
    vincent = True
    kazuya_pdf = False
    if log_normal == True:
        print('Are you sure you want to use the lognormal approach??')
else:
    kazuya_pdf = False
    vincent = False
    
if m<0.2 and log_normal == False:
    print('Are you sure you want to use the naive approach??')

def V(phi):
    V = 0.5*(m*phi)**2
    return V

def V_dif(phi):
    V_dif = (m**2)*phi
    return V_dif

def V_ddif(phi):
    V_ddif = (m**2)
    return V_ddif

def bias_func(phi, bias = bias, V = V, PI = PI, phi_i = phi_i):
    if phi<0.9*phi_i:
        H_squared = np.divide(V(phi), 3*M_PL**2)
        H = np.sqrt(H_squared)
        return bias*H/(2*PI)
    elif phi>=0.9*phi_i:
        H_squared = np.divide(V(phi), 3*M_PL**2)
        H = np.sqrt(H_squared)
        return -0*bias*H/(2*PI)
    else:
        raise ValueError('Incorrect field value')
        
def bias_func_sigmoid(phi, bias = bias, V = V, PI = PI, phi_i = phi_i):
    H_squared = np.divide(V(phi), 3*M_PL**2)
    H = np.sqrt(H_squared)
    return bias*(H/(PI))*(1/(1+np.exp(10*(phi-phi_i)))-0.5)

def classical_end_cond(matrices, N, phi_end_infl = phi_end):
    cond = False
    if matrices[0,0] <= phi_end_infl:
        cond = True  
    return cond


    
phi_sqaured_cosmo = \
    sm.Stochastic_Inflation(V, V_dif, V_ddif,\
                            classical_end_cond, a_i)
'''
#Running the simulation many times
'''


start = timer()


def multi_processing_func(phi_i, phi_r, phi_end, N_i, N_f, dN, bias,\
                          num_sims, queue_Ns, queue_ws, queue_refs):
    results =\
            is_code.many_simulations_importance_sampling(phi_i, phi_r, phi_end,\
                     N_i, N_f, dN, bias, num_sims, V, V_dif, V_ddif,\
                         bias_type = 'diffusion', count_refs = count_refs)
    Ns = np.array(results[0][:])
    ws = np.array(results[1][:])
    queue_Ns.put(Ns)
    queue_ws.put(ws)
    if count_refs == True:
        num_refs = np.array(results[2][:])
        queue_refs.put(num_refs)

if __name__ == "__main__":
    queue_Ns = Queue()
    queue_ws = Queue()
    queue_refs = Queue()
    cores = int(mp.cpu_count()/1)

    print('Number of cores used: '+str(cores))
    processes = [Process(target=multi_processing_func,\
                args=(phi_i, phi_r,  phi_end, 0.0, N_f, dN, bias,\
                num_sims, queue_Ns, queue_ws, queue_refs)) for i in range(cores)]
        
    for p in processes:
        p.start()
    
    #for p in processes:
     #   p.join()
    
    Ns_array = np.array([queue_Ns.get() for p in processes])
    ws_array = np.array([queue_ws.get() for p in processes])
    if count_refs == True:
        num_refs_array = np.array([queue_refs.get() for p in processes])
    end = timer()
    print(f'The simulations took: {end - start}')
    
#As there num_sims on the cores
num_sims = cores*num_sims
#Combine into columns into 1
sim_N_dist  = Ns_array.flatten()
w_values = ws_array.flatten()


#Sort in order of increasing Ns
sort_idx = np.argsort(sim_N_dist)
sim_N_dist = sim_N_dist[sort_idx]
w_values = w_values[sort_idx]

if count_refs == True:
    num_refs = num_refs_array.flatten()
    num_refs = num_refs[sort_idx]


estimated_mean = cosfuncs.importance_sampling_mean(sim_N_dist, w_values)
estimated_st = cosfuncs.importance_sampling_st(sim_N_dist, w_values)
    
    

#Checking if multipprocessing error occured, by looking at correlation
pearson_corr = np.corrcoef(sim_N_dist, np.log10(w_values))
pearson_corr = pearson_corr[0,1]

if pearson_corr > -0.55:#Data is uncorrelated
    print('Multiprocessing error occured, ignoring')
    print('Peasron correlation coefficent is ' + str(pearson_corr))

'''
Truncating the data
'''
sim_N_old =  sim_N_dist 
w_values_old = w_values
sim_N_dist, w_values = cosfuncs.histogram_data_truncation(sim_N_dist,\
                          N_f, weights=w_values,\
                          num_sub_samples = num_sub_samples)
num_sims_used = len(sim_N_dist)
if count_refs == True:
    num_refs = num_refs[:num_sims_used]
    ref_ending_prob = np.sum(w_values[num_refs>0])/num_sims_used
    ref_ending_prob_analytical = 1-cosfuncs.ending_probability(phi_i, phi_end,\
                                                             phi_r, V)



    

'''
#Post processesing
'''


sim_N_mean = cosfuncs.importance_sampling_mean(sim_N_dist, w_values)
sim_N_var = cosfuncs.importance_sampling_var(sim_N_dist, w_values)
sim_N_skew = cosfuncs.importance_sampling_skew(sim_N_dist, w_values)
sim_N_kurtosis = cosfuncs.importance_sampling_kurtosis(sim_N_dist, w_values)

sim_3rd_cumulant =\
    cosfuncs.importance_sampling_3rd_cumulant(sim_N_dist, w_values)
sim_4th_cumulant =\
    cosfuncs.importance_sampling_4th_cumulant(sim_N_dist, w_values)

sim_mean_error = cosfuncs.jackknife(sim_N_dist, num_sub_samples,\
                cosfuncs.importance_sampling_mean, weights = w_values)
sim_var_error = cosfuncs.jackknife(sim_N_dist, num_sub_samples,\
                cosfuncs.importance_sampling_var, weights = w_values)
sim_skew_error = cosfuncs.jackknife(sim_N_dist, num_sub_samples,\
                cosfuncs.importance_sampling_skew, weights = w_values)
sim_kurtosis_error = cosfuncs.jackknife(sim_N_dist, num_sub_samples,\
                cosfuncs.importance_sampling_kurtosis, weights = w_values)


#Expected values in the near the classical limit
analytic_N_var =\
    phi_sqaured_cosmo.delta_N_squared_sto_limit(phi_i, phi_end)
analytic_N_st = np.sqrt(analytic_N_var)
analytic_N_mean = phi_sqaured_cosmo.mean_N_sto_limit(phi_i, phi_end)
analytic_N_skew = phi_sqaured_cosmo.skewness_N_sto_limit(phi_i, phi_end)
analytic_N_kurtosis = phi_sqaured_cosmo.kurtosis_N_sto_limit(phi_i, phi_end)
analytic_N_4th_cmoment =\
    cosfuncs.fourth_central_moment_N_sto_limit(V,V_dif, V_ddif, phi_i, phi_end)
analytic_power_spectrum = phi_sqaured_cosmo.power_spectrum_sto_limit(phi_i)
eta_criterion = phi_sqaured_cosmo.classicality_criterion(phi_i)

        
N_star = analytic_N_mean + 4*analytic_N_st

analytic_gauss_deviation_pos =\
    cosfuncs.gaussian_deviation(analytic_N_mean, analytic_N_var**0.5,\
    analytic_N_skew*analytic_N_var**1.5,\
    analytic_N_4th_cmoment-3*analytic_N_var**2, nu=fit_threshold/100)



#
#Now analysisng creating the PDF data
#

if bias>0.2:
    bin_centres,heights,errors,num_sims_used, bin_edges_untruncated =\
        isfuncs.data_points_pdf(sim_N_dist, w_values, num_sub_samples,\
        bins=num_bins, include_std_w_plot = include_std_w_plot,\
        min_bin_size = min_bin_size, log_normal = log_normal,\
        num_sims = num_sims, log_normal_method='ML', w_hist_num = 10,\
        p_value_plot = True)
#I'm not convinced 
elif bias<=0.2:
    bin_centres,heights,errors,num_sims_used,_ =\
        isfuncs.data_points_pdf(sim_N_dist, w_values, num_sub_samples,\
        bins=num_bins, min_bin_size = min_bin_size, log_normal = False,\
        num_sims = num_sims)
    #If using log normal, need to make the errors asymmetric for later
    #data analysis
    if log_normal == True:
        errors_new = np.zeros((2,len(errors)))
        errors_new[0,:] = errors
        errors_new[1:] = errors
        errors = errors_new
        
    
bin_centres_analytical =\
        np.linspace(bin_centres[0], bin_centres[-1],2*num_bins)
        
if m>0.6:#Less than this Pattison 2017 breaks down:
    start = timer()
    PDF_analytical_test = isfuncs.large_mass_pdf(bin_centres_analytical,phi_i,phi_end,V)
    end = timer()
    print(f'The analytical answer took: {end - start}') 
    


    
'''
Saving data
'''
if save_results == True:
    data_dict = {}
    data_dict['bin_centres'] = bin_centres
    data_dict['PDF'] = heights
    if log_normal == False:
        data_dict['errors'] = errors
    elif log_normal == True:
        data_dict['errors_lower'] = errors[0,:]
        data_dict['errors_upper'] = errors[1,:]
        
    
    data_pandas_results = pd.DataFrame(data_dict)
    
    my_file_name = 'results_for_N_'+str(N_starting)+'_dN_'+str(dN)+'_m_'+('%s' % float('%.3g' % m))+\
    '_iterations_'+str(num_sims)+'_bias_'+str(bias)+'_phi_UV_'+str(phi_r/phi_i)+'phi_i'+'.csv'
    #Saving to a directory for the language used
    if publication_plots == True:
        data_pandas_results.to_csv('for_paper/'+my_file_name)
        #Remembering to remove column numbering
        sim_data = pd.read_csv('for_paper/'+my_file_name,\
                               index_col=0)
    else:
        data_pandas_results.to_csv(comp_language+'_results/'+my_file_name)
        #Remembering to remove column numbering
        sim_data = pd.read_csv(comp_language+'_results/'+my_file_name,\
                               index_col=0)
    #Now read this data back  
    bin_centres = np.array(sim_data['bin_centres'])
    heights = np.array(sim_data['PDF'])
    if log_normal == False:
        errors = np.array(sim_data['errors'])
    elif log_normal == True:
        errors = np.zeros((2, len(heights)))
        errors[0,:] = np.array(sim_data['errors_lower'])
        errors[1,:] = np.array(sim_data['errors_upper'])
    
    
if save_raw_data == True:
    data_dict_raw = {}
    data_dict_raw['N'] = sim_N_dist
    if bias>0:
        data_dict_raw['w'] = w_values
    
    data_pandas_raw = pd.DataFrame(data_dict_raw)
    
    raw_file_name = 'raw_data_for_N_'+str(N_starting)+'_dN_'+str(dN)+'_m_'+('%s' % float('%.3g' % m))+\
    '_iterations_'+str(num_sims)+'_bias_'+str(bias)+'_phi_UV_'+str(phi_r/phi_i)+'phi_i'+'.csv'
    #Saving to a directory for the language used
    if publication_plots == True:
        data_pandas_raw.to_csv('for_paper/'+raw_file_name)
        #Remembering to remove column numbering
        raw_data = pd.read_csv('for_paper/'+raw_file_name,\
                               index_col=0)
    else:
        data_pandas_raw.to_csv(comp_language+'_results/'+raw_file_name)
        #Remembering to remove column numbering
        raw_data = pd.read_csv(comp_language+'_results/'+raw_file_name,\
                               index_col=0)
    #Now read this data back  
    sim_N_dist = np.array(raw_data['N'])
    if bias>0:
        w_values = np.array(raw_data['w'])

    
'''
raw_data_08 = pd.read_csv('for_paper/diffusion_dom/'+\
                          'raw_data_for_N_10_dN_0.02_m_1.0_iterations_200000_bias_0.8_phi_UV_1.0phi_i.csv',\
                       index_col=0)

#Now read this data back  
sim_N_dist = np.array(raw_data_08['N'])

w_values  = np.array(raw_data_08['w']) 
        
    
raw_data_04 = pd.read_csv('for_paper/diffusion_dom/'+\
                          'raw_data_for_N_10_dN_0.02_m_1.0_iterations_200000_bias_0.4_phi_UV_1.0phi_i.csv',\
                       index_col=0)

#Now read this data back  
sim_N_dist_04 = np.array(raw_data_04['N'])

w_values_04  = np.array(raw_data_04['w'])
    
raw_data_direct = pd.read_csv('for_paper/diffusion_dom/'+\
                              'raw_data_for_N_10_dN_0.02_m_1.0_iterations_200000_bias_0.0_phi_UV_1.0phi_i.csv',\
                       index_col=0)

#Now read this data back  
sim_N_dist_direct = np.array(raw_data_direct['N'])

w_values_direct = np.zeros(len(sim_N_dist_direct))+1
    

bin_centres,heights,errors,num_sims_used,bin_edges_untruncated_08_bias =\
    isfuncs.data_points_pdf(sim_N_dist, w_values, num_sub_samples,\
    bins=num_bins, include_std_w_plot = include_std_w_plot,\
    min_bin_size = min_bin_size, log_normal = log_normal,\
    num_sims = num_sims, log_normal_method='ML', w_hist_num = 10,\
    p_value_plot = True)


bin_centres_04_bias,heights_04_bias,errors_04_bias,num_sims_used,_ =\
    isfuncs.data_points_pdf(sim_N_dist_04, w_values_04, num_sub_samples,\
    bins=bin_edges_untruncated_08_bias, include_std_w_plot = include_std_w_plot,\
    min_bin_size = min_bin_size, log_normal = log_normal,\
    num_sims = num_sims, log_normal_method='ML', w_hist_num = 10,\
    p_value_plot = True)


bin_centres_direct,heights_direct,errors_direct,num_sims_used,_ =\
    isfuncs.data_points_pdf(sim_N_dist_direct, w_values_direct, num_sub_samples,\
    bins=bin_edges_untruncated_08_bias, min_bin_size = min_bin_size, log_normal = False,\
    num_sims = num_sims)

'''


'''
Fitting models to the tail
'''

if emg_fitting == 'chi_squared':
    def log_of_exponnorm_pdf(x, K, mean, sigma):
        return np.log(sci_stat.exponnorm.pdf(x, K, mean, sigma))
    EMG_params, cv =\
    scipy.optimize.curve_fit(log_of_exponnorm_pdf, bin_centres,\
                             np.log(heights),\
                            p0 = (sim_N_skew, sim_N_mean, sim_N_var**0.5))
elif emg_fitting == 'stats':
     emg_mu, emg_sigma, emg_K =\
         cosfuncs.expo_mod_gauss_params_guess(sim_N_mean, sim_N_var**0.5,\
                                              sim_N_skew)

        
        
if tail_analysis == True:
    #The classical prediction of Gaussian with skewness and
    #kurtosis
    if edgeworth_series == True:
        classical_prediction =\
            cosfuncs.pdf_gaussian_skew_kurtosis(bin_centres, analytic_N_mean,\
            analytic_N_var**0.5, analytic_N_skew*analytic_N_var**1.5,\
            analytic_N_4th_cmoment-3*analytic_N_var**2)
    else:
        classical_prediction = sci_stat.norm.pdf(bin_centres, analytic_N_mean,\
                                      analytic_N_st)
    percentage_diff =\
        100*np.divide(np.abs(heights-classical_prediction),classical_prediction)
        
    #Now finding when the deviation from prediction first occurs
    tail_start_idx = len(heights)+1
    for i in range(len(heights)):
        #The tail is if this differance is greater than fit_threshold
        if percentage_diff[i]>fit_threshold and bin_centres[i]>analytic_N_mean+min_tail_start*analytic_N_var**0.5:
            tail_start_idx = i
            break;
    
    #If there was a value with deviation greater than fit_threshold and a has
    #a few data points
    if len(heights)-tail_start_idx>5:
        print(str(len(heights)-tail_start_idx)+' data points deviated by more'+
              ' than '+str(fit_threshold)+'% from classical prediction: fitting exponential')
        heights_tail =  heights[tail_start_idx:]
        bin_centres_tail = bin_centres[tail_start_idx:]
        if log_normal == True:
            errors_tail = errors[:,tail_start_idx:]
        else:
            errors_tail = errors[tail_start_idx:]
        
        def expo_model(x, a, b):
            return a*np.exp(b*x-b*N_starting)
        
        def log_of_expo_model(x, log_a, b):
            return log_a+b*(x-N_starting)
        
        #Using data points to make an initial parameter guess
        b_guess = np.log(heights_tail[0]/heights_tail[-1])/\
            (bin_centres_tail[0]-bin_centres_tail[-1])
        log_a_guess =\
            np.log(heights_tail[0])-b_guess*(bin_centres_tail[0]-N_starting)
        log_expo_params_guess = (log_a_guess, b_guess)
        
        #Now fitting the expo model to the tail
        if include_errors == 'no':#Not including the errors in the fit
            expo_fit_params, cv =\
                scipy.optimize.curve_fit(log_of_expo_model, bin_centres_tail,\
                                         np.log(heights_tail),\
                                             p0 = log_expo_params_guess)
            a_expo = np.exp(expo_fit_params[0])
            b_expo = expo_fit_params[1]
        elif include_errors == 'yes':#Including errors in the fit
            #As using log of data, need to explictly caclulate the errors
            if log_normal == True:
                errors_lower = errors_tail[0,:]
                    
                errors_upper = errors_tail[1,:]
                    
            else:
                errors_lower = errors_tail
                errors_upper = errors_tail
                
            log_error_lower =\
                np.log(heights_tail)-np.log(heights_tail-errors_lower)
                
            log_error_upper =\
                np.log(heights_tail+errors_upper)-np.log(heights_tail)
                    

            #Taking the average for now, as scipy can only have one error   
            log_error_tail = (log_error_lower+log_error_upper)/2
                    
            expo_fit_params, cv =\
                scipy.optimize.curve_fit(log_of_expo_model, bin_centres_tail,\
                                         np.log(heights_tail),\
                                         sigma = log_error_tail,\
                                         p0 = log_expo_params_guess)
            a_expo = np.exp(expo_fit_params[0])
            b_expo = expo_fit_params[1]
            
            #error in parameter estimation
            expo_fit_params_errs = np.sqrt(np.diag(cv))
            #Doing explicit error calculation
            a_expo_err = a_expo*(np.exp(expo_fit_params_errs[0])-1)
            b_expo_err = expo_fit_params_errs[1]

        '''
        #Modified expo fit
        def modified_expo_model(x, a, b, c):
            return a*np.exp(b*x)/(x**c)
        
        #Using data points to make an initial parameter guess
        c_guess_numerator = np.log(heights_tail[0]/heights_tail[-1])/\
                        (bin_centres_tail[0]-bin_centres_tail[-1])-\
                            np.log(heights_tail[0]/heights_tail[1])/\
                        (bin_centres_tail[0]-bin_centres_tail[1])
        c_guess_denominator = np.log(bin_centres_tail[0]/bin_centres_tail[-1])/\
                        (bin_centres_tail[0]-bin_centres_tail[-1])-\
                            np.log(bin_centres_tail[0]/bin_centres_tail[1])/\
                        (bin_centres_tail[0]-bin_centres_tail[1])
                        
        c_m_guess = c_guess_numerator/c_guess_denominator
        
        b_m_guess = np.log(heights_tail[0]/heights_tail[-1])/\
            (bin_centres_tail[0]-bin_centres_tail[-1])+\
                c_m_guess*np.log(bin_centres_tail[0]/bin_centres_tail[-1])/\
            (bin_centres_tail[0]-bin_centres_tail[-1])
            
        a_m_guess = heights_tail[0]*np.exp(-b_m_guess*bin_centres_tail[0])*\
            (bin_centres_tail[0]**c_m_guess)
            
        log_m_expo_params_guess = (np.log(a_m_guess), b_m_guess, c_m_guess)
        
        #Now fitting the expo model to the tail
        m_expo_fit_params, cv =\
            scipy.optimize.curve_fit(modified_expo_model, bin_centres_tail,\
                                     np.log(heights_tail), log_m_expo_params_guess)
        a_m_expo = np.exp(m_expo_fit_params[0])
        b_m_expo = m_expo_fit_params[1]
        c_m_expo = m_expo_fit_params[2]
        '''
        
    elif len(heights)-tail_start_idx>1:
        print('Only '+str(len(heights)-tail_start_idx)+' data points deviated by more'+
              ' than '+str(fit_threshold)+'% from classical prediction: no enough to fit data')
        tail_analysis = False
    else:
        print('No deviated from classical prediction greater than '+str(fit_threshold)+'%')
        tail_analysis = False
    


'''
 Plotting
'''



_,bins,_ = plt.hist(sim_N_dist, num_bins, density=True,\
         label='{0}'.format('Unweighted bins') )


if m>0.6:#Less than this it breaks down:
    best_fit_line2 = PDF_analytical_test
    dist_fit_label2 = r'Pattison 2017'
    plt.plot(bin_centres_analytical, best_fit_line2,\
             label='{0}'.format(dist_fit_label2))
else:
    best_fit_line2 = sci_stat.norm.pdf(bin_centres_analytical, analytic_N_mean,\
                                      analytic_N_st)
    dist_fit_label2 = r'Gaussian $\sqrt{\delta \mathcal{N}^2}$='+str(round(analytic_N_st,4))
    plt.plot(bin_centres_analytical, best_fit_line2, label='{0}'.format(dist_fit_label2))
'''
plt.axvline(sim_N_mean, color='k', linestyle='dashed', linewidth=2,\
            label='{0}'.format( r'$\langle\mathcal{N}\rangle$=' +\
            ('%s' % float('%.6g' % sim_N_mean)) + r'$\pm$' +\
            ('%s' % float('%.2g' % sim_mean_error))) ))
'''


if bias != 0:
    '''
    best_fit_line3 = sci_stat.norm.pdf(bins, sim_N_mean,\
                                  sim_N_mean)
    dist_fit_label3 = 'Importance sampling est'
    plt.plot(bins, best_fit_line3, label='{0}'.format(dist_fit_label3))
    '''
    histogram_name = 'N_distribution_for_' + 'importance_sample_near_' +\
        str(N_starting)+'_dN_' + str(dN) + '_m_'+('%s' % float('%.3g' % m)) +'_Is_shift_'+str(bias)+ '_iterations_' +\
            str(num_sims) +'_NOT_wighted.pdf'
    plt.title(r'bias=' + str(bias) +', '+str(num_sims) + ' sims, ' + r'dN=' +\
              ('%s' % float('%.3g' % dN))+\
              r', $m$=' + ('%s' % float('%.3g' % m)) )
else:
    histogram_name = 'N_distribution_for_' + 'N_' + str(N_starting)+'_dN_' +\
        str(dN) + '_m_'+('%s' % float('%.3g' % m)) + '_iterations_' +\
            str(num_sims)+'.pdf'
    plt.title(str(num_sims) + ' sims, ' + r'dN='+('%s' % float('%.3g' % dN))+\
              r', $m$=' + ('%s' % float('%.3g' % m)) )
plt.xlabel(r'$\mathcal{N}$')
plt.ylabel(r'$P(\mathcal{N})$')
plt.legend()
#Including if I have used importance sampling
#Saving to a directory for the language used
if save_other_plots == True:
    plt.savefig(comp_language+'_results/'+histogram_name,transparent=True)
plt.show()
plt.clf()


if bias != 0:
    _,_,_ = plt.hist(sim_N_dist, num_bins, weights = w_values,\
                             density=True, label='{0}'.format('Weighted bins') ) 
    histogram_name = 'N_distribution_for_' + 'IS_near_' +\
    str(N_starting)+'_dN_' + str(dN) + '_m_'+('%s' % float('%.3g' % m)) +'_Is_shift_'+str(bias)+ '_iterations_' +\
        str(num_sims_used)+'_wighted.pdf'
    if m<=0.6:
        dist_fit_label2 = r'Analytical $\sqrt{\delta \mathcal{N}^2}$='+str(round(analytic_N_st,4))
    else:
        dist_fit_label2 = r'Pattison 2017'
    plt.plot(bin_centres_analytical, best_fit_line2, label='{0}'.format(dist_fit_label2))
    plt.title(r'bias=' + str(bias) +', '+str(num_sims_used) + ' sims, ' +\
              r'dN=' + ('%s' % float('%.3g' % dN)) +\
              r', $m$=' + ('%s' % float('%.3g' % m)) )
    plt.xlabel(r'$\mathcal{N}$')
    plt.ylabel(r'$\mathcal{N}$')
    plt.legend()
    #Including if I have used importance sampling
    #Saving to a directory for the language used
    if save_other_plots == True:
        plt.savefig(comp_language+'_results/'+histogram_name,transparent=True)
    plt.show()
    plt.clf()

    
if bias != 0:
    #Plotting the weights, and number of reflections if appropriate
    scatter_name = '_dN_' + str(dN) +'_m_'+\
            ('%s' % float('%.3g' % m)) +'_phi_UV_'+\
            str(phi_r/phi_i)+ '_m_'+ ('%s' % float('%.3g' % m)) +\
            '_iterations_' + str(num_sims)+'_bias_' + str(bias)
    if count_refs == True:
        scatters = plt.scatter(sim_N_dist, np.log10(w_values), c=num_refs)
        cbar = plt.colorbar(scatters)
        cbar.set_label(r'# reflections')
        scatter_title = r'$\phi_{UV}=$ '+('%s' % float('%.3g' % phi_r))+r', $dN=$'\
                  + ('%s' % float('%.3g' % dN))+ r', $m$=' +\
                      ('%s' % float('%.3g' % m)) 
        scatter_name = 'weights_with_counted_reflections' + scatter_name + '.png'
    elif contour == True:
        h, xedges, yedges, _ =\
            plt.hist2d(sim_N_dist, np.log10(w_values), (50, 50))
        plt.clf()
        xedges_centre =\
            np.array([(xedges[i]+xedges[i+1])/2 for i in range(len(xedges)-1)])
        yedges_centre =\
            np.array([(yedges[i]+yedges[i+1])/2 for i in range(len(yedges)-1)])
        X, Y = np.meshgrid(xedges_centre, yedges_centre)
        fig, ax = plt.subplots()
        CS = ax.contour(X, Y, h, (20,100,1000), colors='k')
        ax.clabel(CS, fontsize=9, inline=True)
        scatter_name = 'weights_contour' + scatter_name + '.png'
    elif scater_density_plot == True:
        plt.hist2d(sim_N_dist, np.log10(w_values), (50, 50), norm=LogNorm())
        cbar = plt.colorbar()
        cbar.set_label(r'# Data Points')
        scatter_title = r'bias = '+('%s' % float('%.3g' % bias))+r', $dN=$' +\
          ('%s' % float('%.3g' % dN))+ r', $m$=' + ('%s' % float('%.3g' % m)) 
        scatter_name = 'weights_2D_histogram' + scatter_name + '.pdf'
    else:
        plt.scatter(sim_N_dist,np.log10(w_values))
        scatter_title = r'bias = '+('%s' % float('%.3g' % bias))+r', $dN=$' +\
          ('%s' % float('%.3g' % dN))+ r', $m$=' + ('%s' % float('%.3g' % m)) 
        scatter_name = 'log_of_weights_of_IS' + scatter_name + '.png'
              
    if publication_plots==True:
        plt.xlabel(r'$\mathcal{N}$', fontsize = fontsize)
        plt.ylabel(r'${\rm log}_{10}(w)$', fontsize = fontsize)
        plt.margins(tight=True)
        plt.savefig('for_paper/'+scatter_name, transparent=True, dpi=800)
    else:
        plt.xlabel(r'$\mathcal{N}$')
        plt.ylabel(r'${\rm log}_{10}(w)$')
        plt.title(scatter_title)
        
    if count_refs == True:
        scatter_name += '_phi_UV_'+('%s' % float('%.3g' % phi_r))
    if save_other_plots == True:
        plt.savefig(comp_language+'_results/'+scatter_name,transparent=True, dpi=400)
        

    plt.show()
    plt.clf()
    
          
    
    
    #Plotting the log of the distribution
    histogram_name= 'IS_near_'\
        +str(N_starting)+'_dN_' + ('%s' % float('%.2g' % dN)) + '_m_'+\
        ('%s' % float('%.3g' % m)) +'_phi_UV_'+str(phi_r/phi_i)+'phi_i'+\
        '_bias_'+str(bias)+'_iters_' + str(num_sims_used) +'_bin_size_' +\
        str(min_bin_size)
    if log_normal == True:
        plt.errorbar(bin_centres, heights,\
                     yerr = errors, fmt = '.', capsize=3,\
                    label='{0}'.format('Log-normal'))
        histogram_name='error_bar_distribution_for_'+histogram_name
    elif include_errors == 'yes' and log_normal == False:
        plt.errorbar(bin_centres, heights, yerr = errors, fmt =".k",\
                     capsize=3, label='{0}'.format('Importance Sample'))
        histogram_name='error_bar_distribution_for_'+histogram_name
    else:
        #Plotting the log of the distribution
        heights,bins,_ =\
            plt.hist(sim_N_dist, num_bins,\
                     weights = w_values,\
                         density=True, label='{0}'.format('Weighted'),\
                             histtype="step" )
        _,_,_ =\
            plt.hist(sim_N_dist, num_bins,\
                     density=True, label='{0}'.format('NOT weighted'),\
                         histtype="step" )
        histogram_name='stepped_N_dist_for_'+histogram_name
        
    if vincent == True and bin_centres[-1]>15:
            bins_in_tail = bin_centres[bin_centres>15]
            vincent_near_tail =\
                np.array([cosfuncs.vincent_near_tail_fit(bin_tail, m, phi_i,\
                numerical_integration = False) for bin_tail in bins_in_tail])
                
            plt.plot(bins_in_tail,vincent_near_tail,\
                     label='{0}'.format('Vincent near tail'), linewidth = 2)
                
            
    if m>=1:
        plt.plot(bin_centres_analytical, best_fit_line2,\
                 label='{0}'.format('Pattison 2017'))
        #plt.xlim(right = N_cut_off)
    elif m<1 and m>0.6:
        plt.plot(bin_centres, best_fit_line2, label='{0}'.format('Pattison 2017'))
        if edgeworth_series == True:
            plt.plot(bin_centres, cosfuncs.pdf_gaussian_skew_kurtosis(bin_centres, analytic_N_mean,\
                    analytic_N_var**0.5, analytic_N_skew*analytic_N_var**1.5,\
                        analytic_N_4th_cmoment-3*analytic_N_var**2),\
                     label='{0}'.format('Edgeworth expansion'))
    else:
        '''
        plt.plot(bins, sci_stat.skewnorm.pdf(bins, 0,\
                loc = analytic_N_mean,scale = analytic_N_st),\
                 label='{0}'.format('Gaussian'))
        '''
        '''
        plt.plot(bins, sci_stat.skewnorm.pdf(bins, analytic_N_skew,\
                loc = analytic_N_mean,scale = analytic_N_st),\
                 label='{0}'.format('Gaussian+Skew'))
        '''
        if edgeworth_series == True:
            plt.plot(bin_centres, cosfuncs.pdf_gaussian_skew_kurtosis(bin_centres, analytic_N_mean,\
                    analytic_N_var**0.5, analytic_N_skew*analytic_N_var**1.5,\
                        analytic_N_4th_cmoment-3*analytic_N_var**2),\
                     label='{0}'.format('Edgeworth expansion'))
            '''
            plt.axvline(analytic_gauss_deviation_pos, color='k',\
                        linestyle='dashed', linewidth=2,\
                        label='{0}'.format(r'Expected Gaussian diff'))
            '''

        plt.plot(bin_centres, sci_stat.norm.pdf(bin_centres,\
                analytic_N_mean, analytic_N_st),\
                 label='{0}'.format('Gaussian'))
        '''
        plt.axvline(N_star, color='k', linestyle='dashed', linewidth=2,\
                label='{0}'.format(r'$<\mathcal{N}>+4\sqrt{\delta \mathcal{N}^2}$'))
        '''
    plt.title( str(num_sims_used) + r', bias=' +str(bias)+r', $dN$=' +\
          ('%s' % float('%.2g' % dN))+', m='+('%s' % float('%.3g' % m)))
    plt.xlabel(r'$\mathcal{N}$')
    plt.ylabel(r'$P(\mathcal{N})$')
    plt.ylim(bottom = np.min(heights[heights>0]))
    plt.xlim(right = np.max(bin_centres[heights>0]))
    if emg_fitting == 'chi_squared':
        plt.plot(bin_centres, sci_stat.exponnorm.pdf(bin_centres,\
                EMG_params[0], EMG_params[1], EMG_params[2]),\
                 label='{0}'.format(r'EMG - $\chi^2$'))
        histogram_name  += '_EMG_chi'
    elif emg_fitting == 'stats':
        plt.plot(bin_centres, sci_stat.exponnorm.pdf(bin_centres,\
                emg_K, emg_mu, emg_sigma),\
                 label='{0}'.format('EMG - Stats'))
        histogram_name  += '_EMG_stats'
    if kazuya_pdf == True:
        if phi_r==phi_i and m==2:
            def kazuya_pdf_new(N):
                return 4.39566*np.exp(- 0.391993*N)
            plt.plot(bin_centres[bin_centres>15], kazuya_pdf_new(bin_centres[bin_centres>15]),\
                 label='{0}'.format(r'Kazuya residual'))
        elif m == 0.3:
            kazuya_data = pd.read_csv(comp_language+'_results/'+'kazuya_results_m_0.3.csv', index_col=0)
            plt.plot(kazuya_data['N'], kazuya_data['pdf'], label='{0}'.format('Kazuya'))
        histogram_name  += '_kazuya_pdf'
    if tail_analysis == True:
        plt.plot(bin_centres_tail, expo_model(bin_centres_tail,a_expo,b_expo),\
                 label='{0}'.format('Expo fit: '+('%s' % float('%.3g' % -b_expo))))
        histogram_name += '_expo_fit_to_tail'
    plt.legend(loc='lower left')
    plt.yscale('log')
    plt.savefig(comp_language+'_results/'+histogram_name+'.pdf',transparent=True)
    plt.show()
    plt.close()
    
    if publication_plots == True:
        #Plotting the log of the distribution
        histogram_name= 'publishable_error_bar_IS_near_'\
            +str(N_starting)+'_dN_' + ('%s' % float('%.2g' % dN)) + '_m_'+\
            ('%s' % float('%.3g' % m)) +'_phi_UV_'+str(phi_r/phi_i)+'phi_i'+\
            '_bias_'+str(bias)+'_iters_' + str(num_sims_used) +'_bin_size_' +\
            str(min_bin_size)
        if bias==0:
            plt.errorbar(bin_centres, heights, yerr = errors, fmt =".", ms=7,\
                         capsize=3, color = CB_color_cycle[7],\
                             label='{0}'.format(r'Direct ($\mathcal{A}=0$)'))
        else:
            plt.errorbar(bin_centres, heights, yerr = errors, fmt =".", ms=7,\
                         capsize=3, color = CB_color_cycle[0],\
                             label='{0}'.format(r'$\mathcal{A}=$'+str(bias)))

            #r'Na'+u'\u00EF'+'ve method'
        if vincent == True and bin_centres[-1]>15 and phi_r>phi_i:
            bins_in_tail = bin_centres[bin_centres>15]
            vincent_near_tail =\
                np.array([cosfuncs.vincent_near_tail_fit(bin_tail, m, phi_i,\
                numerical_integration = False) for bin_tail in bins_in_tail])
                
            plt.plot(bins_in_tail,vincent_near_tail, color = CB_color_cycle[3],\
                     label='{0}'.format('Near tail approx.'), linewidth = 2.5)
                    
                
        if m>=1:
            plt.plot(bin_centres_analytical, best_fit_line2,\
                     label='{0}'.format(r'Exact $\phi_{\rm UV} \rightarrow \infty$'),\
                     linewidth = 2, color = CB_color_cycle[1], linestyle='dashed')
            #plt.xlim(right = N_cut_off)
        elif m<1 and m>0.6:
            plt.plot(bin_centres, best_fit_line2,\
                     label='{0}'.format(r'Exact $\phi_{\rm UV} \rightarrow \infty$'), linewidth = 2)
            if edgeworth_series == True:
                plt.plot(bin_centres, cosfuncs.pdf_gaussian_skew_kurtosis(bin_centres, analytic_N_mean,\
                        analytic_N_var**0.5, analytic_N_skew*analytic_N_var**1.5,\
                            analytic_N_4th_cmoment-3*analytic_N_var**2),\
                         label='{0}'.format('Edgeworth expansion'), linewidth = 2)
        else:
            if edgeworth_series == True:
                plt.plot(bin_centres, cosfuncs.pdf_gaussian_skew_kurtosis(bin_centres, analytic_N_mean,\
                        analytic_N_var**0.5, analytic_N_skew*analytic_N_var**1.5,\
                            analytic_N_4th_cmoment-3*analytic_N_var**2),\
                         label='{0}'.format('Edgeworth'), linewidth = 2,\
                            color = CB_color_cycle[2])
                    
                plt.axvline(analytic_gauss_deviation_pos, color='dimgrey',\
                            linestyle='dashed', linewidth=2)
    
            plt.plot(bin_centres, sci_stat.norm.pdf(bin_centres,\
                    analytic_N_mean, analytic_N_st),\
                     label='{0}'.format('Gaussian'), linewidth = 2,\
                    color = CB_color_cycle[1])
            '''
            plt.axvline(N_star, color='k', linestyle='dashed', linewidth=2,\
                    label='{0}'.format(r'$<\mathcal{N}>+4\sqrt{\delta \mathcal{N}^2}$'))
            '''
        plt.xlabel(r'$\mathcal{N}$', fontsize = fontsize)
        plt.ylabel(r'$P(\mathcal{N})$', fontsize = fontsize)
        plt.ylim(bottom = np.min(heights[heights>0]))
        plt.xlim(right = np.max(bin_centres[heights>0]))
        if kazuya_pdf == True:
            if phi_r==phi_i:
                if m==2:#From applying residual theorem to leading pole
                    def kazuya_pdf_new(N):
                        return 4.39565*np.exp(-0.391993*N)
                elif m==1:#From applying residual theorem to leading pole
                    def kazuya_pdf_new(N):
                        return 213.842*np.exp(-0.652823*N)
                plt.plot(bin_centres[bin_centres>15],\
                         kazuya_pdf_new(bin_centres[bin_centres>15]),\
                     label='{0}'.format(r'Leading pole'),linewidth = 2, color = CB_color_cycle[2])
                histogram_name  += '_kazuya_pdf'
        #In case you want to change label order
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [i for i in range(len(handles))]
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]
        plt.legend(fontsize = fontsize, handles = handles, labels=labels)
        plt.yscale('log')
        plt.margins(tight=True)
        plt.savefig('for_paper/'+histogram_name+'new_colours.pdf', transparent=True)
        plt.show()
        plt.close()
    
if bias == 0:
    if include_errors == 'yes':
        plt.errorbar(bin_centres, heights, yerr = errors, fmt =".k",\
                     capsize=3, label='{0}'.format('Sim'))
        plt.ylim(bottom = np.min(heights[heights>0])) 
    else:
    #Plotting the log of the distribution
        bin_height,bins,_ =\
            plt.hist(sim_N_dist, num_bins,\
                         density=True, label='{0}'.format('Data'),\
                             histtype="step" )
        plt.ylim(bottom = np.min(bin_height[bin_height>0])) 
        
    if m>=1:
        plt.plot(bin_centres_analytical, best_fit_line2, label='{0}'.format('Pattison 2017'))
        plt.plot()
        plt.xlim(right = N_cut_off)
    elif m<1 and m>0.6:
        plt.plot(bin_centres, best_fit_line2, label='{0}'.format('Pattison 2017'))
        plt.plot(bin_centres, sci_stat.skewnorm.pdf(bin_centres, analytic_N_skew,\
                loc = analytic_N_mean,scale = analytic_N_st),\
                 label='{0}'.format('Gaussian+Skew'))
        plt.ylim(bottom = np.min(best_fit_line2)) 
        plt.xlim(right = N_cut_off)
    else:
        plt.plot(bins, sci_stat.skewnorm.pdf(bins, analytic_N_skew,\
                loc = analytic_N_mean,scale = analytic_N_st),\
                 label='{0}'.format('Gaussian+Skew'))
        plt.axvline(N_star, color='k', linestyle='dashed', linewidth=2,\
                label='{0}'.format(r'$<\mathcal{N}>+4\sqrt{\delta \mathcal{N}^2}$'))
    #If including error bars on the histogram
    #plt.xlim(0, 0.15) 
    plt.xlabel(r'$\mathcal{N}$')
    plt.ylabel(r'$\mathcal{N}$')
    plt.yscale('log')
    plt.title( str(num_sims) + r', bias=' +str(bias)+r', $dN$=' +\
          ('%s' % float('%.2g' % dN))+', m='+('%s' % float('%.3g' % m)))
    #plt.xlim((4, 12))
    #plt.ylim((0, 0.05)) 
    plt.legend()
    histogram_name = 'stepped_N_distribution_for_' + '_near_' +\
    str(N_starting)+'_dN_' + str(dN) + '_m_'+('%s' % float('%.3g' % m)) +\
        '_Is_shift_'+str(bias)+ '_iterations_' + str(num_sims)+'.pdf'
    plt.savefig(comp_language+'_results/'+histogram_name,transparent=True)
    plt.show()
    plt.close()
    



