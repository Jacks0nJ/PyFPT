    # -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:50:47 2021

Varying Delta v to see if the coefficent of the exponential fit stays the
 same and is therefore robust

@author: jjackson
"""

import numpy as np
import pandas as pd

import scipy.stats as sci_stat
from timeit import default_timer as timer
import multiprocessing as mp
from multiprocessing import Process, Queue
import scipy.optimize

import importance_sampling_sr_cython11 as is_code
import inflation_functions_e_foldings as cosfuncs
import is_data_analysis as isfuncs
import stochastic_inflation_cosmology as sm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.lines as mlines
import mpl_style
plt.style.use(mpl_style.style1)


#M_PL = 2.435363*10**18 old value
M_PL = 1.0# Using units of M_PL
PI = np.pi
m = 1

###Intial conditions and tolerances
N_starting = 10#In some sense, this should techically be negative
a_i = 1
phi_end = M_PL*2**0.5
phi_i = M_PL*(4*N_starting+2)**0.5#M_PL*(4*N_starting+2)**0.5
N_cut_off = 300
N_f = 60.0
tol = 2*10**(-9)
dN = 0.02*m
num_bins = 50
num_sub_samples = 20
min_bin_size = 400
fit_threshold = 200
N_threshold = 55
min_tail_start = 4


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

step_type = 'Euler_Maruyama'
comp_language = 'Cython'
ps_calculation = 'no'
using_IS = 'yes'
include_errors = 'yes'
tail_analysis = False
include_std_w_plot = False
log_normal = False
publication_plots = True
contour = True
scater_density_plot = False
fontsize = 20
save_data = False
emg_fitting = False#'chi_squared'
edgeworth_series = False
manual_norm = True
storage = True
wind_type = 'diffusion'

if m == 0.3:
    kazuya_pdf = True
    vincent = False
elif m>0.6:
    vincent = True
    kazuya_pdf = True
    analytical_pdf = True
else:
    kazuya_pdf = False
    vincent = False

if emg_fitting == 'chi_squared':
    guess = None
        
'''
Need to run the main multiple times, for each bias value
'''

#Using different Delta v for the same m
bias = 0.4
phi_r_range = np.linspace(phi_i, 1.4*phi_i, 3)

num_full_sims = len(phi_r_range)

b_params = np.zeros(num_full_sims)
b_param_errs = np.zeros(num_full_sims)
tail_start = np.zeros(num_full_sims)
tail_start_err = np.zeros(num_full_sims)#Based on bin width
std_range = np.zeros(num_full_sims)
mean_range = np.zeros(num_full_sims)
if include_std_w_plot == True:
    log_w_std_average = np.zeros(num_full_sims)

#If storing the different histograms
if storage == True:
    height_storage = np.zeros((num_bins,len(phi_r_range)))
    bin_centres_storage = np.zeros((num_bins,len(phi_r_range)))
    if log_normal == True:
        errors_storage = np.zeros((len(phi_r_range), 2, num_bins))
    elif include_errors == 'yes':
        errors_storage = np.zeros((num_bins,len(phi_r_range)))
    if contour == True:
        xedges_centre_storage = np.zeros((num_bins,len(phi_r_range)))
        yedges_centre_storage = np.zeros((num_bins,len(phi_r_range)))
        h_storage = np.zeros((len(phi_r_range), num_bins, num_bins))
        
        
for j, phi_r in enumerate(phi_r_range):
    if phi_r==phi_i:
        print('Now simulating for phi_r = ' + 'phi_i')
    else:
        print('Now simulating for phi_r = ' +\
              ('%s' % float('%.2g' % phi_r)))
    
    
    num_sims = int(200000/mp.cpu_count())
    def V(phi):
        V = 0.5*(m*phi)**2
        return V
    
    def V_dif(phi):
        V_dif = (m**2)*phi
        return V_dif
    
    def V_ddif(phi):
        V_ddif = (m**2)
        return V_ddif
    
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
    
    if using_IS == 'yes':
        def multi_processing_func(phi_i, phi_r, phi_end, N_i, N_f, dN, bias,\
                                  num_sims, queue_Ns, queue_ws):
            #Doing the double absorbing surface code
    
            Ns, ws =\
                is_code.many_simulations_importance_sampling(phi_i, phi_r, phi_end,\
                         N_i, N_f, dN, bias, num_sims, V, V_dif, V_ddif,\
                             bias_type = 'diffusion')
            '''
            Ns, ws =\
                cython_code.many_simulations_importance_sampling(phi_i, phi_end,\
                         0, 200, dN, bias, num_sims)
            '''
            Ns = np.array(Ns)
            ws = np.array(ws)
            queue_Ns.put(Ns)
            queue_ws.put(ws)
    
        if __name__ == "__main__":
            queue_Ns = Queue()
            queue_ws = Queue()
            cores = int(mp.cpu_count()/1)
    
            print('Number of cores used: '+str(cores))
            processes = [Process(target=multi_processing_func,\
                                 args=(phi_i, phi_r,  phi_end, 0.0, N_f, dN, bias,\
                                  num_sims, queue_Ns, queue_ws)) for i in range(cores)]
            for p in processes:
                p.start()
            
            #for p in processes:
             #   p.join()
            
            Ns_array = np.array([queue_Ns.get() for p in processes])
            ws_array = np.array([queue_ws.get() for p in processes])
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
        
        estimated_mean = cosfuncs.importance_sampling_mean(sim_N_dist, w_values)
        estimated_st = cosfuncs.importance_sampling_st(sim_N_dist, w_values)
        
        
    else:
        N_values =\
        is_code.many_simulations(phi_i, phi_end, 0.0, 150, dN, num_sims)
        sim_N_dist = np.array(N_values)
        
    
    
    
    '''
    Saving data
    '''
    if save_data == True:
        data_dict = {}
        data_dict['Ns'] = sim_N_dist
        if using_IS == 'yes':
            data_dict['ws'] = w_values
        
        data_pandas = pd.DataFrame(data_dict)
        
        my_file_name = 'N_'+str(N_starting)+'_dN_'+str(dN)+'_m_'+('%s' % float('%.3g' % m))+\
        '_iterations_'+str(num_sims)+'_bias_'+str(bias)+'_'+step_type+'_'+\
            comp_language+'.csv'
        #Saving to a directory for the language used
        data_pandas.to_csv(comp_language+'_results/'+my_file_name)
        #Remembering to remove column numbering
        sim_data = pd.read_csv(comp_language+'_results/'+my_file_name, index_col=0)
        
        sim_N_dist = np.array(sim_data['Ns'])
        w_values = np.array(sim_data['ws'])
        
    sim_N_dist, w_values = cosfuncs.histogram_data_truncation(sim_N_dist,\
                              N_threshold, weights=w_values,\
                              num_sub_samples = num_sub_samples)
    num_sims = len(sim_N_dist)
        
    
    '''
    #Post processesing
    '''
    
    if bias != 0:
        sim_N_mean = cosfuncs.importance_sampling_mean(sim_N_dist, w_values)
        sim_N_var = cosfuncs.importance_sampling_var(sim_N_dist, w_values)
        sim_N_skew = cosfuncs.importance_sampling_skew(sim_N_dist, w_values)
        sim_N_kurtosis = cosfuncs.importance_sampling_kurtosis(sim_N_dist, w_values)
        
        sim_3rd_cumulant =\
            cosfuncs.importance_sampling_3rd_cumulant(sim_N_dist, w_values)
        sim_4th_cumulant =\
            cosfuncs.importance_sampling_4th_cumulant(sim_N_dist, w_values)
        
        sim_mean_error = cosfuncs.jackknife(sim_N_dist, int(num_sims/num_sub_samples),\
                        cosfuncs.importance_sampling_mean, weights = w_values)
        sim_var_error = cosfuncs.jackknife(sim_N_dist, int(num_sims/num_sub_samples),\
                        cosfuncs.importance_sampling_var, weights = w_values)
        sim_skew_error = cosfuncs.jackknife(sim_N_dist, int(num_sims/num_sub_samples),\
                        cosfuncs.importance_sampling_skew, weights = w_values)
        sim_kurtosis_error = cosfuncs.jackknife(sim_N_dist, int(num_sims/num_sub_samples),\
                        cosfuncs.importance_sampling_kurtosis, weights = w_values)
    else:#When not using importance sampling
        #The statistics of the data
        sim_N_mean, sim_N_st = sci_stat.norm.fit(sim_N_dist)
        sim_N_var = sim_N_st**2
        sim_N_skew = sci_stat.skew(sim_N_dist)
        sim_N_kurtosis = sci_stat.kurtosis(sim_N_dist)
        
        sim_mean_error = cosfuncs.jackknife(sim_N_dist, int(num_sims/num_sub_samples), np.mean)
        sim_st_error = cosfuncs.jackknife(sim_N_dist, int(num_sims/num_sub_samples), np.std)
        sim_skew_error = cosfuncs.jackknife(sim_N_dist, int(num_sims/num_sub_samples),\
                                            sci_stat.skew)
        sim_kurtosis_error = cosfuncs.jackknife(sim_N_dist, int(num_sims/num_sub_samples),\
                                            sci_stat.kurtosis)
    
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
    

    '''
    PDF analysis
    '''

    if np.abs(bias)>0.2:
        bin_centres,heights,errors,num_sims_used,_ =\
            isfuncs.data_points_pdf(sim_N_dist, w_values, num_sub_samples,\
            bins=num_bins, include_std_w_plot = include_std_w_plot,\
            min_bin_size = min_bin_size, log_normal = log_normal,\
            num_sims = num_sims, log_normal_method='ML', w_hist_num = 10,\
            p_value_plot = True)
    #I'm not convinced 
    elif np.abs(bias)<=0.2:
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
        
    '''
    Fitting models to the tail
    '''
        
        
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
            
            #Storing the best fit values
            b_params[j] = b_expo
            b_param_errs[j] = b_expo_err
            
            
            
            
            #Storing the start of the tail and the standard deviation
            tail_start[j] = bin_centres[tail_start_idx]
            #Using bin width as rough estimate of error
            tail_start_err[j] = np.diff(bin_centres)[0]
            
            tail_start_classical =\
                cosfuncs.gaussian_deviation(analytic_N_mean,\
                analytic_N_var**0.5, analytic_N_skew*analytic_N_var**1.5,\
                analytic_N_4th_cmoment-3*analytic_N_var**2,\
                nu=fit_threshold/100)
            std_range[j] = np.std(sim_N_dist)
            mean_range[j] = np.mean(sim_N_dist)
            
        
    elif len(heights)-tail_start_idx>1:
        print('Only '+str(len(heights)-tail_start_idx)+' data points deviated by more'+
              ' than '+str(fit_threshold)+'% from classical prediction: no enough to fit data')
        tail_analysis = False
        b_params[j] = 0
        b_param_errs[j] = 0
        tail_start[j] = 0
        tail_start_err[j] = 0
        std_range[j] = np.std(sim_N_dist)
        mean_range[j] = np.mean(sim_N_dist)
    else:
        print('No deviated from classical prediction greater than '+str(fit_threshold)+'%')
        tail_analysis = False
        b_params[j] = 0
        b_param_errs[j] = 0
        tail_start[j] = 0
        tail_start_err[j] = 0
        std_range[j] = np.std(sim_N_dist)
        mean_range[j] = np.mean(sim_N_dist)
        
        
    #Checking if multipprocessing error occured, by looking at correlation
    pearson_corr = np.corrcoef(sim_N_dist, np.log10(w_values))
    pearson_corr = pearson_corr[0,1]
    
    #Checking if multiprocessing error occured by looking at weights
    #Plotting the weights
    plt.scatter(sim_N_dist,np.log10(w_values))
    plt.xlabel(r'$\mathcal{N}$')
    plt.ylabel(r'$log_{10}(w)$')
    plt.title( r'#'+str(num_sims)+r', $dN=$' + str(dN)+ r', $m$=' + ('%s' % float('%.3g' % m)) )
    plt.show()
    plt.clf()

    if contour == True:
        h, xedges, yedges, _ =\
            plt.hist2d(sim_N_dist, np.log10(w_values), (50, 50))
        plt.clf()
        xedges_centre =\
            np.array([(xedges[k]+xedges[k+1])/2 for k in range(len(xedges)-1)])
        yedges_centre =\
            np.array([(yedges[k]+yedges[k+1])/2 for k in range(len(yedges)-1)])
        if storage == True:
            xedges_centre_storage[:,j] = xedges_centre
            yedges_centre_storage[:,j] = yedges_centre
            h_storage[j,:,:] = h
            
    elif scater_density_plot == True:
        plt.hist2d(sim_N_dist, np.log10(w_values), (50, 50), norm=LogNorm())
        cbar = plt.colorbar()
        cbar.set_label(r'# Data Points')
        scatter_title = r'bias = '+('%s' % float('%.3g' % bias))+r', $dN=$' +\
          ('%s' % float('%.3g' % dN))+ r', $m$=' + ('%s' % float('%.3g' % m)) 
    else:
        plt.scatter(sim_N_dist,np.log10(w_values))
        scatter_title = r'bias = '+('%s' % float('%.3g' % bias))+r', $dN=$' +\
          ('%s' % float('%.3g' % dN))+ r', $m$=' + ('%s' % float('%.3g' % m))              
    plt.show()
    plt.clf()
    
    #If storing the different histograms
    if storage == True:
        height_storage[:len(heights),j] = heights
        bin_centres_storage[:len(heights),j] = bin_centres
        if log_normal == True:
            errors_storage[j,:,:len(heights)] = errors
        elif include_errors == 'yes':
            errors_storage[:len(heights),j] = errors
            
        if j==0:
            all_Ns = sim_N_dist
            all_ws = w_values
        else:
            all_Ns = np.hstack((all_Ns,sim_N_dist))
            all_ws = np.hstack((all_ws,w_values))
    
    
    plt.errorbar(bin_centres, heights, yerr = errors, fmt =".k",\
             capsize=3, label='{0}'.format('Importance Sample'))
    if edgeworth_series == True:
        plt.plot(bin_centres, sci_stat.norm.pdf(bin_centres,\
            analytic_N_mean, analytic_N_st),\
             label='{0}'.format('Gaussian'))
        plt.plot(bin_centres, cosfuncs.pdf_gaussian_skew_kurtosis(bin_centres, analytic_N_mean,\
                analytic_N_var**0.5, analytic_N_skew*analytic_N_var**1.5,\
                    analytic_N_4th_cmoment-3*analytic_N_var**2),\
                 label='{0}'.format('Edgeworth'))
    else:
        plt.plot(bin_centres, sci_stat.norm.pdf(bin_centres,\
            analytic_N_mean, analytic_N_st),\
             label='{0}'.format('Gaussian'))
            
    if emg_fitting == 'chi_squared' and bias>0:
        def log_of_exponnorm_pdf(x, K, mean, sigma):
            return np.log(sci_stat.exponnorm.pdf(x, K, mean, sigma))
        if guess == None:
            guess = (sim_N_skew, sim_N_mean, sim_N_var**0.5)
        EMG_params, cv =\
        scipy.optimize.curve_fit(log_of_exponnorm_pdf, bin_centres,\
                                 np.log(heights),\
                                p0 = guess)
            
        plt.plot(bin_centres, sci_stat.exponnorm.pdf(bin_centres,\
                EMG_params[0], EMG_params[1], EMG_params[2]),\
                 label='{0}'.format('EMG'))
        guess = (EMG_params[0], EMG_params[1], EMG_params[2])
        
    elif emg_fitting == 'stats' and bias>0:
         emg_mu, emg_sigma, emg_K =\
             cosfuncs.expo_mod_gauss_params_guess(sim_N_mean, sim_N_var**0.5,\
                                                  sim_N_skew)

    if len(heights)-tail_start_idx>5:
        plt.plot(bin_centres_tail, expo_model(bin_centres_tail,a_expo,b_expo),\
                 label='{0}'.format('Expo fit to tail'))
    plt.ylim(bottom = np.min(heights[heights>0]))
    #plt.axvline(analytic_gauss_deviation_pos, color='k', linestyle='dashed', linewidth=2,\
    #        label='{0}'.format(r'Expected Gaussian diff'))
    plt.title( str(num_sims) + r', bias=' +('%s' % float('%.3g' % bias))+r', $dN$=' +\
          ('%s' % float('%.2g' % dN))+', m='+('%s' % float('%.3g' % m)))
    plt.yscale('log')
    plt.legend()
    plt.show()
    plt.clf()
    if pearson_corr > -0.55 and pearson_corr < 0.55:#Data is uncorrelated
        print('Multiprocessing error occured, ignoring. m = '+str(m))
        print('Peasron correlation coefficent is ' + str(pearson_corr))
        b_params[j] = 0
        b_param_errs[j] = 0
        tail_start[j] = 0
        tail_start_err[j] = 0
        std_range[j] = 0
        mean_range[j] = 0
    
    
'''
 Plotting and post processing
'''
#Analytical values
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

#Remove cases where no exponential fit could be found
has_data_points_logic = np.abs(b_params)>0
b_params = b_params[has_data_points_logic]
tail_start = tail_start[has_data_points_logic]
tail_start_err = tail_start_err[has_data_points_logic]
std_range_og = std_range 
std_range = std_range[has_data_points_logic]
mean_range_og = mean_range
mean_range = mean_range[has_data_points_logic]

tail_start_from_mean = tail_start - analytic_N_mean



num_full_sims = len(b_params)




#plotting the different distributions on top of each other
if storage == True:
    colour_cycle_og = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',\
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    #Custom colour order, to be consistant with the other graphs.
    colour_cycle = ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd',\
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    data_dict = {}
    #Looping over all of the different plots
    for j in range(len(phi_r_range)):
        heights_temp = height_storage[:,j]
        #only used filled heights
        heights_temp = heights_temp[heights_temp>0]
        
        #Same for the bin centres
        bin_centres_temp = bin_centres_storage[:,j]
        #only used filled heights
        bin_centres_temp = bin_centres_temp[bin_centres_temp>0]
        
        if log_normal == True:
            errors_temp = errors_storage[j,:,:]
            errors_temp = errors_temp[:,errors_temp[0,:]>0]
            #This is so you can artifically increase the errors on the log plot
            alpha = 1
            errors_temp[0,:] =\
                heights_temp*(1-(1-errors_temp[0,:]/heights_temp)**alpha)
            errors_temp[1,:] =\
                heights_temp*((errors_temp[1,:]/heights_temp+1)**alpha-1)
                
        elif include_errors == 'yes':
            errors_temp = errors_storage[:,j]
            errors_temp = errors_temp[errors_temp>0]
            
            

        #stack the heights bin centres
        if j==0:
            bin_centres_combined=bin_centres_temp
            heights_combined = heights_temp
            if include_errors == 'yes' or log_normal == True:
                errors_combined = errors_temp
        else:
            bin_centres_combined=np.hstack((bin_centres_combined,bin_centres_temp))
            heights_combined=np.hstack((heights_combined,heights_temp))
            if include_errors == 'yes' or log_normal == True:
                errors_combined = np.hstack((errors_combined,errors_temp))
            
        
        if include_errors == 'yes':#Plot with errors
            if j==0:
                colour_index = 0
            else:
                colour_index = j+3
            plt.errorbar(bin_centres_temp, heights_temp, yerr=errors_temp,\
                         capsize=3, fmt =".", ms=7,\
                         label='{0}'.format( r'$\phi_{\rm UV}$ = ' +\
                         ('%s' % float('%.3g' % (phi_r_range[j]/phi_i)))+\
                         r'$\phi_i$' ), color = CB_color_cycle[colour_index])
        else:#Plot just data points
            plt.errorbar(bin_centres_temp, heights_temp, fmt =".")
            
    ###Loop finished
         
    sort_idx = np.argsort(bin_centres_combined)
    heights_combined = heights_combined[sort_idx]
    bin_centres_combined = bin_centres_combined[sort_idx]
    if log_normal == True:
        errors_combined = errors_combined[:,sort_idx]
    else:
        errors_combined = errors_combined[sort_idx]
    
    
    data_dict = {}
    data_dict['heights'] = heights_combined
    data_dict['bins'] = bin_centres_combined
    if include_errors == 'yes':
        if log_normal == True:
            data_dict['errors_lower'] = errors_combined[0,:]
            data_dict['errors_upper'] = errors_combined[1,:]
        else:
            data_dict['errors_symmetric'] = errors_combined
    
    data_pandas = pd.DataFrame(data_dict)
    
    my_file_name = 'combined_data_phi_r_range_'\
                +('%s' % float('%.3g' % np.min(phi_r_range)))+'_to_'\
                    +('%s' % float('%.3g' % np.max(phi_r_range)))+'_with_m_'\
                +str(m)+'_phi_UV_'+str(phi_r/phi_i)+'phi_i'+'min_bin_size_'\
                    +str(min_bin_size)+'.csv'
    #Saving to a directory for the language used
    data_pandas.to_csv(comp_language+'_results/'+my_file_name)
    #Remembering to remove column numbering
    combined_data = pd.read_csv(comp_language+'_results/'+my_file_name, index_col=0)
    
    
    heights_combined = np.array(combined_data['heights'])
    bin_centres_combined = np.array(combined_data['bins'])
    if include_errors == 'yes':
        if log_normal == True:
            errors_combined = np.zeros((2, len(heights_combined)))
            errors_combined[0,:] = np.array(combined_data['errors_lower'])
            errors_combined[1,:] = np.array(combined_data['errors_upper'])
        else:
            errors_combined = data_dict['errors_symmetric']
    
    
    #Add finish plot
    if edgeworth_series == True:
        plt.plot(bin_centres_combined, cosfuncs.pdf_gaussian_skew_kurtosis(bin_centres_combined, analytic_N_mean,\
                        analytic_N_var**0.5, analytic_N_skew*analytic_N_var**1.5,\
                            analytic_N_4th_cmoment-3*analytic_N_var**2),\
                         label='{0}'.format('Edgeworth'), linewidth = 2, color = CB_color_cycle[2])
        plt.plot(bin_centres_combined, sci_stat.norm.pdf(bin_centres_combined,\
                analytic_N_mean, analytic_N_var**0.5),\
                label='{0}'.format('Gaussian'), color = CB_color_cycle[1])
    elif m<0.6:
        plt.plot(bin_centres_combined, sci_stat.norm.pdf(bin_centres_combined,\
                analytic_N_mean, analytic_N_var**0.5),\
                label='{0}'.format('Gaussian'))
            
    overlap_name = 'overlap_plot_m_'+str(m) + '_phi_r_range_log_' +\
        ('%s' % float('%.3g' % (np.min(phi_r_range)/phi_i)))+ '_to_' +\
        ('%s' % float('%.3g' % (np.max(phi_r_range)/phi_i))) +'phi_i_bias_'+\
        str(bias)
    if kazuya_pdf == True:
        if m==2:#From applying residual theorem to leading pole
            if np.any([phi_r_range==phi_i]):
                def kazuya_pdf_new(N):
                    return 4.39565*np.exp(-0.391993*N)
                plt.plot(bin_centres[bin_centres>15],\
                     kazuya_pdf_new(bin_centres[bin_centres>15]),\
                 label='{0}'.format(r'NLO pole'),linewidth = 2, color = CB_color_cycle[2])
        elif m==1:#From applying residual theorem to leading pole
            if np.any([phi_r_range==phi_i]):
                def kazuya_pdf_new(N):
                    return 213.842*np.exp(-0.652823*N)
                plt.plot(bin_centres_combined[bin_centres_combined>15],\
                     kazuya_pdf_new(bin_centres_combined[bin_centres_combined>15]),\
                 label='{0}'.format(r'NLO pole'),linewidth = 2, color = CB_color_cycle[2])
                    
            if np.any([phi_r_range==1.2*phi_i]):
                def kazuya_pdf_new(N):
                    return 6.20019*np.exp(-0.379787*N)
                plt.plot(bin_centres_combined[bin_centres_combined>28],\
                     kazuya_pdf_new(bin_centres_combined[bin_centres_combined>28]),\
                 label='{0}'.format(r'NLO pole'),linewidth = 2, color = CB_color_cycle[2])
                    
            if np.any([phi_r_range==1.4*phi_i]):
                def kazuya_pdf_new(N):
                    return 0.241477*np.exp(-0.245846*N)
                plt.plot(bin_centres_combined[bin_centres_combined>36],\
                     kazuya_pdf_new(bin_centres_combined[bin_centres_combined>36]),\
                 label='{0}'.format(r'NLO pole'),linewidth = 2, color = CB_color_cycle[2])
        elif m==0.3:
            kazuya_data = pd.read_csv(comp_language+'_results/'+'kazuya_results_m_0.3.csv', index_col=0)
            plt.plot(kazuya_data['N'], kazuya_data['pdf'], label='{0}'.format('Kazuya'),
                     linewidth = 2)
        
    if vincent == True and bin_centres_combined[-1]>20:
        if m<1.5:
            plt.plot(bin_centres_combined[bin_centres_combined>20],\
                     cosfuncs.vincent_near_tail_fit(bin_centres_combined[bin_centres_combined>20],\
                    m, phi_i), label='{0}'.format('Vincent near tail'))
        else:
            plt.plot(bin_centres[bin_centres>20],\
                     cosfuncs.vincent_far_tail_fit(bin_centres[bin_centres>20],\
                    m, phi_i), label='{0}'.format('Vincent far tail'))
    if m>=0.6:
        if analytical_pdf == True:
            bin_centres_analytical =\
                np.linspace(bin_centres_combined[0],bin_centres_combined[-1],100)
            PDF_analytical =\
                isfuncs.large_mass_pdf(bin_centres_analytical,phi_i,phi_end,V)
            #Remove any zero values
            bin_centres_analytical = bin_centres_analytical[PDF_analytical>0]
            PDF_analytical= PDF_analytical[PDF_analytical>0]
            analytical_pdf = False
        plt.plot(bin_centres_analytical, PDF_analytical,\
                 label='{0}'.format(r'Exact $\phi_{\rm UV} \rightarrow \infty$'),\
                     linewidth = 2, color = CB_color_cycle[1], linestyle = 'dashed')
            
    if emg_fitting == 'chi_squared':
        # Values from chi - squared fit
        def log_of_exponnorm_pdf(x, K, mean, sigma):
            return np.log(sci_stat.exponnorm.pdf(x, K, mean, sigma))
        EMG_chi_squared, cv =\
        scipy.optimize.curve_fit(log_of_exponnorm_pdf, bin_centres_combined,\
                                 np.log(heights_combined),\
                                p0 = guess)
        EMG_chi_squared_expo = 1/(EMG_chi_squared[0]*EMG_chi_squared[2])
        
        
                
        plt.plot(bin_centres_combined, sci_stat.exponnorm.pdf(bin_centres_combined,\
                EMG_chi_squared[0], EMG_chi_squared[1], EMG_chi_squared[2]),\
                 label='{0}'.format(r'EMG - $\chi^2$'))
        overlap_name += '_EMG_chi_squared'
                
        '''
        print(' Skew error is '+str(100*all_skew_error/all_N_skew)+'%')
        plt.plot(bin_centres_combined, sci_stat.exponnorm.pdf(bin_centres_combined,\
                K, mu, sigma),\
                 label='{0}'.format(r'EMG - stats'))
        overlap_name += '_EMG_stats'
        '''
    elif emg_fitting == 'stats' and bias>0:
        all_N_mean = cosfuncs.importance_sampling_mean(all_Ns, all_ws)
        all_N_var = cosfuncs.importance_sampling_var(all_Ns, all_ws)
        all_N_skew = cosfuncs.importance_sampling_skew(all_Ns, all_ws)
        emg_mu, emg_sigma, emg_K =\
            cosfuncs.expo_mod_gauss_params_guess(all_N_mean, all_N_var**0.5,\
                                                 all_N_skew)
                
        plt.plot(bin_centres_combined, sci_stat.exponnorm.pdf(bin_centres_combined,\
                emg_K, emg_mu, emg_sigma),\
                 label='{0}'.format(r'EMG - $\chi^2$'))
        overlap_name += '_EMG_stats'

 
    plt.ylim(bottom = np.min(heights_combined))
    plt.yscale('log')
    
    if publication_plots == True:
        plt.xlabel(r'$\mathcal{N}$', fontsize = fontsize)
        plt.ylabel(r'$P(\mathcal{N})$', fontsize = fontsize)
        plt.legend(fontsize = fontsize)
        plt.margins(tight=True)
        plt.savefig('for_paper/'+overlap_name+'_new_colours.pdf', transparent=True)
        plt.show()
        plt.close()
    else:
        plt.axvline(N_star, color='k', linestyle='dashed', linewidth=2,\
            label='{0}'.format(r'$<\mathcal{N}>+4\sqrt{\delta \mathcal{N}^2}$'))
        plt.xlabel(r'$\mathcal{N}$')
        plt.ylabel('Probability Density')
        plt.title(str(len(phi_r_range)) +r' $\times$ '+str(num_sims_used) +\
                  r', $dN$=' + ('%s' % float('%.2g' % dN))+', m=' +\
                  ('%s' % float('%.3g' % m)))
        plt.legend(loc='lower left')
        plt.savefig(comp_language+'_results/'+overlap_name+'.pdf',transparent=True)
        plt.show()
        plt.close()
    
            
