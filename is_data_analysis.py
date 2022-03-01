#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 28/02/22

@author: jjackson
"""
from timeit import default_timer as timer
import multiprocessing as mp
from multiprocessing import Process, Queue

import inflation_functions_e_foldings as cosfuncs
import importance_sampling_sr_cython12 as is_code

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sci_stat


#M_PL = 2.435363*10**18 old value
M_PL = 1.0# Using units of M_PL
PI = np.pi


def IS_simulation(phi_i, phi_end, V, V_dif, V_ddif, num_sims, bias, bins=50,\
                  dN = False, min_bin_size = 400, num_sub_samples=20,\
                  reconstruction = 'lognormal', save_data = False, N_f = 100,
                  phi_UV = False):
    #If no argument for dN is given, using the classical std to define it
    if dN == False:
        if isinstance(bins, int) == True:
            std =\
                cosfuncs.delta_N_squared_sto_limit(V, V_dif, V_ddif, phi_i,\
                                                   phi_end)
            dN = std/(3*bins)
        elif isinstance(bins, int) == False:
            std =\
                cosfuncs.delta_N_squared_sto_limit(V, V_dif, V_ddif, phi_i,\
                                                   phi_end)
            num_bins = len(bins)-1
            dN = std/(3*bins)
    elif isinstance(dN, float) != True and isinstance(dN, int) != True:
        raise ValueError('dN is not a number')
        
        
    #Default to the naive method
    if reconstruction == 'lognormal':
        log_normal = True
    elif reconstruction == 'naive':
        log_normal = False
    else:
        print('Invalid reconstruction argument, defaulting to naive method')
        log_normal = False
        
    if phi_UV == False:
        phi_UV = 10000*phi_i
    elif isinstance(dN, float) != True and isinstance(dN, int) != True:
        raise ValueError('phi_UV is not a number')
        
        
    #The number of sims per core, so the total is correct
    num_sims_per_core = int(num_sims/mp.cpu_count())
    
    
    
    start = timer()
    
    
    def multi_processing_func(phi_i, phi_UV, phi_end, N_i, N_f, dN, bias,\
                              num_sims, queue_Ns, queue_ws, queue_refs):
        results =\
                is_code.many_simulations_importance_sampling(phi_i, phi_UV, phi_end,\
                         N_i, N_f, dN, bias, num_sims, V, V_dif, V_ddif,\
                             bias_type = 'diffusion', count_refs = False)
        Ns = np.array(results[0][:])
        ws = np.array(results[1][:])
        queue_Ns.put(Ns)
        queue_ws.put(ws)

    

    queue_Ns = Queue()
    queue_ws = Queue()
    queue_refs = Queue()
    cores = int(mp.cpu_count()/1)

    print('Number of cores used: '+str(cores))
    processes = [Process(target=multi_processing_func,\
                args=(phi_i, phi_UV,  phi_end, 0.0, N_f, dN, bias,\
                num_sims_per_core, queue_Ns, queue_ws, queue_refs)) for i in range(cores)]
        
    for p in processes:
        p.start()
    
    #for p in processes:
     #   p.join()
    
    Ns_array = np.array([queue_Ns.get() for p in processes])
    ws_array = np.array([queue_ws.get() for p in processes])
    end = timer()
    print(f'The simulations took: {end - start}')
        
    #Combine into columns into 1
    sim_N_dist  = Ns_array.flatten()
    w_values = ws_array.flatten()
    
    
    #Sort in order of increasing Ns
    sort_idx = np.argsort(sim_N_dist)
    sim_N_dist = sim_N_dist[sort_idx]
    w_values = w_values[sort_idx]
        
    #Checking if multipprocessing error occured, by looking at correlation
    pearson_corr = np.corrcoef(sim_N_dist, np.log10(w_values))
    pearson_corr = pearson_corr[0,1]
    
    if pearson_corr > -0.55:#Data is uncorrelated
        print(len(sim_N_dist))
        print(sim_N_dist)
        raise ValueError('Possible multiprocessing error occured, terminating')

    
    '''
    Truncating the data
    '''
    sim_N_dist, w_values = cosfuncs.histogram_data_truncation(sim_N_dist,\
                              N_f, weights=w_values,\
                              num_sub_samples = num_sub_samples)
    
    '''
    #Post processesing
    '''
    
    if save_data == True:
        data_dict_raw = {}
        data_dict_raw['N'] = sim_N_dist
        if bias>0:
            data_dict_raw['w'] = w_values
        
        data_pandas_raw = pd.DataFrame(data_dict_raw)
        
        raw_file_name = 'IS_data_phi_i_'+('%s' % float('%.3g' % phi_i))+\
            '_iterations_'+str(num_sims)+'_bias_'+\
            ('%s' % float('%.3g' % bias))+'.csv'
        #Saving to a directory for the language used

        data_pandas_raw.to_csv(raw_file_name)

    
    #
    #Now analysisng creating the PDF data
    #
    
    if bias>0.2:
        bin_centres,heights,errors,num_sims_used, bin_edges_untruncated =\
            data_points_pdf(sim_N_dist, w_values, num_sub_samples,\
            bins=bins, include_std_w_plot = False,\
            min_bin_size = min_bin_size, log_normal = log_normal,\
            num_sims = num_sims, log_normal_method='ML', w_hist_num = 10,\
            p_value_plot = True)
    #I'm not convinced 
    elif bias<=0.2:
        bin_centres,heights,errors,num_sims_used,_ =\
            data_points_pdf(sim_N_dist, w_values, num_sub_samples,\
            bins=bins, min_bin_size = min_bin_size, log_normal = False,\
            num_sims = num_sims)
        #If using log normal, need to make the errors asymmetric for later
        #data analysis
        if log_normal == True:
            errors_new = np.zeros((2,len(errors)))
            errors_new[0,:] = errors
            errors_new[1:] = errors
            errors = errors_new

    return bin_centres, heights, errors



def data_points_pdf(Ns, ws, num_sub_samples, min_bin_size = None, bins = 50,\
                    log_normal = False, include_std_w_plot = False,\
                    w_hist_num = None, log_normal_method = 'ML',\
                    num_sims = None, p_value_plot = False):
    #If no number of simulations argument is passed.
    if isinstance(num_sims, int) != True:
        num_sims = len(Ns)
    
    #If the number of bins used has been specified
    if isinstance(bins, int) == True:
        num_bins = bins
        #Want raw heights of histogram bars
        heights_raw,bins,_ =\
            plt.hist(Ns, num_bins, weights = ws)
        plt.clf()
    #If the bins have been specified
    else:
        num_bins = len(bins)-1# as bins is the bin edges, so plus 1
        #Want raw heights of histogram bars
        heights_raw, bins,_ =\
            plt.hist(Ns, bins = bins, weights = ws)
        plt.clf()
        
        
    analytical_norm =\
        cosfuncs.histogram_analytical_normalisation(bins, num_sims)
        
    data_in_bins, weights_in_bins =\
        cosfuncs.histogram_data_in_bins(Ns, ws, bins)
    
    #Predictions need the bin centre to make comparison
    bin_centres = np.array([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)])
    
    
    #Removing underfilled bins if needed
    if isinstance(min_bin_size, int) == True:
        #Then loop through to find where length is greater than min_bin_size
        filled_bins = []
        num_sims_used = len(Ns) 
        for i in range(len(bins)-1):
            data_in_bin = data_in_bins[:,i]
            data_in_bin = data_in_bin[data_in_bin>0]
            #First, remove if empty 
            if len(data_in_bin)==0:
                filled_bins.append(False)
            #If there is enough data in this bin
            elif len(data_in_bin)>=min_bin_size:
               filled_bins.append(True) 
            #Don't include under filled tail bins
            else:
                filled_bins.append(False)
                #Reflect in number of succesful simulatios
                num_sims_used -= len(data_in_bin)
        #Make the Boolean logic into a numpy array
        filled_bins = np.array(filled_bins)
        bin_centres_uncut = bin_centres
        bin_centres = bin_centres[filled_bins]
        
    if log_normal == False:
        heights = heights_raw/analytical_norm
        errors =\
            cosfuncs.histogram_weighted_bin_errors_jackknife(Ns, ws,\
            bins, num_sub_samples)
        if isinstance(min_bin_size, int) == True:
            heights = heights[filled_bins]
            errors = errors[filled_bins]
    elif log_normal == True:
        
        heights_est = np.zeros(num_bins)
        #The errors for the log-normal case are asymmetric
        errors_est = np.zeros((2,num_bins))
        for i in range(num_bins):
            w = weights_in_bins[:,i]
            #Only calculate filled bins
            if filled_bins[i] == True or\
                (np.any([w>0]) == True and\
                 isinstance(min_bin_size, int) == False):
                w = w[w>0]
                heights_est[i] =\
                    log_normal_height(w, method = log_normal_method,\
                                      position = bin_centres_uncut[i])
                errors_est[:,i] = log_normal_errors(w, method = log_normal_method)
                
        #Include only filled values   
        #Remember to normalise errors as well
        heights = heights_est[errors_est[0,:]>0]/analytical_norm
        #The errors are a 2D array, so need to slice correctly
        errors = errors_est[:,errors_est[0,:]>0]/analytical_norm
        if log_normal_method == 'jackknife':
            errors = log_normal_errors_jackknife(Ns, ws, bins, num_sub_samples)
            errors = errors[:,filled_bins]
    else:
        raise ValueError('log_normal must be boolean')
        
    #If plotting the variance of w in different bins
    if include_std_w_plot == True:
        #Find the variance of log(w)
        std_log10_w = np.zeros(len(bins)-1)
        for i in range(len(bins)-1):
            #Only use data in the filled bins
            if filled_bins[i] == True:
                w_in_bin = weights_in_bins[:,i]
                #Remove zeros
                w_in_bin = w_in_bin[w_in_bin>0]
                w_in_bin_log10 = np.log10(w_in_bin)
                std_log10_w[i] = np.std(w_in_bin_log10)
                
        std_log10_w = std_log10_w[std_log10_w>0]
        
        plt.errorbar(bin_centres, std_log10_w, fmt =".k")
        plt.title( str(num_sims_used))
        plt.xlabel(r'$\mathcal{N}$')
        plt.ylabel(r'$log_{10}(\sigma_w)$')  
        plt.show()
        plt.clf()
        
    if isinstance(w_hist_num, int) == True:
        w = weights_in_bins[:,w_hist_num]
        log_w = np.log(w[w>0])
        num_ws = len(log_w)
        
        mean = np.mean(log_w)
        std = np.std(log_w)
        skew = sci_stat.skew(log_w)
        kurtosis = sci_stat.kurtosis(log_w)
        _,p = sci_stat.normaltest(log_w)
        
        cumulant_3 = sci_stat.kstat(log_w, n=3)
        cumulant_4 = sci_stat.kstat(log_w, n=4)
        
        '''
        def edgeworth_expansion(x, cum3, cum4):
            return cosfuncs.pdf_gaussian_skew_kurtosis(x, mean,\
                                std, cum3, cum4)

                
        heights_log_w, bins_log_w = np.histogram(log_w, bins=50, density=True)
        bin_centres_log_w =\
            np.array([(bins_log_w[i]+bins_log_w[i+1])/2 for i in range(len(bins_log_w)-1)])

            
        edgeworth_params, cv =\
            optimize.curve_fit(edgeworth_expansion, bin_centres_log_w,\
                                 heights_log_w,\
                                p0 = (cumulant_3, cumulant_4))
        '''
        #skew normal distribution, from the wikipedia page
        #https://en.wikipedia.org/wiki/Skew_normal_distribution
        skew_mod = np.abs(skew)
        delta = np.sign(skew)*np.sqrt( 0.5*PI*np.divide(skew_mod**(2/3),\
                                    skew_mod**(2/3) + (2+0.5*PI)**(2/3)) )
        alpha = delta/np.sqrt(1-delta**2)
        omega = std/np.sqrt(1-2*(delta**2)/PI)
        xi = mean - omega*delta*np.sqrt(2/PI)
        
        #skew_error = cosfuncs.skewness_std(len(log_w))
        #kurtosis_error = cosfuncs.kurtosis_std(len(log_w))
        _,log_w_bins,_ = plt.hist(log_w, bins = 50,\
                                  histtype='step', linewidth=2,\
                        label='{0}'.format(r'$\mathcal{N}_{\rm cen}=$ '+\
                        ('%s' % float('%.5g' % bin_centres[w_hist_num]))+', std = '+\
                            ('%s' % float('%.3g' % np.std(log_w)))))
        bin_width = np.diff(log_w_bins)[0]
        inverse_norm = num_ws*bin_width

        print(np.std(log_w))
        print('num data in bin is'+str(len(log_w)))
        plt.plot(log_w_bins,\
            inverse_norm*sci_stat.norm.pdf(log_w_bins, mean, std),\
            label='{0}'.format('Gaussian fit'), linewidth=2, color='#ff7f00')
        
        '''
        plt.plot(log_w_bins,\
            inverse_norm*sci_stat.skewnorm.pdf(log_w_bins, alpha, xi, omega),\
            label='{0}'.format(r'skew normal 2' ) )
        plt.plot(log_w_bins,\
            inverse_norm*cosfuncs.pdf_gaussian_skew_kurtosis(log_w_bins, mean,\
            std, cumulant_3, cumulant_4),\
            label='{0}'.format(r'Edgeworth' ) )
        plt.title(r'$\mathcal{N}_{\rm cen}$='+\
                  ('%s' % float('%.5g' % bin_centres[w_hist_num]))+\
                      r', #'+str(len(log_w))+r', p-value = '\
                    +('%s' % float('%.3g' % p)) )
        '''
        plt.legend()
        plt.xlabel(r'${\rm ln}(w)$', fontsize = 20)
        plt.ylabel('Counts', fontsize = 20)
        plt.title(r'$p=1.3$ and $\mathcal{A}=1$')
        #plt.savefig('Cython_results/'+'ln_w_plot1.pdf',transparent=True)
        plt.yscale('log')
        
        plt.margins(tight=True)
        '''
        plt.savefig('for_paper/'+'log_w_plot_at_'+\
                    ('%s' % float('%.5g' % bin_centres[w_hist_num]))\
                    +'_bias_0.8_m_1_1phi_i.pdf', transparent=True, dpi=800)
        
        '''
        plt.show()
        plt.clf()
        
    if p_value_plot == True:
        p_values = np.zeros(num_bins)
        p_values_theory = np.zeros(num_bins)
        under_filled_bins = False
        for i in range(len(p_values)):
            if filled_bins[i] == True:
                w = weights_in_bins[:,i]
                log_w = np.log(w[w>0])
                _,p_values[i] = sci_stat.normaltest(log_w)
                _,p_values_theory[i] =\
                    sci_stat.normaltest(np.random.normal(0,1,len(log_w)))
                    
                if under_filled_bins == False and len(log_w)<5000 and\
                    i>int(len(p_values)/4):
                    under_filled_bins = bin_centres_uncut[i]
                    
        p_values = p_values[filled_bins]
        p_values_theory = p_values_theory[filled_bins]
        print(p_values[w_hist_num])
        
        plt.errorbar(bin_centres, p_values, fmt = '.', ms=7)
        plt.hlines(0.005, np.min(bin_centres), np.max(bin_centres),\
                   color = 'k', linestyle = 'dashed',\
                      label='{0}'.format('0.5% threshold'), linewidth=2)
        '''
        if under_filled_bins != False:
            plt.axvline(x=under_filled_bins,\
                       color = 'k', linestyle = 'dashdot',\
                          label='{0}'.format('<5000'))
        '''
        plt.yscale('log')
        #plt.title('Data: p-values with bin centres')
        plt.legend(fontsize = 20)
        plt.xlabel(r'$\mathcal{N}$', fontsize = 20)
        plt.ylabel('p-values', fontsize = 20)
        '''
        plt.margins(tight=True)
        plt.savefig('for_paper/'+'p_values_m_1_bias_0.8.pdf',\
                    transparent=True, dpi=800)
        
        '''
        plt.show()
        plt.clf()
        
        plt.errorbar(bin_centres, p_values_theory, fmt = '.')
        plt.hlines(0.005, np.min(bin_centres), np.max(bin_centres),\
                   color = 'k', linestyle = 'dashed',\
                      label='{0}'.format('0.5% threshold'))
        plt.yscale('log')
        plt.title('Theoretical: p-values with bin centres')
        plt.legend(fontsize=22)
        plt.xlabel(r'$\mathcal{N}$', fontsize=22)
        plt.ylabel('p-values', fontsize=22)
        plt.show()
        plt.clf()
        
    return bin_centres, heights, errors, num_sims_used, bins
    

def log_normal_height(w, method = 'ML', position = None):
    return len(w)*cosfuncs.log_normal_mean(w, method, position = position)

#The most basic way to estimate the error, assuming symmetric errors
#SHOULD THIS USE n-1 or n IN THE STD??????????????????????????
def log_normal_errors(ws, method = 'ML', Z_alpha=1, B=10**3):
    log_w = np.log(ws)
    log_var = np.var(log_w, ddof = 1)#unbiased variance
    log_mean = np.mean(log_w)
    n = len(ws)
    if method == 'Shen':
        mean_est = cosfuncs.log_normal_mean(ws, method)
        std_tau = np.sqrt(log_var/n +\
            np.divide(8*(n-1)*((n+4)*log_var)**2,(3*log_var+2*(n+4))**4))
        Normal = np.random.normal(0, 1, B)
        Chi = np.random.chisquare(n-1, B)
        T_values =\
            [T_shen(log_var**0.5, Normal[i], Chi[i], n) for i in range(B)]
        T_values = np.array(T_values)
        T_values = np.sort(T_values)
        
        alpha = 1-0.68#THE CONFIDANCE LEVEL!!!!!!!!!!!!
        t_1 = T_values[int(0.5*alpha*B)]
        t_2 = T_values[int((1-0.5*alpha)*B)]
        lower_err = n*(1-np.exp(-t_2*std_tau))*mean_est
        upper_err = n*(np.exp(-t_1*std_tau)-1)*mean_est
        
        return np.array([lower_err, upper_err])
        
    elif method == 'naive':
        error = n*(np.exp(log_var)-1)*np.exp(2*log_mean+log_var)/np.sqrt(n)
        return np.array([error,error])
    elif method == 'ML_bootstrap':
        mean_est = cosfuncs.log_normal_mean(ws, method)
        std_tau = np.sqrt(log_var/n +\
            np.divide(8*(n-1)*((n+4)*log_var)**2,(3*log_var+2*(n+4))**4))
        Normal = np.random.normal(0, 1, B)
        Chi = np.random.chisquare(n-1, B)
        T_values = [T_ML(log_var**0.5, Normal[i], Chi[i], n) for i in range(B)]
        T_values = np.array(T_values)
        T_values = np.sort(T_values)
        
        alpha = 1-0.68#THE CONFIDANCE LEVEL!!!!!!!!!!!!
        t_1 = T_values[int(0.5*alpha*B)]
        t_2 = T_values[int((1-0.5*alpha)*B)]
        lower_err = n*(1-np.exp(-t_2*std_tau))*mean_est
        upper_err = n*(np.exp(-t_1*std_tau)-1)*mean_est
        
        return np.array([lower_err, upper_err])
    else:#I.e. default to ML
        log_err = Z_alpha*np.sqrt(log_var/n+(log_var**2)/(2*n-2))
        upper_err = n*np.exp(log_mean+log_var/2)*(np.exp(log_err)-1)
        lower_err = n*np.exp(log_mean+log_var/2)*(1-np.exp(-log_err))
        return np.array([lower_err, upper_err])
    
def log_normal_errors_jackknife(Ns, ws, bins, num_sub_samps):
    return cosfuncs.histogram_weighted_bin_errors_jackknife(Ns, ws, bins,\
                                            num_sub_samps, lognormal = True)

#From Eq. (7) of Shen2006 Statist. Med. 2006; 25:3023–3038.
#Assuming n-1 of does not cancel
def T_shen(sigma, N, C, n):
    n_minus_1 = n-1
    C_p = C/n_minus_1 
    n_plus_4 = n+4
    numerator = N + 0.5*sigma*np.sqrt(n)*\
        (np.divide(2*n_minus_1*C_p, 2*n_plus_4+3*C_p*sigma**2)-1)
    denuminator = C_p+\
        np.divide(8*n*n_minus_1*(sigma*n_plus_4*C_p)**2,\
                  (3*C_p*sigma**2+2*n_plus_4)**4)
            
    return numerator/np.sqrt(denuminator)

#From Eq. (3) of Zhou STATISTICS IN MEDICINE, VOL. 16, 783Ð790 (1997)
def T_ML(sigma, N, C, n):
    n_minus_1 = n-1
    C_p = C/n_minus_1 
    numerator = N + 0.5*sigma*np.sqrt(n)*(C_p-1)
    denuminator = C_p*(1+0.5*C_p*sigma**2)
            
    return numerator/np.sqrt(denuminator)
            
def large_mass_pdf(bin_centres, phi_i, phi_end, V):
    def chi(t):
        return cosfuncs.chaotic_inflation_characteristic_function(t,phi_i,\
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
        np.array([2*cosfuncs.continuous_ft(N,chi, component = 'real',t_lower = 0,\
                t_upper = t0max)/(2*PI)**0.5 for N in bin_centres])
    return PDF_analytical_test



