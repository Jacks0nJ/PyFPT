# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:41:37 2021

@author: user
"""

import numpy as np
import pandas as pd

import scipy.stats as sci_stat
from timeit import default_timer as timer
import scipy.optimize


import inflation_functions_e_foldings as cosfuncs
import is_data_analysis2 as isfuncs
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mpl_style
plt.style.use(mpl_style.style1)

#M_PL = 2.435363*10**18 old value
M_PL = 1.0# Using units of M_PL
PI = np.pi
#m = 10**(-6)*M_PL#Based on John McDonald's calculations in PHYS461
m = 0.1*M_PL#4*PI*6**0.5

###Intial conditions and tolerances
N_starting = 10#In some sense, this should techically be negative
phi_end = M_PL*2**0.5
phi_i = M_PL*(4*N_starting+2)**0.5#M_PL*(4*N_starting+2)**0.5
phi_r = 100*phi_i
N_cut_off = 300
N_f = 100
dN = 0.02*m#Assuming std(N) is proportional to m, was dN=0.02m
num_sims = 100000
num_bins = 50
num_sub_samples = 20


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

min_bin_size = 400
fit_threshold = 100
#Minimum number of std before exponential tail can be fit
min_tail_start = 4


include_errors = 'yes'
tail_analysis = False
edgeworth_series = False
manual_norm = True
w_hist = False
save_results = True
save_raw_data = True
log_normal = True
contour = False
fontsize = 20
include_std_w_plot = True
save_plots = True
count_refs = False
scater_density_plot = True

wind_type = 'diffusion'
bias = 1


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


analytic_N_var =\
    cosfuncs.delta_N_squared_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)
analytic_N_st = np.sqrt(analytic_N_var)
analytic_N_mean = cosfuncs.mean_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)
analytic_N_skew =\
    cosfuncs.skewness_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)
analytic_N_kurtosis =\
    cosfuncs.kurtosis_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)
analytic_N_4th_cmoment =\
    cosfuncs.fourth_central_moment_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)
eta_criterion = cosfuncs.classicality_criterion(V, V_dif, V_ddif, phi_i)

        
N_star = analytic_N_mean + 4*analytic_N_st

analytic_gauss_deviation_pos =\
    cosfuncs.gaussian_deviation(analytic_N_mean, analytic_N_var**0.5,\
    analytic_N_skew*analytic_N_var**1.5,\
    analytic_N_4th_cmoment-3*analytic_N_var**2, nu=fit_threshold/100)
'''
#Running the simulation many times
'''

bin_centres,heights,errors =\
    isfuncs.IS_simulation(phi_i, phi_end, V, V_dif, V_ddif, num_sims, bias,\
    bins=50, dN = dN, reconstruction = 'lognormal', save_data = True,\
    phi_UV = False)
        
'''
Reading the saved data
'''
raw_file_name = 'IS_data_phi_i_'+('%s' % float('%.3g' % phi_i))+\
            '_iterations_'+str(num_sims)+'_bias_'+\
            ('%s' % float('%.3g' % bias))+'.csv'
raw_data = pd.read_csv(raw_file_name, index_col=0)
sim_N_dist = np.array(raw_data['N'])
if bias>0:
    w_values = np.array(raw_data['w'])

    
bin_centres_analytical =\
        np.linspace(bin_centres[0], bin_centres[-1],2*num_bins)
    

if m>0.6:#Less than this it breaks down:
    start = timer()
    PDF_analytical_test = isfuncs.large_mass_pdf(bin_centres_analytical,phi_i,phi_end,V)
    end = timer()
    print(f'The analytical answer took: {end - start}') 
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

    data_pandas_results.to_csv(my_file_name)
    #Remembering to remove column numbering
    sim_data = pd.read_csv(my_file_name,\
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
    


    



'''
Fitting models to the tail
'''


        
        
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



if bias != 0:
    _,_,_ = plt.hist(sim_N_dist, num_bins, weights = w_values,\
                             density=True, label='{0}'.format('Weighted bins') ) 
    histogram_name = 'N_distribution_for_' + 'IS_near_' +\
    str(N_starting)+'_dN_' + str(dN) + '_m_'+('%s' % float('%.3g' % m)) +'_Is_shift_'+str(bias)+ '_iterations_' +\
        str(num_sims)+'_wighted.pdf'
    if m<=0.6:
        dist_fit_label2 = r'Analytical $\sqrt{\delta \mathcal{N}^2}$='+str(round(analytic_N_st,4))
    else:
        dist_fit_label2 = r'Pattison 2017'
    plt.plot(bin_centres_analytical, best_fit_line2, label='{0}'.format(dist_fit_label2))
    plt.title(r'bias=' + str(bias) +', '+str(num_sims) + ' sims, ' +\
              r'dN=' + ('%s' % float('%.3g' % dN)) +\
              r', $m$=' + ('%s' % float('%.3g' % m)) )
    plt.xlabel(r'$\mathcal{N}$')
    plt.ylabel(r'$\mathcal{N}$')
    plt.legend()
    #Including if I have used importance sampling
    #Saving to a directory for the language used
    if save_plots == True:
        plt.savefig(histogram_name,transparent=True)
    plt.show()
    plt.clf()

    
if bias != 0:
    #Plotting the weights, and number of reflections if appropriate
    scatter_name = '_dN_' + str(dN) +'_m_'+\
            ('%s' % float('%.3g' % m)) +'_phi_UV_'+\
            str(phi_r/phi_i)+ '_m_'+ ('%s' % float('%.3g' % m)) +\
            '_iterations_' + str(num_sims)+'_bias_' + str(bias)
    if contour == True:
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
              
    if save_plots == True:
        plt.xlabel(r'$\mathcal{N}$', fontsize = fontsize)
        plt.ylabel(r'${\rm log}_{10}(w)$', fontsize = fontsize)
        plt.margins(tight=True)
        plt.savefig(scatter_name, transparent=True, dpi=800)
    plt.show()
    plt.clf()
    
          
    
    #Plotting the log of the distribution
    histogram_name= 'publishable_error_bar_IS_near_'\
        +str(N_starting)+'_dN_' + ('%s' % float('%.2g' % dN)) + '_m_'+\
        ('%s' % float('%.3g' % m)) +'_phi_UV_'+str(phi_r/phi_i)+'phi_i'+\
        '_bias_'+str(bias)+'_iters_' + str(num_sims) +'_bin_size_' +\
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
    plt.savefig(histogram_name+'.pdf', transparent=True)
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
    plt.savefig(histogram_name,transparent=True)
    plt.show()
    plt.close()
    



