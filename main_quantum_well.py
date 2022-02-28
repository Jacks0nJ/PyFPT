# -*- coding: utf-8 -*-python3 setup.py build_ext --inplace
"""
Created on Wed Mar  3 17:05:02 2021
    Main to run a quantum well simulation of inflation.
@author: user
"""


import numpy as np
import pandas as pd
import zeus
import arianna as arn
import scipy.stats as sci_stat
from scipy.optimize import minimize
from timeit import default_timer as timer
import multiprocessing as mp
from multiprocessing import Process, Queue

import inflation_functions_e_foldings as cosfuncs
import stochastic_inflation_cosmology as sm
import is_data_analysis as isfuncs
import matplotlib.pyplot as plt
import mpl_style
from matplotlib.colors import LogNorm
#import importance_sampling_sr_cython3 as test_cy
import tilted_quantum_well_cython19 as qw_cy

#M_PL = 2.435363*10**18 old value
M_PL = 1.0# Using units of M_PL
PI = np.pi

#The tilted quantum well initial parameters
phi_end = 1.0*M_PL
mu = 0.2#The normalised width of the well
x = 0.99#Where you start the simulation
V_0 = 0.0001
v_0 = V_0/(24*(PI**2)*(M_PL**4))
delta_phi = M_PL*mu*(v_0**0.5)
phi_i = x*delta_phi+phi_end
phi_r = delta_phi+phi_end
tilt = 0*(v_0**0.5)/(mu*M_PL)
well_type = 'wide'
#The resulting parameters for the tilted quantum well
a_i = 0.0

#The simulation parameters
alpha = 0.02
bc_type = 'reflective'
wind_type = 'one way'
count_reflects = 'yes'
saving_data = False
scater_density_plot = True
log_normal = False
bias = 0#Was 18
Using_IS = 'yes'
step_type = 'Euler_Maruyama'

min_bin_size = 40
fit_threshold = 800
num_bins = 50
num_sub_samples = 100
N_threshold = 45
#Minimum number of std before exponential tail can be fit
min_tail_start = 4

num_sims = int(200000/mp.cpu_count())#As num_sims is done per core
dN = 1.5*(alpha*M_PL*mu)**2#Using my formula
N_i = 0.0
N_f = 40
n = 40

bayesian = 'no'
def qw_po(phi):
    V = V_0
    return V

def qw_po_dif(phi):
    V_dif = 0.0
    return V_dif

def qw_po_ddif(phi):
    V_ddif = 0.0
    return V_ddif

def classical_end_cond(matrices, N, phi_end_infl = phi_end):
    cond = False
    if matrices[0,0] <= phi_end_infl:
        cond = True
    
    return cond

def symmertic_end_cond(matrices, N, phi_end_infl = phi_end,\
                       width = 2*delta_phi):
    cond = False
    if (matrices[0,0]<=phi_end_infl) or (matrices[0,0]>=phi_end_infl+width):
        cond = True
    
    return cond

def reflection_condition_well(matrices, N, phi_r = delta_phi+phi_end):
    cond = False
    if matrices[0,0] >= phi_r:
        cond = True
        matrices_reflected = np.copy(matrices)
        matrices_reflected[0,0] = 2*phi_r - matrices[0,0]
    else:
        matrices_reflected = None
        
    return cond, matrices_reflected


quantum_well = sm.Stochastic_Inflation(qw_po, qw_po_dif, qw_po_ddif,\
                                       classical_end_cond, a_i,\
                                       reflection_cond =\
                                       reflection_condition_well, mu = mu)
    
quantum_well_sym = sm.Stochastic_Inflation(qw_po, qw_po_dif, qw_po_ddif,\
                                       symmertic_end_cond, a_i, mu = mu)

start = timer()
#Using multiprocessing
def multi_processing_func(mu, x, tilt, N_i, N_f, dN, bias, num_sims,\
                          queue_Ns, queue_ws, queue_ref_counts,\
                              count_reflects = 'no'):
    #Doing the double absorbing surface code
    qw_results = qw_cy.many_simulations_importance_sampling(mu, x, tilt, N_i,\
                N_f, dN, bias, num_sims, boundary_type = bc_type,\
                    count_reflects = count_reflects, wind_type = wind_type)
    Ns = np.array(qw_results[0][:])
    ws = np.array(qw_results[1][:])
    queue_Ns.put(Ns)
    queue_ws.put(ws)
    if count_reflects == 'yes':
        ref_counts = np.array(qw_results[2][:])
        queue_ref_counts.put(ref_counts)

if __name__ == "__main__":
    queue_Ns = Queue()
    queue_ws = Queue()
    queue_ref_counts = Queue()
    cores = int(mp.cpu_count()/1)#mp.cpu_count()

    print('Number of cores used: '+str(cores))
    processes = [Process(target=multi_processing_func,\
                args=(mu, x, tilt, N_i, N_f, dN, bias, num_sims,\
                queue_Ns, queue_ws, queue_ref_counts, count_reflects)) for i in range(cores)]
    for p in processes:
        p.start()
    
    #for p in processes:
     #   p.join()
    
    Ns_array = np.array([queue_Ns.get() for p in processes])
    ws_array = np.array([queue_ws.get() for p in processes])
    if count_reflects == 'yes':
        ref_counts_array = np.array([queue_ref_counts.get() for p in processes])
    #As there num_sims on the cores
    num_sims = cores*num_sims
#Combine into columns into 1
qw_sim_N_dist  = Ns_array.flatten()
qw_sim_w_values = ws_array.flatten()
if count_reflects == 'yes':
    qw_sim_ref_counts = ref_counts_array.flatten()
'''
qw_sim_results = qw_cy.many_simulations_importance_sampling(mu, x, tilt, N_i,\
                                                           N_f, dN, bias,\
                                                               num_sims)
#Extract the data to numpy arrays    
qw_sim_N_dist = np.array(qw_sim_results[0][:])
qw_sim_w_values = np.array(qw_sim_results[1][:])
'''


#Sort in order of increasing Ns
sort_idx = np.argsort(qw_sim_N_dist)
qw_sim_N_dist = qw_sim_N_dist[sort_idx]
qw_sim_w_values = qw_sim_w_values[sort_idx]
if count_reflects == 'yes':
    qw_sim_ref_counts = qw_sim_ref_counts[sort_idx]

end = timer()
print(f'The simulations took: {end - start}')

'''
Truncating the data
'''
    
qw_sim_N_dist, qw_sim_w_values = cosfuncs.histogram_data_truncation(qw_sim_N_dist,\
                          N_threshold, weights=qw_sim_w_values,\
                          num_sub_samples = num_sub_samples)
num_sims = len(qw_sim_N_dist)
num_sims_used = num_sims
if count_reflects == 'yes':
    qw_sim_ref_counts = qw_sim_ref_counts[:num_sims]


'''
Saving data
'''

if saving_data == True:
    qw_data_dict = {}
    qw_data_dict['Ns'] = qw_sim_N_dist
    if Using_IS == 'yes':
        qw_data_dict['ws'] = qw_sim_w_values
    if count_reflects == 'yes':
        qw_data_dict['reflections'] = qw_sim_ref_counts
    qw_data_pandas = pd.DataFrame(qw_data_dict)
    
    qw_file_name = 'qw_mu_'+str(mu)+'_x_' +str(x)+'_bias_'+str(bias)+'_tilt_'+\
        str(tilt) + '_dN_'+('%s' % float('%.5g' % dN))+'_iterations_'+\
            str(num_sims)+'_'+step_type+'.csv'
    qw_data_pandas.to_csv('Cython_qw_results/'+qw_file_name)
    #Loading from file. Remembering to remove column numbering
    qw_sim_data = pd.read_csv('Cython_qw_results/'+qw_file_name, index_col=0)



'''
Post processesing
'''


if Using_IS == 'yes':
    sim_N_mean = cosfuncs.importance_sampling_mean(qw_sim_N_dist, qw_sim_w_values)
    sim_N_var = cosfuncs.importance_sampling_var(qw_sim_N_dist, qw_sim_w_values)
    sim_N_skew = cosfuncs.importance_sampling_skew(qw_sim_N_dist, qw_sim_w_values)
    sim_N_kurtosis = cosfuncs.importance_sampling_kurtosis(qw_sim_N_dist, qw_sim_w_values)
    
    sim_mean_error = cosfuncs.jackknife(qw_sim_N_dist, int(num_sims/num_sub_samples),\
                    cosfuncs.importance_sampling_mean, weights = qw_sim_w_values)
    sim_var_error = cosfuncs.jackknife(qw_sim_N_dist, int(num_sims/num_sub_samples),\
                    cosfuncs.importance_sampling_var, weights = qw_sim_w_values)
    sim_skew_error = cosfuncs.jackknife(qw_sim_N_dist, int(num_sims/num_sub_samples),\
                    cosfuncs.importance_sampling_skew, weights = qw_sim_w_values)
    sim_kurtosis_error = cosfuncs.jackknife(qw_sim_N_dist, int(num_sims/num_sub_samples),\
                    cosfuncs.importance_sampling_kurtosis, weights = qw_sim_w_values)
else:#When not using importance sampling
    #The statistics of the data
    sim_N_mean, sim_N_st = sci_stat.norm.fit(qw_sim_N_dist)
    sim_N_var = sim_N_st**2
    sim_N_skew = sci_stat.skew(qw_sim_N_dist)
    sim_N_kurtosis = sci_stat.kurtosis(qw_sim_N_dist)
    
    sim_mean_error = cosfuncs.jackknife(qw_sim_N_dist, int(num_sims/num_sub_samples), np.mean)
    sim_st_error = cosfuncs.jackknife(qw_sim_N_dist, int(num_sims/num_sub_samples), np.std)
    sim_skew_error = cosfuncs.jackknife(qw_sim_N_dist, int(num_sims/num_sub_samples),\
                                        sci_stat.skew)
    sim_kurtosis_error = cosfuncs.jackknife(qw_sim_N_dist, int(num_sims/num_sub_samples),\
                                        sci_stat.kurtosis)




if bias>0.2:
    bin_centres,heights,errors,num_sims_used =\
        isfuncs.data_points_pdf(qw_sim_N_dist, qw_sim_w_values, num_bins,\
        num_sub_samples, include_std_w_plot = True,\
        min_bin_size = min_bin_size, log_normal = log_normal,\
        num_sims = num_sims, log_normal_method='ML', w_hist_num = 10,\
        p_value_plot = True)
#I'm not convinced 
elif bias<=0.2:
    bin_centres,heights,errors,num_sims_used =\
        isfuncs.data_points_pdf(qw_sim_N_dist, qw_sim_w_values, num_bins,\
        num_sub_samples, min_bin_size = min_bin_size, log_normal = False,\
        num_sims = num_sims, w_hist_num = 10)
    #If using log normal, need to make the errors asymmetric for later
    #data analysis
    if log_normal == True:
        errors_new = np.zeros((2,len(errors)))
        errors_new[0,:] = errors
        errors_new[1:] = errors
        errors = errors_new



#Expected values
if tilt==0.0:
    qw_analytic_N_prob_dist =\
        quantum_well_sym.quantum_diffusion_N_probability_dist(bin_centres,\
                                                      x, n)
    qw_analytic_N_mean = quantum_well_sym.quantum_diffusion_mean_N(x)
    qw_analytic_N_var = quantum_well_sym.quantum_diffusion_var_N(x)
      
elif well_type == 'wide' and tilt != 0.0:
    qw_analytic_N_prob_dist =\
        cosfuncs.wide_tilted_well_N_probability_dist(bin_centres,\
                                                     x, tilt, delta_phi, v_0, n)
else:
    qw_analytic_N_prob_dist =\
        quantum_well.quantum_diffusion_N_probability_dist(bin_centres,\
                                                      x, n)
    qw_analytic_N_mean = quantum_well.quantum_diffusion_mean_N(x)
    qw_analytic_N_var = quantum_well.quantum_diffusion_var_N(x)
    
    

'''
Plotting
'''
plt.style.use(mpl_style.style1)

if Using_IS == 'yes':
    _,bins,_ = plt.hist(qw_sim_N_dist, weights = qw_sim_w_values ,\
                        bins=50, density=True,\
             label='{0}'.format(r'$x=$'+('%s' % float('%.5g' % x)) + r', $\mu$='\
                + ('%s' % float('%.5g' % mu))))
    if tilt == 0:
        dist_fit_label = r'Analytical, $\langle\mathcal{N}\rangle = $' +\
            ('%s' % float('%.5g' % qw_analytic_N_mean))
        plt.title( str(num_sims) + r', weighted, bias=' +str(bias)+r', $dN$=' +\
              ('%s' % float('%.2g' % dN)))
        plt.axvline(sim_N_mean, color='k', linestyle='dashed',
                linewidth=2, label='{0}'.format(r'$\langle\mathcal{N}\rangle$=' +\
                ('%s' % float('%.5g' % sim_N_mean)) +\
                    ', IS estimate'))
    elif tilt!=0:
        dist_fit_label = r'Ezquiaga 2020'
        plt.title( str(num_sims) + r', dN=' +\
              ('%s' % float('%.2g' % dN)) +r', $\gamma=$' + ('%s' % float('%.2g' % tilt)))
        plt.axvline(sim_N_mean, color='k', linestyle='dashed',
                linewidth=2, label='{0}'.format(r'$\langle\mathcal{N}\rangle$=' +\
                ('%s' % float('%.5g' % sim_N_mean)) + r'$\pm$' +\
                ('%s' % float('%.2g' % sim_mean_error))))
    plt.plot(bin_centres,\
             qw_analytic_N_prob_dist,label='{0}'.format(dist_fit_label))
    
    plt.xlabel(r'$\mathcal{N}$')
    plt.ylabel('Probability Density')
    plt.legend()
    histogram_name = 'qw_N_distribution_for_' +'tilt_' + ('%s' % float('%.2g' % tilt))+'_x_' +\
        ('%s' % float('%.5g' % x))+'_mu_'+str(mu)+'_dN_' + ('%s' % float('%.3g' % dN))+\
            '_iterations_'+str(num_sims)+'_'+step_type+'_'+bc_type+'_weighted.pdf'
    plt.savefig('Cython_qw_results/'+histogram_name,transparent=True)
    plt.show()
    plt.clf()


height_unweighted,bins,_ = plt.hist(qw_sim_N_dist,\
                    bins=50, density=True,\
         label='{0}'.format(r'$x=$'+('%s' % float('%.5g' % x)) + r', $\mu$='\
            + ('%s' % float('%.5g' % mu))))
if tilt == 0 and Using_IS != 'yes':
    dist_fit_label = r'Analytical, $\langle\mathcal{N}\rangle = $' +\
        ('%s' % float('%.5g' % qw_analytic_N_mean))
    plt.title( str(num_sims) + r', $dN$=' +\
          ('%s' % float('%.2g' % dN)) +r', $\gamma=$' + ('%s' % float('%.2g' % tilt)))
    plt.axvline(sim_N_mean, color='k', linestyle='dashed',
            linewidth=2, label='{0}'.format(r'$\langle\mathcal{N}\rangle$=' +\
            ('%s' % float('%.5g' % sim_N_mean)) + r'$\pm$' +\
            ('%s' % float('%.2g' % sim_mean_error))))
elif tilt == 0 and Using_IS == 'yes':
    dist_fit_label = r'Analytical, $\langle\mathcal{N}\rangle = $' +\
        ('%s' % float('%.5g' % qw_analytic_N_mean))
    plt.title( str(num_sims) + r', unweighted, bias=' +str(bias)+r', $dN$=' +\
          ('%s' % float('%.2g' % dN)))
    plt.axvline(sim_N_mean, color='k', linestyle='dashed',
            linewidth=2, label='{0}'.format(r'$\langle\mathcal{N}\rangle$=' +\
            ('%s' % float('%.5g' % sim_N_mean)) +\
                ', IS estimate'))
elif tilt!=0 and bias==0:
    dist_fit_label = r'Ezquiaga 2020'
    plt.title( str(num_sims) + r', dN=' +\
          ('%s' % float('%.2g' % dN)) +r', $\gamma=$' + ('%s' % float('%.2g' % tilt)))
    plt.axvline(sim_N_mean, color='k', linestyle='dashed',
            linewidth=2, label='{0}'.format(r'$\langle\mathcal{N}\rangle$=' +\
            ('%s' % float('%.5g' % sim_N_mean)) + r'$\pm$' +\
            ('%s' % float('%.2g' % sim_mean_error))))
plt.plot(bin_centres,\
         qw_analytic_N_prob_dist,label='{0}'.format(dist_fit_label))

plt.xlabel(r'$\mathcal{N}$')
plt.ylabel('Probability Density')
plt.legend(loc = 'upper right')
histogram_name = 'qw_N_distribution_for_' +'tilt_' + ('%s' % float('%.2g' % tilt))+'_x_' +\
    ('%s' % float('%.5g' % x))+'_mu_'+str(mu)+'_dN_' + ('%s' % float('%.3g' % dN))+\
        '_iterations_'+str(num_sims)+'_'+step_type+'_'+bc_type+'_NOT_weighted.pdf'
plt.savefig('Cython_qw_results/'+histogram_name,transparent=True)
plt.show()
plt.clf()


N_lim = 1

if Using_IS == 'yes':
    max_count  = 100000
    if count_reflects == 'yes':
        max_count_logic = qw_sim_ref_counts < max_count
        if max_count>1:
            scatters = plt.scatter(qw_sim_N_dist[max_count_logic],\
                       np.log10(qw_sim_w_values[max_count_logic]),\
                       c=qw_sim_ref_counts[max_count_logic] )
            cbar = plt.colorbar(scatters)
            cbar.set_label(r'# reflections')
        else:
            scatters = plt.scatter(qw_sim_N_dist[max_count_logic],\
                       np.log10(qw_sim_w_values[max_count_logic]))
    
    else:
        plt.scatter(qw_sim_N_dist,np.log10(qw_sim_w_values))
    plt.xlabel(r'$\mathcal{N}$')
    plt.ylabel(r'$log_{10}(w)$')
    plt.title( str(num_sims) + r', dN=' +\
          ('%s' % float('%.2g' % dN)) +r', bias = ' +\
              ('%s' % float('%.2g' % bias))+r', $\mu$='+str(mu))
    scatter_name = 'log_of_qw_weights_of_importance_sampling_with_N' +'_dN_' + str(dN) +\
         '_iterations_' + str(num_sims)+ bc_type+\
          '_bias_' + str(bias)+ '.png'
    if max_count  == 1:
        scatter_name = 'no_reflection'+scatter_name 
    plt.savefig('Cython_qw_results/'+scatter_name ,transparent=True, dpi=400)
    plt.show()
    plt.clf()
    
    
    if scater_density_plot == True:
        plt.hist2d(qw_sim_N_dist, np.log10(qw_sim_w_values), (50, 50), norm=LogNorm())
        cbar = plt.colorbar()
        cbar.set_label(r'# Data Points')
        plt.title( str(num_sims) + r', dN=' +\
              ('%s' % float('%.2g' % dN)) +r', bias = ' +\
                  ('%s' % float('%.2g' % bias))+r', $\mu$='+str(mu))

        plt.xlabel(r'$\mathcal{N}$')
        plt.ylabel(r'$log_{10}(w)$')
        scatter_name = 'scatter_density_qw_weights_of_importance_sampling_with_N' +'_dN_' + str(dN) +\
             '_iterations_' + str(num_sims)+ bc_type+\
              '_bias_' + str(bias)+ '.png'
        plt.savefig('Cython_qw_results/'+scatter_name ,transparent=True, dpi=400)
        
        plt.show()
        plt.clf()
    '''
plt.scatter(qw_sim_N_dist[qw_sim_w_values>1],np.log10(qw_sim_w_values[qw_sim_w_values>1]),label='{0}'.format('double absorbing'))
plt.xlabel(r'$\mathcal{N}$')
plt.ylabel(r'$log_{10}(w)$')
plt.title( str(qw_sim_w_values[qw_sim_w_values>1].size) + r', dN=' +\
      ('%s' % float('%.2g' % dN)) +r', bias = ' +\
          ('%s' % float('%.2g' % bias))+r', $\mu$='+str(mu))
scatter_name = 'log_of_qw_weights_of_importance_sampling_with_N_for_left_absorbing_boundary' +'_dN_' + str(dN) +\
     '_iterations_' + str(num_sims)+ bc_type+\
      '_bias_' + str(bias)+ '.png'
plt.xlim(0, 0.6) 
plt.ylim(0.22, 0.8) 
plt.legend()
plt.savefig('Cython_qw_results/'+scatter_name ,transparent=True, dpi=400)
plt.show()
plt.clf()
    '''
    
    def analytical_pdf_func(x,n):
        if tilt == 0:
            def func(Ns):
                return quantum_well_sym.quantum_diffusion_N_probability_dist(Ns,\
                                                          x, n)
        else:
            def func(Ns):
                return cosfuncs.wide_tilted_well_N_probability_dist(Ns,\
                                                     x, tilt, delta_phi, v_0, n)
        return func
    
    bins2 = cosfuncs.histogram_same_num_data_bins(qw_sim_N_dist, 50)
    bin_centres2 =\
        np.array([(bins2[i]+bins2[i+1])/2 for i in range(len(bins2)-1)])
    qw_analyical_bin_height2 =\
        cosfuncs.histogram_analytical_height(bins2, analytical_pdf_func(x,n))
    height_error2 =\
        cosfuncs.histogram_weighted_bin_errors_jackknife(qw_sim_N_dist,\
                    qw_sim_w_values, bins2, 20)
        
        
    bin_height2,_,_ = plt.hist(np.array(qw_sim_N_dist), weights =\
                           np.array(qw_sim_w_values), bins = bins2,\
                               density = True)
    plt.clf()
    plt.errorbar(bin_centres2, bin_height2, yerr=height_error2, fmt='.', \
                 capsize=3,label='{0}'.format(r'Weighted bar height')) 
    plt.errorbar(bin_centres2, qw_analyical_bin_height2, fmt=".", capsize=3,\
                     label='{0}'.format(r'Anaytical bar Height'))
    #plt.xlim(0, 0.15) 
    plt.xlabel(r'$\mathcal{N}$')
    plt.ylabel('Probability Density')
    plt.yscale('log')
    plt.title( str(num_sims) + r', bias=' +str(bias)+r', $dN$=' +\
          ('%s' % float('%.2g' % dN))+r', $\mu$='+str(mu))
    #plt.xlim((0, N_lim))
    #plt.ylim((0, 0.05)) 
    plt.legend(loc = 'upper right')
    histogram_name =\
        'qw_scatter_plot_tilt_' + ('%s' % float('%.2g' % tilt))+'_bias_'\
        +('%s' % float('%.2g' % bias))+'_x_' + ('%s' % float('%.5g' % x))+\
            '_mu_'+str(mu)+'_dN_' + ('%s' % float('%.3g' % dN))+\
        '_iterations_'+str(num_sims)+'_'+step_type+'_'+bc_type+'_same_bin_size.pdf'
    plt.savefig('Cython_qw_results/'+histogram_name,transparent=True)
    plt.show()
    plt.close()
        
    #Log plot, to show behavior in the tail.
    
    bin_height,bins,_ =\
        plt.hist(qw_sim_N_dist, bins=50,\
                 weights = qw_sim_w_values,\
                     density=True, label='{0}'.format('Weighted bins'),\
                         histtype="step" )
    _,bins,_ =\
        plt.hist(qw_sim_N_dist, bins=50,\
                 density=True, label='{0}'.format('NOT weighted bins'),\
                     histtype="step" )
    plt.plot(bin_centres,\
         qw_analytic_N_prob_dist,\
             label='{0}'.format('Analytical'))
    #plt.xlim(0, 0.15) 
    plt.xlabel(r'$\mathcal{N}$')
    plt.ylabel('Probability Density')
    plt.yscale('log')
    plt.title( str(num_sims) + r', bias=' +str(bias)+r', $dN$=' +\
          ('%s' % float('%.2g' % dN))+r', $\mu$='+str(mu))
    #plt.xlim((0, N_lim))
    #plt.ylim((0, 0.05)) 
    plt.legend(loc = 'lower left')
    histogram_name =\
        'qw_stepped_histogram_tilt_' + ('%s' % float('%.2g' % tilt))+'_bias_'\
        +('%s' % float('%.2g' % bias))+'_x_' + ('%s' % float('%.5g' % x))+\
            '_mu_'+str(mu)+'_dN_' + ('%s' % float('%.3g' % dN))+\
        '_iterations_'+str(num_sims)+'_'+step_type+'_'+bc_type+'_weighted.pdf'
    plt.savefig('Cython_qw_results/'+histogram_name,transparent=True)
    plt.show()
    plt.close()
    
    bin_height3,_,_ = plt.hist(qw_sim_N_dist[qw_sim_N_dist<0.4], weights =\
                           qw_sim_w_values[qw_sim_N_dist<0.4], bins = bins2,\
                               density = True)
    plt.clf()
    
    plt.errorbar(bin_centres, heights, yerr = errors, fmt='. k', \
                 capsize=4,label='{0}'.format(r'Simulation'),markersize=8)
    plt.plot(bin_centres,\
         qw_analytic_N_prob_dist,\
             label='{0}'.format('Analytical'))
    #plt.xlim(0, 0.15) 
    plt.xlabel(r'$\mathcal{N}$')
    plt.ylabel('Probability Density')
    plt.yscale('log')
    plt.title( str(num_sims) + r', bias=' +str(bias)+r', $dN$=' +\
          ('%s' % float('%.2g' % dN)))
    plt.legend()
    histogram_name =\
        'error-bar_plot' + ('%s' % float('%.2g' % tilt))+'_bias_'\
        +('%s' % float('%.2g' % bias))+'_x_' +\
        ('%s' % float('%.5g' % x))+'_mu_'+str(mu)+'_dN_' + '_iterations_'\
            +str(num_sims)+'_'+step_type+'_'+bc_type+'.pdf'
    plt.savefig('Cython_qw_results/'+histogram_name,transparent=True)
    plt.show()
    plt.close()


    '''Poster
    bin_height3,bins3,_ =\
        plt.hist(qw_sim_N_dist-qw_analytic_N_mean, bins=20,\
                 weights = qw_sim_w_values,\
                     density=True, label='{0}'.format('Weighted bins'),\
                         histtype="step" )
    plt.close()
    if bias>0:
        bins_stored = bins3
    bin_centres3 =\
        np.array([(bins3[i]+bins3[i+1])/2 for i in range(len(bins3)-1)])
    qw_analyical_bin_height3 =\
        cosfuncs.histogram_analytical_height(bins3, analytical_pdf_func(x,n))
    height_error3 =\
        cosfuncs.histogram_weighted_bin_errors_jackknife(np.array(qw_sim_N_dist)-qw_analytic_N_mean,\
                    np.array(qw_sim_w_values), bins3, 20, density = 'True')
            
    plt.figure(figsize = [8., 5.])    
    plt.errorbar(bin_centres3, bin_height3, yerr=height_error3, fmt='.', \
                 capsize=5,label='{0}'.format(r'Simulation'),markersize=10) 
    plt.plot(qw_sim_N_dist-qw_analytic_N_mean,\
         qw_analytic_N_prob_dist,\
             label='{0}'.format('Analytical'))
    plt.axvline(1, color='k', linestyle='dashed', linewidth=3)
    plt.text(1-0.1,0.01,r'$\zeta_{\rm crit}$',rotation=90, fontsize=24)
    plt.xlim(0, 1.4) 
    plt.ylim(10**-12, 10) 
    plt.xlabel(r'Curvature Perturbation, $\zeta$', fontsize=24)
    plt.ylabel(r'$P(\zeta)$', fontsize=24)
    plt.yscale('log')
    plt.title( 'Direct Sample')
    plt.legend(loc = 'lower left', fontsize=24)
    histogram_name =\
        'for_poster_histogram' + ('%s' % float('%.2g' % tilt))+'_bias_'\
        +('%s' % float('%.2g' % bias))+'_x_' + ('%s' % float('%.5g' % x))+\
            '_mu_'+str(mu)+'_dN_' + ('%s' % float('%.3g' % dN))+\
        '_iterations_'+str(num_sims)+'_'+step_type+'_'+bc_type+'_same_bin_size.pdf'
    plt.savefig('Cython_qw_results/'+histogram_name,transparent=True)
    plt.show()
    plt.close()
    '''
    
    
    
    
    
elif Using_IS != 'yes':
    bin_height,bins,_ =\
        plt.hist(qw_sim_N_dist, bins=50,\
                     density=True, label='{0}'.format('No bias'),\
                         histtype="step" )
    plt.plot(bin_centres,\
         qw_analytic_N_prob_dist,\
             label='{0}'.format('Analytical'))
    #plt.xlim(0, 0.15) 
    plt.xlabel(r'$\mathcal{N}$')
    plt.ylabel('Probability Density')
    plt.yscale('log')
    plt.title( str(num_sims) + r', bias=' +str(bias)+r', $dN$=' +\
          ('%s' % float('%.2g' % dN)))
    plt.legend()
    histogram_name =\
        'qw_stepped_histogram_tilt_' + ('%s' % float('%.2g' % tilt))+'_bias_'\
        +('%s' % float('%.2g' % bias))+'_x_' +\
        ('%s' % float('%.5g' % x))+'_mu_'+str(mu)+'_dN_' + '_iterations_'\
            +str(num_sims)+'_'+step_type+'_'+bc_type+'.pdf'
    plt.savefig('Cython_qw_results/'+histogram_name,transparent=True)
    plt.show()
    plt.close()
    
    
    


'''
Bayesian Analysis - going to assume the tail is given by an exponential with 
with N, with some amplitude. Using the results of arXiv:1707.00537v3 to give 
the priors for the Bayesian analysis. The Bayesian analysis is based on the 
results of https://github.com/minaskar/BayesWorkshop 

Cheers Minas!

The model will be P = A*e^(-B*N) when far into the tail

Will use the heights of the histograms as the y data points and the centre of
the corresponding bins as the time value
'''
if bayesian == 'yes':
    #Estimating the limits on the exponential, to give the prior
    A_true = (PI/mu**2)*np.sin(0.5*x*PI)
    B_true = (PI/(2*mu))**2
    A_max = 6.5*2*PI*mu**(-2)#taking the max to e n = 5
    B_max = 30.25*(PI/mu)**2#taking the max to e n = 5
    #Estimating the limits for the Gaussian fit model
    mean_max = 4*qw_analytic_N_mean
    std_max = 4*qw_analytic_N_var**0.5
    
    
    #Using the histograms to define the 'data points' 
    bin_centres = np.array([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)])
    binned_N, binned_w =\
        cosfuncs.histogram_analytical_height(qw_sim_N_dist, qw_sim_w_values, bins)
    #area of none density weighted histogram
    area_norm = np.sum(np.sum(binned_w, axis=0)*np.diff(bins))
    #The error is given by 1 standard deviation, and scaled by the histogram area
    '''
    height_error =\
        cosfuncs.histogram_weighted_bin_errors(binned_w, num_sims,normalisation\
                                               = area_norm ,\
                                                   num_std = 1)
    '''
    height_error = cosfuncs.histogram_weighted_bin_errors_jackknife(np.array(qw_sim_N_dist), np.array(qw_sim_w_values), bins,\
                                                20, density = 'True')
    #As we are looking at the fitting an exponential curve, we need to look at the
    # `tail' of the distrbution
    N_tail = qw_analytic_N_mean + 5*qw_analytic_N_var**0.5#4 std from mean
    #Now truncating the values not in the tail
    bin_centres_tail = bin_centres[bin_centres>N_tail]
    height_tail = heights[bin_centres>N_tail]
    height_error_tail = height_error[bin_centres>N_tail]
    
    #Remove empty bins
    height_error_tail = height_error_tail[height_tail>0]
    bin_centres_tail = bin_centres_tail[height_tail>0]
    #Truncate last so the others have the correct condition applied
    height_tail = height_tail[height_tail>0]
    
    #Defining the functions for model and the Bayesian analysis
    def expo_model(params, N):
        A, B = params
        return A*np.exp(-B*N)
    
    def gaussian_model(params, N):
        mean, std = params
        return np.exp(-0.5*((N-mean)/std)**2)/(std*np.sqrt(2*PI))
    
    #Assuming independent priors
    def expo_log_prior(params):
        A, B = params
        # A -> [0.0, A_max]
        # B -> [0.0, B_max]
    
        if A<=0.0 or A>A_max or B<=0.0 or B>B_max:
            return -np.inf
        return 0.0
    
    #Assuming independent priors
    def gauss_log_prior(params):
        mean, std = params
        # A -> [0.0, A_max]
        # B -> [0.0, B_max]
    
        if mean<=0.0 or mean>mean_max or std<=0.0 or std>std_max:
            return -np.inf
        return 0.0
    
    def expo_log_like(params, N, P, Perr):
        # compute for each t point, where it should lie in y
        y_model = expo_model(params, N)
        # compute likelihood
        loglike = -0.5 *np.sum(((y_model - P) / Perr)**2)
    
        return loglike
    
    def expo_log_like_func_maker(N, P, Perr):
        def like_func(params):
            # compute for each t point, where it should lie in y
            y_model = expo_model(params, N)
            # compute likelihood
            loglike = -0.5 *np.sum(((y_model - P) / Perr)**2)
            return loglike
        return like_func
    
    def gauss_log_like(params, N, P, Perr):
        # compute for each t point, where it should lie in y
        y_model = gaussian_model(params, N)
        # compute likelihood
        loglike = -0.5 *np.sum(((y_model - P) / Perr)**2)
    
        return loglike
    
    def gauss_log_like_func_maker(N, P, Perr):
        def like_func(params):
            # compute for each t point, where it should lie in y
            y_model = gaussian_model(params, N)
            # compute likelihood
            loglike = -0.5 *np.sum(((y_model - P) / Perr)**2)
            return loglike
        return like_func
    
    def log_post(params, N, P, Perr):
        lp = expo_log_prior(params)
        if ~np.isfinite(lp):#i.e., not finite so infinite
            return -np.inf 
        return lp + expo_log_like(params, N, P, Perr)
    
    def log_post_gauss(params, N, P, Perr):
        lp = gauss_log_prior(params)
        if ~np.isfinite(lp):#i.e., not finite so infinite
            return -np.inf 
        return lp + gauss_log_like(params, N, P, Perr)
    
    def expo_log_post_func_maker(N, P, Perr):
        def func(params):
            lp = expo_log_prior(params)
            if ~np.isfinite(lp):#i.e., not finite so infinite
                return -np.inf 
            return lp + expo_log_like(params, N, P, Perr)
        return func
    
    
    # Initial guess
    p_guess = [300.0, 100.0]#arbitary guess currently
    p_guess_gaussian = [qw_analytic_N_mean, qw_analytic_N_var**0.5]
    
    #Let's check how sure we are the tail is indeed an exponential
    
    #So that log postier only takes parameters as arguments
    expo_like =\
        expo_log_like_func_maker(bin_centres_tail, height_tail, height_error_tail)
    gauss_like =\
        gauss_log_like_func_maker(bin_centres_tail, height_tail, height_error_tail)
    log_post_params =\
        expo_log_post_func_maker(bin_centres_tail, height_tail, height_error_tail)
          
        
    # Run the minimisation of the negative of the log of procedure using the Nelder-Mead method
    #This gives the max
    bayesian_results =\
        minimize(lambda x :\
                 -log_post(x, bin_centres_tail, height_tail, height_error_tail),\
                  p_guess, method='Nelder-Mead',\
                      options={'maxiter':2000, 'disp':True})
    
    bayesian_results_gauss =\
        minimize(lambda x :\
                 -log_post_gauss(x, bin_centres_tail, height_tail, height_error_tail),\
                  p_guess_gaussian, method='Nelder-Mead',\
                      options={'maxiter':2000, 'disp':True})
    
            
    print('MAP =', bayesian_results.x)
    
    #The expected exponential tail, so n = 1 in the sum  
    analytic_expo_tail =\
        cosfuncs.quantum_diffusion_N_probability_dist_alt(bin_centres_tail, x,\
                                                          delta_phi, v_0, 1)
    baysian_expo_tail = expo_model(bayesian_results.x, bin_centres_tail)
    #Now let's estimate the uncertainities on t
    ndim = len(bayesian_results.x)
    nwalkers = 2*ndim+2#Need at least as many walkers as dimensions and even
    nsteps = 1000
    labels = ['A', 'B']
    #Perturbbded away from the optimal solution
    start = bayesian_results.x*(1 + 0.001 * np.random.randn(nwalkers, ndim))
    
    # initialise the sampler
    sampler = zeus.EnsembleSampler(nwalkers, ndim, log_post_params)
    #Work on this simpler object
    sampler.run_mcmc(start, nsteps)
    
    chain = sampler.get_chain(flat=True, discard=0.5, thin=7)
    #zeus.cornerplot(chain, labels=labels, truth=[A_true, B_true])#corner plot
    
    #Finding the 1sigma uncertainty on the parameters
    Aerr = np.std(chain[:,0])
    Berr = np.std(chain[:,1])
    
    '''
    '''
    
    ntemps = 40
    nwalkers = 2
    ndim_expo = 2
    ndim_gauss = 2
    nsteps = 1000
    
    if False:#So I can easily trun it off
        #Now running MCMC for the exponential model
        start_expo =\
            bayesian_results.x*(1 + 0.001* np.random.randn(ntemps, nwalkers,\
                                                          ndim_expo))
        sampler_expo =\
            arn.ReplicaExchangeSampler(ntemps, nwalkers, ndim_expo, expo_like,\
                                       expo_log_prior)
        sampler_expo.run_mcmc(start_expo, nsteps)
        logz_expo = sampler_expo.get_logz()
        #Now running MCMC for the gaussian model
        start_gauss =\
            bayesian_results_gauss.x*(1 + 0.001* np.random.randn(ntemps, nwalkers,\
                                                          ndim_expo))
        sampler_gauss =\
            arn.ReplicaExchangeSampler(ntemps, nwalkers, ndim_gauss, gauss_like,\
                                       gauss_log_prior)
        sampler_gauss.run_mcmc(start_gauss, nsteps)
        logz_gauss = sampler_gauss.get_logz()
        
        baysian_model_probability = np.exp(logz_expo-logz_gauss)
        print("Expo model is %.2f more likely than gaussian" \
              % baysian_model_probability)
        
        
    '''
    '''
    
    
    #The expected exponential tail, so n = 1 in the sum  
    analytic_expo_tail =\
        cosfuncs.quantum_diffusion_N_probability_dist_alt(bin_centres_tail, x,\
                                                          delta_phi, v_0, 1)
    baysian_expo_tail = expo_model(bayesian_results.x, bin_centres_tail)
        
    plt.plot(bin_centres_tail, analytic_expo_tail,label='{0}'.format('Expected tail'))
    plt.errorbar(bin_centres_tail, height_tail, yerr = 2*height_error_tail,\
                 fmt=".k", capsize=3, label='{0}'.format(r'2$\sigma$ errors'))
    #plt.plot(bin_centres_tail, baysian_expo_tail, label='{0}'.format('Bayesian tail'))#The best Baysian fit
    #plt.plot(bin_centres_tail, gaussian_model(bayesian_results_gauss.x, bin_centres_tail), label='{0}'.format('Gaussian tail'))
    #Showing the range of plots
#    plt.fill_between(bin_centres_tail,\
#                     expo_model(bayesian_results.x+[2*Aerr,2*Berr], bin_centres_tail),\
#                     expo_model(bayesian_results.x-[2*Aerr,2*Berr], bin_centres_tail),\
 #                    alpha=0.5)
    #plt.scatter(bin_centres_tail, height_tail, label='{0}'.format('simulation binned'))
    plt.yscale('log')
    plt.xlabel(r'$\mathcal{N}$')
    plt.ylabel(r'Probability Density')
    plt.title(r'Tail for $\mu=$'+str(mu)+' and #sims='+str(num_sims))
    plt.legend()
    plt.savefig('Cython_qw_results/'+'bayesian_analysis_check'+'_bias_'\
            +('%s' % float('%.2g' % bias))+'_x_' + ('%s' % float('%.5g' % x))+\
                '_mu_'+str(mu)+'_dN_' + ('%s' % float('%.3g' % dN))+\
            '_iterations_'+str(num_sims)+'_'+step_type+'_'+bc_type+'_with_certainties.pdf',\
                transparent=True)
    plt.show()
    plt.close()
    
    
    '''
    labels = ['A', 'B']
    plt.figure(figsize=(14,8))
    for i in range(ndim):
        plt.subplot(ndim,1,i+1)
        plt.plot(samples[:,:,i],alpha=0.6)
        plt.ylabel(labels[i], fontsize=18)
    plt.xlabel('Iteration', fontsize=19)
    plt.tight_layout()
    plt.savefig('Cython_qw_results/'+'bayesian_analysis_walker_plot'+'_bias_'\
            +('%s' % float('%.2g' % bias))+'_x_' + ('%s' % float('%.5g' % x))+\
                '_mu_'+str(mu)+'_dN_' + ('%s' % float('%.3g' % dN))+\
            '_iterations_'+str(num_sims)+'_'+step_type+'_'+bc_type+'.pdf',\
                transparent=True)
     '''