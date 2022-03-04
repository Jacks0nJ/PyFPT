# !/usr/bin/env python3
#  -*- coding: utf-8 -*-
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


# M_PL = 2.435363*10**18 old value
M_PL = 1.0  # Using units of M_PL
PI = np.pi


def IS_simulation(phi_i, phi_end, V, V_dif, V_ddif, num_sims, bias, bins=50,
                  dN=None, min_bin_size=400, num_sub_samples=20,
                  reconstruction='lognormal', save_data=False, N_f=100,
                  phi_UV=None):
    # If no argument for dN is given, using the classical std to define it
    if dN is None:
        if isinstance(bins, int) is True:
            std =\
                cosfuncs.delta_N_squared_sto_limit(V, V_dif, V_ddif, phi_i,
                                                   phi_end)
            dN = std/(3*bins)
        elif isinstance(bins, int) is False:
            std =\
                cosfuncs.delta_N_squared_sto_limit(V, V_dif, V_ddif, phi_i,
                                                   phi_end)
            num_bins = len(bins)-1
            dN = std/(3*num_bins)
    elif isinstance(dN, float) is not True and isinstance(dN, int) is not True:
        raise ValueError('dN is not a number')

    if reconstruction != 'lognormal' and reconstruction != 'naive':
        print('Invalid reconstruction argument, defaulting to naive method')
        reconstruction = 'naive'

    if bias < 0.2:
        reconstruction = 'naive'

    if phi_UV is None:
        phi_UV = 10000*phi_i
    elif isinstance(phi_UV, float) is False:
        if isinstance(phi_UV, int) is True:
            if isinstance(phi_UV, bool) is True:
                raise ValueError('phi_UV is not a number')
            else:
                pass
        else:
            raise ValueError('phi_UV is not a number')

    # The number of sims per core, so the total is correct
    num_sims_per_core = int(num_sims/mp.cpu_count())

    start = timer()

    def multi_processing_func(phi_i, phi_UV, phi_end, N_i, N_f, dN, bias,
                              num_sims, queue_Ns, queue_ws, queue_refs):
        results =\
            is_code.many_simulations_importance_sampling(phi_i, phi_UV,
                                                         phi_end, N_i, N_f, dN,
                                                         bias, num_sims, V,
                                                         V_dif, V_ddif,
                                                         bias_type='diffusion',
                                                         count_refs=False)
        Ns = np.array(results[0][:])
        ws = np.array(results[1][:])
        queue_Ns.put(Ns)
        queue_ws.put(ws)

    queue_Ns = Queue()
    queue_ws = Queue()
    queue_refs = Queue()
    cores = int(mp.cpu_count()/1)

    print('Number of cores used: '+str(cores))
    processes = [Process(target=multi_processing_func,
                         args=(phi_i, phi_UV,  phi_end, 0.0, N_f, dN, bias,
                               num_sims_per_core, queue_Ns, queue_ws,
                               queue_refs)) for i in range(cores)]

    for p in processes:
        p.start()

    Ns_array = np.array([queue_Ns.get() for p in processes])
    ws_array = np.array([queue_ws.get() for p in processes])
    end = timer()
    print(f'The simulations took: {end - start}')

    # Combine into columns into 1
    sim_N_dist = Ns_array.flatten()
    w_values = ws_array.flatten()

    # Sort in order of increasing Ns
    sort_idx = np.argsort(sim_N_dist)
    sim_N_dist = sim_N_dist[sort_idx]
    w_values = w_values[sort_idx]

    # Checking if multipprocessing error occured, by looking at correlation
    multi_processing_error(sim_N_dist, w_values)

    # Truncating the data
    sim_N_dist, w_values =\
        cosfuncs.histogram_data_truncation(sim_N_dist, N_f, weights=w_values,
                                           num_sub_samples=num_sub_samples)
    # Saving the data
    if save_data is True:
        save_data_to_file(sim_N_dist, w_values, phi_i, num_sims, bias=bias)

    # Now analysisng creating the PDF data
    bin_centres, heights, errors, num_sims_used, bin_edges_untruncated =\
        data_points_pdf(sim_N_dist, w_values, num_sub_samples,
                        reconstruction, bins=bins,
                        min_bin_size=min_bin_size, num_sims=num_sims)

    return bin_centres, heights, errors


def data_points_pdf(Ns, ws, num_sub_samples, reconstruction,
                    min_bin_size=None, bins=50, num_sims=None):
    # If no number of simulations argument is passed.
    if isinstance(num_sims, int) is not True:
        num_sims = len(Ns)

    # If the number of bins used has been specified
    if isinstance(bins, int) is True:
        num_bins = bins
        # Want raw heights of histogram bars
        heights_raw, bins, _ =\
            plt.hist(Ns, num_bins, weights=ws)
        plt.clf()
    # If the bins have been specified
    else:
        num_bins = len(bins)-1  # as bins is the bin edges, so plus 1
        # Want raw heights of histogram bars
        heights_raw, bins, _ =\
            plt.hist(Ns, bins=bins, weights=ws)
        plt.clf()

    analytical_norm =\
        cosfuncs.histogram_analytical_normalisation(bins, num_sims)

    data_in_bins, weights_in_bins =\
        cosfuncs.histogram_data_in_bins(Ns, ws, bins)

    # Predictions need the bin centre to make comparison
    bin_centres = np.array([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)])

    # Removing underfilled bins if needed
    if isinstance(min_bin_size, int) is True:
        # Then loop through to find where length is greater than min_bin_size
        filled_bins = []
        num_sims_used = len(Ns)
        for i in range(len(bins)-1):
            data_in_bin = data_in_bins[:, i]
            data_in_bin = data_in_bin[data_in_bin > 0]
            # First, remove if empty
            if len(data_in_bin) == 0:
                filled_bins.append(False)
            # If there is enough data in this bin
            elif len(data_in_bin) >= min_bin_size:
                filled_bins.append(True)
            # Don't include under filled tail bins
            else:
                filled_bins.append(False)
                # Reflect in number of succesful simulatios
                num_sims_used -= len(data_in_bin)
        bin_centres_uncut = bin_centres
        bin_centres = bin_centres[filled_bins]

    if reconstruction == 'naive':
        heights = heights_raw/analytical_norm
        errors =\
            cosfuncs.histogram_weighted_bin_errors_jackknife(Ns, ws, bins,
                                                             num_sub_samples)
        if isinstance(min_bin_size, int) is True:
            heights = heights[filled_bins]
            errors = errors[filled_bins]
    elif reconstruction == 'lognormal':

        heights_est = np.zeros(num_bins)
        # The errors for the log-normal case are asymmetric
        errors_est = np.zeros((2, num_bins))
        for i in range(num_bins):
            w = weights_in_bins[:, i]
            # Only calculate filled bins
            if filled_bins[i] is True or\
                (np.any([w > 0]) is True and isinstance(min_bin_size, int)
                 is False):
                w = w[w > 0]
                heights_est[i] =\
                    cosfuncs.log_normal_height(w,
                                               position=bin_centres_uncut[i])
                errors_est[:, i] = cosfuncs.log_normal_errors(w)

        # Include only filled values
        # Remember to normalise errors as well
        heights = heights_est[errors_est[0, :] > 0]/analytical_norm
        # The errors are a 2D array, so need to slice correctly
        errors = errors_est[:, errors_est[0, :] > 0]/analytical_norm

        # Checking p-values if lognormal was used
        lognormality_check(bin_centres, weights_in_bins, filled_bins,
                           num_bins)

    else:
        raise ValueError('Not valid reconstrcution method')

    return bin_centres, heights, errors, num_sims_used, bins


def save_data_to_file(sim_N_dist, w_values, phi_i, num_sims, bias=0):
    data_dict_raw = {}
    data_dict_raw['N'] = sim_N_dist
    if bias > 0:
        data_dict_raw['w'] = w_values

    data_pandas_raw = pd.DataFrame(data_dict_raw)

    raw_file_name = 'IS_data_phi_i_' + ('%s' % float('%.3g' % phi_i)) +\
        '_iterations_' + str(num_sims) + '_bias_' +\
        ('%s' % float('%.3g' % bias)) + '.csv'
    # Saving to a directory for the language used

    data_pandas_raw.to_csv(raw_file_name)


# This function tests if a multiprocessing error has occured. This is when the
# data from the different cores becomes mixed, and the weights and N are not
# correct
def multi_processing_error(sim_N_dist, w_values):
    # Checking if multipprocessing error occured, by looking at correlation
    pearson_corr = np.corrcoef(sim_N_dist, np.log10(w_values))
    pearson_corr = pearson_corr[0, 1]

    if pearson_corr > -0.55:  # Data is uncorrelated
        print('Possible multiprocessing error occured, terminating')


def lognormality_check(bin_centres, weights_in_bins, filled_bins, num_bins):
    # Checking p-values if lognormal was used
    p_values = np.zeros(num_bins)
    p_values_theory = np.zeros(num_bins)
    for i in range(len(p_values)):
        if filled_bins[i] is True:
            w = weights_in_bins[:, i]
            log_w = np.log(w[w > 0])
            _, p_values[i] = sci_stat.normaltest(log_w)
            _, p_values_theory[i] =\
                sci_stat.normaltest(np.random.normal(0, 1, len(log_w)))

    p_values = p_values[filled_bins]
    p_values_theory = p_values_theory[filled_bins]
    if any(p_values) < 0.005:
        print('Possibly not log normal distribution, see p-value plot')
        print(p_values)
        plt.errorbar(bin_centres, p_values, fmt='.', ms=7)
        plt.hlines(0.005, np.min(bin_centres), np.max(bin_centres),
                   color='k', linestyle='dashed',
                   label='{0}'.format('0.5% threshold'), linewidth=2)

        plt.yscale('log')
        # plt.title('Data: p-values with bin centres')
        plt.legend(fontsize=20)
        plt.xlabel(r'$\mathcal{N}$', fontsize=20)
        plt.ylabel('p-values', fontsize=20)

        plt.show()
        plt.clf()

        plt.errorbar(bin_centres, p_values_theory, fmt='.')
        plt.hlines(0.005, np.min(bin_centres), np.max(bin_centres),
                   color='k', linestyle='dashed',
                   label='{0}'.format('0.5% threshold'))
        plt.yscale('log')
        plt.title('Theoretical: p-values with bin centres')
        plt.legend(fontsize=22)
        plt.xlabel(r'$\mathcal{N}$', fontsize=22)
        plt.ylabel('p-values', fontsize=22)
        plt.show()
        plt.clf()
