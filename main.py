# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:41:37 2021

@author: user
"""

import numpy as np
import pandas as pd

import scipy.stats as sci_stat
from timeit import default_timer as timer


import inflation_functions_e_foldings as cosfuncs
import is_data_analysis as isfuncs
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mpl_style
plt.style.use(mpl_style.style1)

# M_PL = 2.435363*10**18 old value
M_PL = 1.0  # Using units of M_PL
PI = np.pi
# m = 10**(-6)*M_PL#Based on John McDonald's calculations in PHYS461
m = 0.1*M_PL  # 4*PI*6**0.5

# Intial conditions and tolerances
N_starting = 10  # In some sense, this should techically be negative
phi_end = M_PL*2**0.5
phi_i = M_PL*(4*N_starting+2)**0.5  # M_PL*(4*N_starting+2)**0.5
phi_r = 100*phi_i
N_cut_off = 300
N_f = 100
dN = 0.02*m  # Assuming std(N) is proportional to m, was dN=0.02m
num_sims = 50000
num_bins = 50
num_sub_samples = 20


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

min_bin_size = 400
fit_threshold = 100

edgeworth_series = True
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


if (m == 2 or m == 1) and phi_i == phi_r:
    kazuya_pdf = True
    vincent = False
elif m > 0.6:
    vincent = True
    kazuya_pdf = False
    if log_normal is True:
        print('Are you sure you want to use the lognormal approach??')
else:
    kazuya_pdf = False
    vincent = False

if m < 0.2 and log_normal is False:
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
    cosfuncs.variance_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)
analytic_N_st = np.sqrt(analytic_N_var)
analytic_N_mean = cosfuncs.mean_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)

N_star = analytic_N_mean + 4*analytic_N_st

analytic_gauss_deviation_pos =\
    cosfuncs.gaussian_deviation_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)

if edgeworth_series is True:
    edgeworth_pdf = cosfuncs.edgeworth_pdf_sto_limit(V, V_dif, V_ddif, phi_i,
                                                     phi_end)


'''
#Running the simulation many times
'''


if log_normal is True:
    reconstruction = 'lognormal'
else:
    reconstruction = 'naive'

bin_centres, heights, errors =\
    isfuncs.IS_simulation(phi_i, phi_end, V, V_dif, V_ddif, num_sims, bias,
                          bins=50, dN=dN, reconstruction=reconstruction,
                          save_data=True, phi_UV=100, min_bin_size=100)


'''
Reading the saved data
'''
raw_file_name = 'IS_data_phi_i_' + ('%s' % float('%.3g' % phi_i)) +\
            '_iterations_'+str(num_sims) + '_bias_' +\
            ('%s' % float('%.3g' % bias))+'.csv'
raw_data = pd.read_csv(raw_file_name, index_col=0)
sim_N_dist = np.array(raw_data['N'])
if bias > 0:
    w_values = np.array(raw_data['w'])


bin_centres_analytical = np.linspace(bin_centres[0], bin_centres[-1],
                                     2*num_bins)

if m > 0.6:  # Less than this it breaks down:
    start = timer()
    PDF_analytical_test = cosfuncs.large_mass_pdf(bin_centres_analytical,
                                                  phi_i, phi_end, V)
    end = timer()
    print(f'The analytical answer took: {end - start}')
    best_fit_line2 = PDF_analytical_test
    dist_fit_label2 = r'Pattison 2017'
    plt.plot(bin_centres_analytical, best_fit_line2,
             label='{0}'.format(dist_fit_label2))
else:
    best_fit_line2 = sci_stat.norm.pdf(bin_centres_analytical,
                                       analytic_N_mean, analytic_N_st)
    dist_fit_label2 = r'Gaussian $\sqrt{\delta \mathcal{N}^2}$=' +\
        str(round(analytic_N_st, 4))

'''
Saving data
'''
if save_results is True:
    data_dict = {}
    data_dict['bin_centres'] = bin_centres
    data_dict['PDF'] = heights
    if log_normal is False:
        data_dict['errors'] = errors
    elif log_normal is True:
        data_dict['errors_lower'] = errors[0, :]
        data_dict['errors_upper'] = errors[1, :]

    data_pandas_results = pd.DataFrame(data_dict)

    my_file_name = 'results_for_N_' + str(N_starting) + '_dN_'+str(dN) + '_m_'\
        + ('%s' % float('%.3g' % m)) + '_iterations_' + str(num_sims) +\
        '_bias_'+str(bias)+'_phi_UV_'+str(phi_r/phi_i)+'phi_i'+'.csv'
    # Saving to a directory for the language used

    data_pandas_results.to_csv(my_file_name)
    # Remembering to remove column numbering
    sim_data = pd.read_csv(my_file_name, index_col=0)

    # Now read this data back
    bin_centres = np.array(sim_data['bin_centres'])
    heights = np.array(sim_data['PDF'])
    if log_normal is False:
        errors = np.array(sim_data['errors'])
    elif log_normal is True:
        errors = np.zeros((2, len(heights)))
        errors[0, :] = np.array(sim_data['errors_lower'])
        errors[1, :] = np.array(sim_data['errors_upper'])

'''
 Plotting
'''

if bias != 0:
    _, _, _ = plt.hist(sim_N_dist, num_bins, weights=w_values, density=True,
                       label='{0}'.format('Weighted bins'))
    histogram_name = 'N_distribution_for_' + 'IS_near_' +\
        str(N_starting)+'_dN_' + str(dN) + '_m_' + ('%s' % float('%.3g' % m))\
        + '_Is_shift_' + str(bias) + '_iterations_' + str(num_sims) +\
        '_wighted.pdf'
    if m <= 0.6:
        dist_fit_label2 = r'Analytical $\sqrt{\delta \mathcal{N}^2}$=' +\
            str(round(analytic_N_st, 4))
    else:
        dist_fit_label2 = r'Pattison 2017'
    plt.plot(bin_centres_analytical, best_fit_line2,
             label='{0}'.format(dist_fit_label2))
    plt.title(r'bias=' + str(bias) + ', ' + str(num_sims) + ' sims, ' +
              r'dN=' + ('%s' % float('%.3g' % dN)) +
              r', $m$=' + ('%s' % float('%.3g' % m)))
    plt.xlabel(r'$\mathcal{N}$')
    plt.ylabel(r'$\mathcal{N}$')
    plt.legend()
    # Including if I have used importance sampling
    # Saving to a directory for the language used
    if save_plots is True:
        plt.savefig(histogram_name, transparent=True)
    plt.show()
    plt.clf()


# Plotting the weights, and number of reflections if appropriate
scatter_name = '_dN_' + str(dN) + '_m_' +\
        ('%s' % float('%.3g' % m)) + '_phi_UV_' +\
        str(phi_r/phi_i) + '_m_' + ('%s' % float('%.3g' % m)) +\
        '_iterations_' + str(num_sims)+'_bias_' + str(bias)
if contour is True:
    h, xedges, yedges, _ =\
        plt.hist2d(sim_N_dist, np.log10(w_values), (50, 50))
    plt.clf()
    xedges_centre =\
        np.array([(xedges[i]+xedges[i+1])/2 for i in range(len(xedges)-1)])
    yedges_centre =\
        np.array([(yedges[i]+yedges[i+1])/2 for i in range(len(yedges)-1)])
    X, Y = np.meshgrid(xedges_centre, yedges_centre)
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, h, (20, 100, 1000), colors='k')
    ax.clabel(CS, fontsize=9, inline=True)
    scatter_name = 'weights_contour' + scatter_name + '.png'
elif scater_density_plot is True:
    plt.hist2d(sim_N_dist, np.log10(w_values), (50, 50), norm=LogNorm())
    cbar = plt.colorbar()
    cbar.set_label(r'# Data Points')
    scatter_title = r'bias = ' + ('%s' % float('%.3g' % bias)) +\
        r', $dN=$' + ('%s' % float('%.3g' % dN)) + r', $m$=' +\
        ('%s' % float('%.3g' % m))

    scatter_name = 'weights_2D_histogram' + scatter_name + '.pdf'
else:
    plt.scatter(sim_N_dist, np.log10(w_values))
    scatter_title = r'bias = ' + ('%s' % float('%.3g' % bias)) +\
        r', $dN=$' + ('%s' % float('%.3g' % dN)) + r', $m$=' +\
        ('%s' % float('%.3g' % m))
    scatter_name = 'log_of_weights_of_IS' + scatter_name + '.png'

if save_plots is True:
    plt.xlabel(r'$\mathcal{N}$', fontsize=fontsize)
    plt.ylabel(r'${\rm log}_{10}(w)$', fontsize=fontsize)
    plt.margins(tight=True)
    plt.savefig(scatter_name, transparent=True, dpi=800)
plt.show()
plt.clf()

# Plotting the log of the distribution
histogram_name = 'publishable_error_bar_IS_near_' + str(N_starting) +\
    '_dN_' + ('%s' % float('%.2g' % dN)) + '_m_' +\
    ('%s' % float('%.3g' % m)) + '_phi_UV_'+str(phi_r/phi_i) + 'phi_i' +\
    '_bias_'+str(bias) + '_iters_' + str(num_sims) + '_bin_size_' +\
    str(min_bin_size)
if bias == 0:
    plt.errorbar(bin_centres, heights, yerr=errors, fmt=".", ms=7,
                 capsize=3, color=CB_color_cycle[7],
                 label='{0}'.format(r'Direct ($\mathcal{A}=0$)'))
else:
    plt.errorbar(bin_centres, heights, yerr=errors, fmt=".", ms=7,
                 capsize=3, color=CB_color_cycle[0],
                 label='{0}'.format(r'$\mathcal{A}=$'+str(bias)))

# r'Na'+u'\u00EF'+'ve method'
if vincent is True and bin_centres[-1] > 15 and phi_r > phi_i:
    bins_in_tail = bin_centres[bin_centres > 15]
    vincent_near_tail =\
        np.array([cosfuncs.vincent_near_tail_fit(bin_tail, m, phi_i,
                  numerical_integration=False) for bin_tail in
                  bins_in_tail])

    plt.plot(bins_in_tail, vincent_near_tail, color=CB_color_cycle[3],
             label='{0}'.format('Near tail approx.'), linewidth=2.5)

if m >= 1:
    plt.plot(bin_centres_analytical, best_fit_line2, label='{0}'.format(
        r'Exact $\phi_{\rm UV} \rightarrow \infty$'), linewidth=2,
        color=CB_color_cycle[1], linestyle='dashed')

# plt.xlim(right = N_cut_off)
elif m < 1 and m > 0.6:
    plt.plot(bin_centres, best_fit_line2, label='{0}'.format(
        r'Exact $\phi_{\rm UV} \rightarrow \infty$'), linewidth=2)
    if edgeworth_series is True:
        plt.plot(bin_centres, edgeworth_pdf(bin_centres),
                 label='{0}'.format('Edgeworth expansion'), linewidth=2)
else:
    if edgeworth_series is True:
        plt.plot(bin_centres, edgeworth_pdf(bin_centres),
                 label='{0}'.format('Edgeworth'), linewidth=2,
                 color=CB_color_cycle[2])

        plt.axvline(analytic_gauss_deviation_pos, color='dimgrey',
                    linestyle='dashed', linewidth=2)

    plt.plot(bin_centres, sci_stat.norm.pdf(bin_centres,
             analytic_N_mean, analytic_N_st),
             label='{0}'.format('Gaussian'), linewidth=2,
             color=CB_color_cycle[1])

plt.xlabel(r'$\mathcal{N}$', fontsize=fontsize)
plt.ylabel(r'$P(\mathcal{N})$', fontsize=fontsize)
plt.ylim(bottom=np.min(heights[heights > 0]))
plt.xlim(right=np.max(bin_centres[heights > 0]))
if kazuya_pdf is True:
    if phi_r == phi_i:
        if m == 2:  # From applying residual theorem to leading pole
            def kazuya_pdf_new(N):
                return 4.39565*np.exp(-0.391993*N)
        elif m == 1:  # From applying residual theorem to leading pole
            def kazuya_pdf_new(N):
                return 213.842*np.exp(-0.652823*N)
        plt.plot(bin_centres[bin_centres > 15],
                 kazuya_pdf_new(bin_centres[bin_centres > 15]),
                 label='{0}'.format(r'Leading pole'), linewidth=2,
                 color=CB_color_cycle[2])
        histogram_name += '_kazuya_pdf'
# In case you want to change label order
handles, labels = plt.gca().get_legend_handles_labels()
order = [i for i in range(len(handles))]
handles = [handles[i] for i in order]
labels = [labels[i] for i in order]
plt.legend(fontsize=fontsize, handles=handles, labels=labels)
plt.yscale('log')
plt.margins(tight=True)
plt.savefig(histogram_name+'.pdf', transparent=True)
plt.show()
plt.close()
