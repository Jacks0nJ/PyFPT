#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 19:48:34 2022

@author: jjackson
"""

#plotting the different distributions on top of each other
fig, ax1 = plt.subplots()
if storage == True:
    colour_cycle_og = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',\
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    #Custom colour order, to be consistant with the other graphs.
    colour_cycle = ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd',\
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    data_dict = {}
    left, bottom, width, height = [0.62, 0.62, 0.23, 0.23]
    ax2 = fig.add_axes([left, bottom, width, height])
    #Looping over all of the different plots
    for j in range(len(bias_range_og)):
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
            if bias_range_og[j]==0:
                ax1.errorbar(bin_centres_temp, heights_temp, yerr=errors_temp,\
                             capsize=3, fmt =".", ms=7, color = colour_cycle[j+2],
                             label='{0}'.format(r'Direct ($\mathcal{A}=0$)'))
                ax2.errorbar(bin_centres_temp, heights_temp, yerr=errors_temp,\
                             capsize=3, fmt =".", ms=7, color = colour_cycle[j+2],
                             label='{0}'.format(r'Direct ($\mathcal{A}=0$)'))
                
            else:
                ax1.errorbar(bin_centres_temp, heights_temp, yerr=errors_temp,\
                             capsize=3, fmt =".", ms=7, color = colour_cycle[j+2],
                             label='{0}'.format(r'$\mathcal{A}$ = ' +\
                                  ('%s' % float('%.3g' % bias_range_og[j]))))
                ax2.errorbar(bin_centres_temp, heights_temp, yerr=errors_temp,\
                             capsize=3, fmt =".", ms=7, color = colour_cycle[j+2],
                             label='{0}'.format(r'Direct ($\mathcal{A}=0$)'))
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

    my_file_name = 'combined_data_bias_range_'\
                +('%s' % float('%.3g' % np.min(bias_range)))+'_to_'\
                    +('%s' % float('%.3g' % np.max(bias_range)))+'_with_m_'\
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
        ax1.plot(bin_centres_combined, cosfuncs.pdf_gaussian_skew_kurtosis(bin_centres_combined, analytic_N_mean,\
                        analytic_N_var**0.5, analytic_N_skew*analytic_N_var**1.5,\
                            analytic_N_4th_cmoment-3*analytic_N_var**2),\
                         label='{0}'.format('Edgeworth'), linewidth = 2,\
                             color = colour_cycle[1])
        ax1.plot(bin_centres_combined, sci_stat.norm.pdf(bin_centres_combined,\
                analytic_N_mean, analytic_N_var**0.5),\
                label='{0}'.format('Gaussian'), color = colour_cycle[0])
    else:
        ax1.plot(bin_centres_combined, sci_stat.norm.pdf(bin_centres_combined,\
                analytic_N_mean, analytic_N_var**0.5),\
                label='{0}'.format('Gaussian'), color = colour_cycle[1])

    overlap_name = 'overlap_plot_m_'+str(m)+\
        '_bias_range_log_'+('%s' % float('%.3g' % np.min(bias_range_og)))+\
        '_to_'+('%s' % float('%.3g' % np.max(bias_range_og))) +'_phi_UV_'+\
        str(phi_r/phi_i)+'phi_i'
    if kazuya_pdf == True:
        kazuya_data = pd.read_csv(comp_language+'_results/'+'kazuya_results_m_0.3.csv', index_col=0)
        plt.plot(kazuya_data['N'], kazuya_data['pdf'], label='{0}'.format('Kazuya'))
        overlap_name += '_kazuya'

    if vincent == True and bin_centres_combined[-1]>20:
        if m<1.5:
            plt.plot(bin_centres_combined[bin_centres_combined>20],\
                     cosfuncs.vincent_near_tail_fit(bin_centres[bin_centres_combined>20],\
                    m, phi_i), label='{0}'.format('Vincent near tail'))
        else:
            plt.plot(bin_centres[bin_centres>20],\
                     cosfuncs.vincent_far_tail_fit(bin_centres[bin_centres>20],\
                    m, phi_i), label='{0}'.format('Vincent far tail'))

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

    ax1.axvline(analytic_gauss_deviation_pos, color='dimgrey',\
                linestyle='dashed', linewidth=2)
    ax1.set_ylim(bottom = np.min(heights_combined))
    ax1.set_yscale('log')
    
    ax2.set_xlim(left = bin_centres_combined[82], right = bin_centres_combined[106])
    ax2.set_ylim(top = heights_combined[82], bottom = heights_combined[106])
    ax2.set_xticks(ticks=[11.8,12.1])
    ax2.set_yscale('log')

    if publication_plots == True:
        ax1.set_xlabel(r'$\mathcal{N}$', fontsize = fontsize)
        ax1.set_ylabel(r'$P(\mathcal{N})$', fontsize = fontsize)
        ax1.legend(loc='lower left', fontsize = fontsize)
        plt.savefig('for_paper/'+overlap_name+'_up_dated_colours_inset.pdf', box_inches='tight',\
                    transparent=True)
        plt.show()
        plt.close()
    else:
        plt.axvline(N_star, color='k', linestyle='dashed', linewidth=2,\
            label='{0}'.format(r'$<\mathcal{N}>+4\sqrt{\delta \mathcal{N}^2}$'))
        plt.xlabel(r'$\mathcal{N}$')
        plt.ylabel('Probability Density')
        plt.title(str(len(bias_range_og)) +r' $\times$ '+str(num_sims_used) +\
                  r', $dN$=' + ('%s' % float('%.2g' % dN))+', m=' +\
                  ('%s' % float('%.3g' % m)))
        plt.legend(loc='lower left')
        plt.savefig(comp_language+'_results/'+overlap_name+'_up_dated_colours_inset.pdf',transparent=True)
        plt.show()
        plt.close()