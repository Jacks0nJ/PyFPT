import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sci_stat


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
