'''
Lognormalality Check
--------------------
This module checks if assuming the wieghts within each first-passage time bin
are drawn from an underlying lognormal distribution is correct, by calculating
the p-values using  `D’Agostino & Pearson's method`_. If any p-value is below
the 0.5% threshold, plots comparing the p-values of the data and the
theoretical predictions are given. If many p-avlues are less than 0.5%, or some
are much, much less than this value, it is likely the assumption is incorrect.

.. _D’Agostino & Pearson's method: https://en.wikipedia.org/wiki/D%27Agostino%\
    27s_K-squared_test
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sci_stat


def lognormality_check(bin_centres, weights_in_bins):
    """Checks if the distribution of weights within each first-passage time bin
    is drawn from an underlying lognormal distribution.

    Parameters
    ----------
    bin_centres : numpy.ndarray
        The centres of the histogram bins.
    weights_columned : numpy.ndarray
        The associated weights to these bins seperated into coloumns, with each
        column containing the weights of that bin.
    """
    # Checking p-values if lognormal was used
    p_values = np.zeros(len(bin_centres))
    p_values_theory = np.zeros(len(bin_centres))
    for i in range(len(bin_centres)):
        w = weights_in_bins[:, i]
        log_w = np.log(w[w > 0])
        _, p_values[i] = sci_stat.normaltest(log_w)
        _, p_values_theory[i] =\
            sci_stat.normaltest(np.random.normal(0, 1, len(log_w)))

    if any(p_values < 0.005) is True:
        print('Possibly not lognormal distribution, see p-value plots')
        plt.errorbar(bin_centres, p_values, fmt='.', ms=7)
        plt.hlines(0.005, np.min(bin_centres), np.max(bin_centres),
                   color='k', linestyle='dashed',
                   label='{0}'.format('0.5% threshold'), linewidth=2)

        plt.yscale('log')
        plt.title('Data: p-values with bin centres')
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
