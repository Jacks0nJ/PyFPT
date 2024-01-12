'''
Lognormality Check
--------------------
This module checks if assuming the weights within each first-passage time bin
are drawn from an underlying lognormal distribution is correct, by calculating
the p-values using  `D’Agostino & Pearson's method`_. If any p-value is below
the 0.5% threshold, plots comparing the p-values of the data and the
theoretical predictions are given. If many p-values are less than 0.5%, or some
are much, much less than this value, it is likely the assumption is incorrect.

.. _D’Agostino & Pearson's method: https://en.wikipedia.org/wiki/D%27Agostino%\
    27s_K-squared_test
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sci_stat


def lognormality_check(bin_centres, weights_in_bins, display=True):
    """Checks if the distribution of weights within each first-passage time bin
    is drawn from an underlying lognormal distribution.

    Parameters
    ----------
    bin_centres : numpy.ndarray
        The centres of the histogram bins.
    weights_columned : numpy.ndarray
        The associated weights to these bins separated into columns, with each
        column containing the weights of that bin.
    display : bool, optional
        If True, p-value plots of both the real data, and the theoretical
        expectation if the underlying distribution is truly lognormal, are
        displayed.
    """
    # Checking p-values if lognormal was used
    p_values = np.zeros(len(bin_centres))
    p_values_theory = np.zeros(len(bin_centres))
    for i in range(len(bin_centres)):
        w = weights_in_bins[:, i]
        log_w = np.log(w[w > 0])
        if len(log_w) > 0:
            _, p_values[i] = sci_stat.normaltest(log_w)
            _, p_values_theory[i] =\
                sci_stat.normaltest(np.random.normal(0, 1, len(log_w)))
        else:  # If bin is empty, set to impossible number and remove later
            p_values[i] = -1
            p_values_theory[i] = -1

    # Remove any -1 values corresponding to empty bins
    remove_logic = p_values > 0
    p_values = p_values[remove_logic]
    p_values_theory = p_values_theory[remove_logic]
    bin_centres = bin_centres[remove_logic]

    if any(p_values < 0.005) is True:
        if display is True:
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
            plt.xlabel(r'$\mathcal{N}$', fontsize=20)
            plt.ylabel('p-values', fontsize=20)
            plt.show()
        else:
            print('Possibly not lognormal distribution. Smallest p-value is:')
            print(np.min(p_values))

        return True
    else:
        return False
