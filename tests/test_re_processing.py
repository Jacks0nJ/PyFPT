import unittest
import numpy as np

from pyfpt.numerics import re_processing


class TestReProcessing(unittest.TestCase):
    def test_re_processing(self):
        # First, testing with unweighted data
        num_bins = 50
        np.random.seed(0)  # For consistent tests, want the same seed
        data =\
            np.random.normal(3, 1, size=100000)
        # As the weights should produce a null effect on the overall
        # distribution, the expected histogram is just that of the unweighted
        # data.
        heights_expected, bins_expected =\
            np.histogram(data, bins=num_bins, density=True)
        bin_centres_expected =\
            np.array([(bins_expected[i]+bins_expected[i+1])/2
                     for i in range(num_bins)])

        # Using PyFPT's data analysis
        bin_centres_naive, heights_naive, _ =\
            re_processing(data, bins=num_bins, estimator='naive')
        # Only check filled bins, which can be determined by comparing the bin
        # centre values. Checking if the truncation is correct is another test
        # script.
        diff_naive = np.zeros(len(heights_naive))
        for i, bin_centre in enumerate(bin_centres_naive):
            # The smallest differance should correspond to the same bin
            bin_diff = np.abs((bin_centres_expected-bin_centre)/bin_centre)
            j = np.argmin(bin_diff)
            diff_naive[i] =\
                np.abs((heights_naive[i]-heights_expected[j]) /
                       heights_expected[j])
        # Now check that differance is very small, as it should be identical
        self.assertTrue(all(diff_naive < 0.00001))

        # Testing this function y using mock data, where although data is
        # weighted, the outcome should be the same as if all weights were 1.
        # This is because the data is centred at the same mean and weights
        # have mean of equals one.
        # Want data weights uncorrelated with data, apart from being centred
        # together. Also need mean of lognormal dist to be 1. This is done
        # using the definition of the lognormal mean
        np.random.seed(0)  # For consistent tests, want the same seed
        dist =\
            np.random.multivariate_normal([5, np.log(1)-0.5*0.01**2],
                                          [[0.1, 0], [0, 0.01]], size=100000)
        # Define data and weights from this multivaraint distribution
        data = dist[:, 0]
        weights = np.exp(dist[:, 1])  # Remeber weights are lognormally dist

        # As the weights should produce a null effect on the overall
        # distribution, the expected histogram is just that of the unweighted
        # data.
        heights_expected, bins_expected =\
            np.histogram(data, bins=num_bins, density=True)
        bin_centres_expected =\
            np.array([(bins_expected[i]+bins_expected[i+1])/2
                     for i in range(num_bins)])

        # First let's test the naive estimator
        bin_centres_naive, heights_naive, _ =\
            re_processing(data, weights=weights, estimator='naive',
                          bins=num_bins)
        # Only check filled bins, which can be determined by comparing the bin
        # centre values. Checking if the truncation is correct is another test
        # script.
        diff_naive = np.zeros(len(heights_naive))
        for i, bin_centre in enumerate(bin_centres_naive):
            # The smallest differance should correspond to the same bin
            bin_diff = np.abs((bin_centres_expected-bin_centre)/bin_centre)
            j = np.argmin(bin_diff)
            diff_naive[i] =\
                np.abs((heights_naive[i]-heights_expected[j]) /
                       heights_expected[j])
        # Now check that differance is small. A small differance is expected
        # as comparing weighted data against unweighted expectation.
        self.assertTrue(all(diff_naive < 0.05))

        # Now let's do the same for the lognormal distribution

        # First let's test the lognormal estimator
        bin_centres_lognormal, heights_lognormal, _ =\
            re_processing(data, weights=weights, estimator='lognormal',
                          bins=num_bins)
        # Only check filled bins, which can be determined by comparing the bin
        # centre values. Checking if the truncation is correct is another test
        # script.
        diff_lognormal = np.zeros(len(heights_lognormal))
        for i, bin_centre in enumerate(bin_centres_lognormal):
            # The smallest differance should correspond to the same bin
            bin_diff = np.abs((bin_centres_expected-bin_centre)/bin_centre)
            j = np.argmin(bin_diff)
            diff_lognormal[i] =\
                np.abs((heights_lognormal[i]-heights_expected[j]) /
                       heights_expected[j])
        # Now check that differance is small
        self.assertTrue(all(diff_lognormal < 0.05))


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
