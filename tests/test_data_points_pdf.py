import unittest
import numpy as np

from pyfpt.numerics import data_points_pdf


class TestDataPointsPdf(unittest.TestCase):
    def test_data_points_pdf(self):
        # Testing this function y using mock data, where although data is
        # weighted, the outcome should be the same as if all weights were 1.
        # This is because the data is centred at the same mean and weights
        # have mean of equals one.
        num_bins = 50
        # Want data weights uncorrelated with data, apart from being centred
        # together. Also need mean of lognormal dist to be 1. This is done
        # using the definition of the lognormal mean
        np.random.seed(0)
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
        bin_centres_naive, heights_naive, _, _, _ =\
            data_points_pdf(data, weights, 'naive', bins=num_bins)
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
        # Now check that differance is small
        self.assertTrue(all(diff_naive < 0.05))

        # Now let's do the same for the lognormal distribution

        # First let's test the lognormal estimator
        bin_centres_lognormal, heights_lognormal, _, _, _ =\
            data_points_pdf(data, weights, 'lognormal', bins=num_bins)
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
