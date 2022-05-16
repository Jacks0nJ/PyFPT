import unittest
import numpy as np

from pyfpt.numerics import jackknife_errors


class TestJackknifeErrors(unittest.TestCase):
    def test_jackknife_errors(self):
        num_bins = 50
        num_sub_samples = 20
        # Comparing correlated data, seeing if the error size is reasonable
        # and the error approximately scales with sqrt(#samples)

        # Need to draw the same random numbers each time, for consistent tests
        np.random.seed(0)
        # Look at large sample first, to use the same bins as small sample
        dist_large_sample =\
            np.random.multivariate_normal([0, 5], [[1, 0.98], [0.98, 1]],
                                          size=100000)
        heights_large_sample, bins_large_sample =\
            np.histogram(dist_large_sample[:, 0], bins=num_bins,
                         weights=dist_large_sample[:, 1], density=True)
        errors_large_sample =\
            jackknife_errors(dist_large_sample[:, 0], dist_large_sample[:, 1],
                             bins_large_sample, num_sub_samples)
        # The well sampled bins should have errors less than a percent of the
        # height of the bar. Let's just look at the bins with the largest
        # number of samples
        error_fraction_large_sample =\
            errors_large_sample[20:30]/heights_large_sample[20:30]
        self.assertTrue(all(error_fraction_large_sample < 0.1))

        # Now let's look at a smaller sample size
        np.random.seed(1)
        dist_small_sample =\
            np.random.multivariate_normal([0, 5], [[1, 0.95], [0.95, 1]],
                                          size=10000)
        heights_small_sample, bins_small_sample =\
            np.histogram(dist_small_sample[:, 0], bins=num_bins,
                         weights=dist_small_sample[:, 1], density=True)
        errors_small_sample =\
            jackknife_errors(dist_small_sample[:, 0], dist_small_sample[:, 1],
                             bins_large_sample, num_sub_samples)
        # The well sampled bins should have errors less than a percent of the
        # height of the bar. Let's just look at the bins with the largest
        # number of samples
        error_fraction_small_sample =\
            errors_small_sample[20:30]/heights_small_sample[20:30]
        self.assertTrue(all(error_fraction_small_sample < 0.3))

        # The error should scale very approximately with sqrt(#samples)

        ratio = error_fraction_small_sample/error_fraction_large_sample
        expected = (dist_large_sample.shape[0]/dist_small_sample.shape[0])**0.5
        self.assertTrue(all(np.abs((ratio - expected)/expected) < 1))


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
