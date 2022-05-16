import unittest
import numpy as np

from pyfpt.numerics import data_in_histogram_bins


class TestDataInHistogramBins(unittest.TestCase):
    def test_data_in_histogram_bins(self):
        num_bins = 50
        # Using mock correlated data, to see if the weights in
        # data_in_histogram_bins reproduce the bin heights, as they should

        # Need to draw the same random numbers each time, for consistent tests
        np.random.seed(0)
        # Look at large sample first, to use the same bins as small sample
        data =\
            np.random.multivariate_normal([0, 5], [[1, 0.98], [0.98, 1]],
                                          size=100000)
        expected, bins =\
            np.histogram(data[:, 0], bins=num_bins,
                         weights=data[:, 1])
        # Remove zeros
        expected = expected[expected > 0]
        data_columned, weights_columned =\
            data_in_histogram_bins(data[:, 0], data[:, 1], bins)

        result = np.sum(weights_columned, axis=0)
        # Remove zeros
        result = result[result > 0]
        self.assertTrue(all(np.abs((result-expected)/expected) < 0.0001))


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
