import unittest
import numpy as np

from pyfpt.numerics import histogram_normalisation


class TestHistogramNormalisation(unittest.TestCase):
    def test_histogram_normalisation(self):
        num_data_points = 100000
        # Testing uneven bins
        bins = np.array([-4., -3.3, -3.2, -3., -2.9, -2.4, -1.9, -1.5, -1.45,
                         -1.3, -0.9, -0.6, -0.4, -0.05, 0.1, 0.6, 0.8, 1.,
                         1.3, 1.8, 2., 2.4, 3., 3.1, 3.5, 4.])
        # Need to draw the same random numbers each time, for consistent tests
        np.random.seed(1)
        # Using a normal distribution as mock data
        data = np.random.normal(size=num_data_points)

        # The normalisation per bin used in histogram_normalisation.py should
        # be approxiately the same as numpy's. So testing against numpy.
        # This of course requires the total area to 1 when normalsied, which
        # is the case for unweighted data
        heights_raw, _ = np.histogram(data, bins=bins)
        heights_normed, _ = np.histogram(data, bins=bins, density=True)
        expected = heights_raw/heights_normed

        result = histogram_normalisation(bins, num_data_points)
        # Just want overal normalisation, not normalisation per bin
        result = result

        # Want the differance to be small
        diff = np.abs((result-expected)/expected)
        self.assertTrue(all(diff <= 0.001))


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
