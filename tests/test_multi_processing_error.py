import unittest
import numpy as np

from pyfpt.numerics import multi_processing_error


class TestMultiProcessingError(unittest.TestCase):
    def test_multi_processing_error(self):
        # Uncorrelated data

        # Need to draw the same random numbers each time, for consistent tests
        np.random.seed(1)
        dist_uncol =\
            np.random.multivariate_normal([0, -5], [[1, 0], [0, 1]], size=1000)
        # Need to make the y data into a lognormal by exponentiating
        dist_uncol[:, 1] = np.exp(dist_uncol[:, 1])
        result_uncol =\
            multi_processing_error(dist_uncol[:, 0], dist_uncol[:, 1])
        self.assertTrue(result_uncol)
        # Correlated data

        # Need to draw the same random numbers each time, for consistent tests
        np.random.seed(1)
        dist_col = np.random.multivariate_normal([0, -5], [[1, 0.8], [0.8, 1]],
                                                 size=1000)
        # Need to make the y data into a lognormal by expoentiating
        dist_col[:, 1] = np.exp(dist_col[:, 1])
        result_uncol =\
            multi_processing_error(dist_col[:, 0], dist_col[:, 1])
        self.assertFalse(result_uncol)


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
