import unittest
import numpy as np

from pyfpt.numerics import log_normal_mean


class TestlogNormalMean(unittest.TestCase):
    def test_log_normal_mean(self):
        # Need to draw the same random numbers each time, for consistent tests
        np.random.seed(1)
        w_dist = np.random.lognormal(size=1000000)

        mean = log_normal_mean(w_dist)
        # Rounding the value, so an approxiate equality can be done
        result = round(mean, 2)
        # The mean should be close to exp(0.5) by definition
        expected = round(np.exp(0.5), 2)
        self.assertEqual(result, expected)


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
