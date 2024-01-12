import unittest
import numpy as np

from pyfpt.numerics import log_normal_height


class TestLogNormalHeight(unittest.TestCase):
    def test_log_normal_height(self):
        # Need to draw the same random numbers each time, for consistent tests
        np.random.seed(1)
        w_dist = np.random.lognormal(size=1000000)

        height = log_normal_height(w_dist)
        # The mean should be close to exp(0.5) by definition. The height is
        # just the mean times the number of data points. So dividing height by
        # the number of data points should recover the mean
        result = height/len(w_dist)
        # Rounding the value, so an approxiate equality can be done
        result = round(result, 2)

        expected = round(np.exp(0.5), 2)
        self.assertEqual(result, expected)


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
