import unittest
import numpy as np

from pyfpt.numerics import log_normal_error


class TestlogNormalError(unittest.TestCase):
    def test_log_normal_error(self):
        # Need to draw the same random numbers each time, for consistent tests
        np.random.seed(1)
        w_dist = np.random.lognormal(size=1000000)

        error = log_normal_error(w_dist)
        # Need to normalise error
        error = error/len(w_dist)

        # Need to check both upper and lower error
        for i in range(len(error)):
            # The error should be close to zero for this large sample
            result = error[i]
            self.assertTrue(result < 10**-2)


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
