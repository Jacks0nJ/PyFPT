import unittest
import numpy as np
from pyfpt.analytics import usr_mean_efolds_diffusion_dom  # noqa: E402


class TestUSRMeanEFoldsDiffusionDom(unittest.TestCase):
    def test_usr_mean_efolds_diffusion_dom(self):
        x = 1.
        mu = 1.0
        y_arr = np.power(10, np.linspace(-2, 0, 5))
        results = usr_mean_efolds_diffusion_dom(x, y_arr, mu)
        # From simply hand checking
        expected = np.array([0.49800799, 0.49370071, 0.48007991, 0.43700713,
                             0.30079906])
        # In case they are perfectly alike, to avoid divide by zero
        expected = 0.99999*expected
        differance = np.abs((results - expected)/expected)
        self.assertTrue(all(differance < 0.01))


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
