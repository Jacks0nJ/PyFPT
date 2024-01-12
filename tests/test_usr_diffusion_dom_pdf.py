import unittest
import numpy as np
from scipy.integrate import quad
from pyfpt.analytics import usr_diffusion_dom_pdf  # noqa: E402


class TestUSRDiffusionDomPDF(unittest.TestCase):
    def test_usr_diffusion_dom_pdf(self):
        x = 1.
        mu = 1.0
        y = 0.1

        pdf_func = usr_diffusion_dom_pdf(x, y, mu)
        N_arr = np.array([0.1, 1, 2, 3, 4, 5])
        results = pdf_func(N_arr)
        # From simply hand checking
        expected = np.array([1.60815555e+00, 2.46483013e-01, 2.06262041e-02,
                             1.74803604e-03, 1.48237214e-04, 1.25712321e-05])
        # In case they are perfectly alike, to avoid divide by zero
        expected = 0.99999*expected
        differance = np.abs((results - expected)/expected)
        self.assertTrue(all(differance < 0.01))

        # Checking if normalsied
        total_area, _ = quad(pdf_func, 0, np.inf)
        differance = np.abs(total_area-1)
        self.assertTrue(differance < 0.0001)


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
