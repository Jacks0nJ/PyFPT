import unittest
import numpy as np

from pyfpt.analytics import slow_roll_drift  # noqa: E402


class TestSlowRollDrift(unittest.TestCase):
    def test_slow_roll_drift(self):
        m = 0.01
        N_starting = 10
        phi_end = 2**0.5
        phi_i = (4*N_starting+2)**0.5

        def V(phi):
            V = 0.5*(m*phi)**2
            return V

        def V_dif(phi):
            V_dif = (m**2)*phi
            return V_dif

        def V_ddif(phi):
            V_ddif = (m**2)
            return V_ddif

        result_func = slow_roll_drift(V, V_dif)

        # Analytical drift for quadratic inflation
        def expected_func(phi, N):
            return -2/phi
        phi_range = np.linspace(phi_end, 100*phi_i, 100)
        differance = np.abs(result_func(phi_range, 10) -
                            expected_func(phi_range, 10))
        self.assertTrue(all(differance < 10**-6))


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
