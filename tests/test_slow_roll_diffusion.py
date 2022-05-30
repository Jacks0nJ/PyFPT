import unittest
import numpy as np

from pyfpt.analytics import slow_roll_diffusion  # noqa: E402


class TestSlowRollDiffusion(unittest.TestCase):
    def test_slow_roll_diffusion(self):
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

        result_func = slow_roll_diffusion(V, V_dif)

        # Analytical diffusion for quadratic inflation
        def expected_func(phi, N):
            pi = 3.141592653589793
            sqirt_6 = 2.449489742783178
            return (m*phi)/(2*pi*sqirt_6)
        phi_range = np.linspace(phi_end, 100*phi_i, 100)
        differance = np.abs(result_func(phi_range, 10) -
                            expected_func(phi_range, 10))
        self.assertTrue(all(differance < 10**-6))


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
