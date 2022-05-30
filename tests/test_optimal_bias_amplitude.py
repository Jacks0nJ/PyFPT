import unittest
import numpy as np

from pyfpt.analytics import optimal_bias_amplitude  # noqa: E402


class TestOptimalBiasAmplitude(unittest.TestCase):
    def test_optimal_bias_amplitude(self):
        # Using the drift-dominated (classical) result for the number of
        # e-folds for quadratic inflation, and assuming the functional form for
        # the bias chosen is to match it to the drift, a simple expression can
        # be found to relate the bias amplitude and target number of e-folds.
        # This can then be used to test against
        m = 0.01
        N_starting = 60
        phi_end = 2**0.5
        phi_in = (4*N_starting+2)**0.5

        def V(phi):
            V = 0.5*(m*phi)**2
            return V

        def V_dif(phi):
            V_dif = (m**2)*phi
            return V_dif

        def bias_func(phi):
            # Notice it is positive, as the drift is negative
            return 2/phi

        # Investigating the N range near CMB
        N_target_values = np.array([59, 59.5, 60.5, 61., 62])

        results =\
            [optimal_bias_amplitude(N_target, phi_in, phi_end, V, V_dif,
                                    bias_function=bias_func)
             for N_target in N_target_values]
        expected = 1-(phi_in**2-phi_end**2)/(4*N_target_values)
        differance = np.abs(expected-results)
        self.assertTrue(all(differance < 10**-6))


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
