import unittest
import numpy as np

from pyfpt.numerics import is_simulation_1dim
from pyfpt.analytics import edgeworth_pdf
from scipy.stats import chisquare


# This is the most basic initial test, needs further refinement
class TestIS_Simulation1Dim(unittest.TestCase):
    def test_is_simulation_1dim(self):
        # This tests needs to look at the numerical output to determine if it
        # working correctly. This of course requires us to run a simulation.
        # The simulation can then be compared to the analytical expectation,
        # which is checked in a seperate test
        m = 0.1
        N_starting = 10
        phi_end = 2**0.5
        phi_i = (4*N_starting+2)**0.5

        def potential(phi):
            V = 0.5*(m*phi)**2
            return V

        def potential_dif(phi):
            V_dif = (m**2)*phi
            return V_dif

        def potential_ddif(phi):
            V_ddif = (m**2)
            return V_ddif

        # Need to define the drift and diffusion as explicit functions
        def drift_func(phi, N):
            return -2/phi

        def diffusion_func(phi, N):
            pi = 3.141592653589793
            sqirt_6 = 2.449489742783178
            return (m*phi)/(2*pi*sqirt_6)

        bias_amp = 0.6
        # As it is a highly composite number, so should work for any core count
        num_runs = 55440

        # Let's run the simulation

        bin_centres, heights, errors =\
            is_simulation_1dim(drift_func, diffusion_func, phi_i, phi_end,
                               num_runs, bias_amp, 0.001, display=False)

        # Let's get a analaytical comparison
        analytical_func =\
            edgeworth_pdf(potential, potential_dif, potential_ddif, phi_i,
                          phi_end)
        expected = analytical_func(bin_centres)
        # SciPy's chi-squared needs the sum to be the same, re normalising
        heights = heights*(np.sum(expected)/np.sum(heights))
        # SciPy's chi-squared also needs large values for effective test
        _, p = chisquare(100*heights, 100*expected)

        # For this mass, the p value should be relatively large. Using the 0.5
        # % threshold also used to test lognormality
        self.assertTrue(p > 0.005)


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
