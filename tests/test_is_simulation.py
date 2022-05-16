import unittest
import numpy as np

from pyfpt.numerics import is_simulation
from pyfpt.analytics import edgeworth_pdf
from scipy.stats import chisquare


# This is the most basic initial test, needs further refinement
class TestIS_Simulation(unittest.TestCase):
    def test_is_simulation(self):
        # This tests needs to look at the numerical output to determine if it
        # working correctly. This of course requires us to run a simulation.
        # The simulation can then be compared to the analytical expectation,
        # which is checked in a seperate test
        m = 0.05
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

        bias_amp = 1.
        # As it is a highly composite number, so should work for any core count
        num_runs = 55440

        # Let's run the simulation

        bin_centres, heights, errors =\
            is_simulation(potential, potential_dif, potential_ddif, phi_i,
                          phi_end, num_runs, bias_amp)
        bin_centres = np.array(bin_centres)
        heights = np.array(heights)
        errors = np.array(errors)

        # Let's get a analaytical comparison
        analytical_func =\
            edgeworth_pdf(potential, potential_dif, potential_ddif, phi_i,
                          phi_end)
        _, p = chisquare(heights, analytical_func(bin_centres))

        # For this mass, the p value should be 1
        self.assertEqual(p, 1.)


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
