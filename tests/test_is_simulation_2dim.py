import unittest
import numpy as np

from pyfpt.numerics import is_simulation_2dim
from pyfpt.analytics import usr_diffusion_dom_pdf
from scipy.stats import chisquare


# This is the most basic initial test, needs further refinement
# It uses the pure USR inflation case, see https://arxiv.org/pdf/2101.05741.pdf
class TestIS_Simulation1Dim(unittest.TestCase):
    def test_is_simulation_1dim(self):
        # This tests needs to look at the numerical output to determine if it
        # working correctly. This of course requires us to run a simulation.
        # The simulation can then be compared to the analytical expectation,
        # which is checked in a seperate test
        x = 1
        x_end = 0.
        x_r = 1.
        y = 0.1
        mu = 1

        diff_const = (2**0.5)/mu

        bias = 0.5

        def update_func(x, y, A, N, dN, dW):
            # Update x
            x += (-3 * y + bias * diff_const) * dN +\
                diff_const * dW[0]
            if x > x_r:
                x = 2*x_r - x
            # Update y
            y += (-3 * y) * dN
            # Update the log of the weight
            A += bias*(0.5*bias*dN + dW[0])
            return x, y, A

        # Time step
        dN = 0.001
        # This let's us define the near end surface, as 2 std away from the end
        x_near_end = x_end + 2*diff_const*(dN**0.5)

        def end_cond(x, y, t):
            if x <= x_end:
                return 1
            elif x <= x_near_end:
                return -1
            else:
                return 0

        # As it is a highly composite number, so should work for any core count
        num_runs = 55440

        # Let's run the simulation

        bin_centres, heights, errors =\
            is_simulation_2dim(update_func, x, y, dN, num_runs, end_cond,
                               min_bin_size=100, estimator='naive',
                               display=False)

        # Let's get a analaytical comparison
        analytical_func = usr_diffusion_dom_pdf(x, y, mu)
        expected = analytical_func(bin_centres)
        # SciPy's chi-squared needs the sum to be the same, re normalising
        heights = heights*(np.sum(expected)/np.sum(heights))
        # SciPy's chi-squared also needs large values for an effective test
        _, p = chisquare(100*heights, 100*expected)

        # For this mass, the p value should be relatively large. Using the 0.5
        # % threshold also used to test lognormality
        self.assertTrue(p > 0.005)


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
