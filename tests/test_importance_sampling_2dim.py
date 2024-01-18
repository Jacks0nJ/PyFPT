import unittest
import numpy as np

from pyfpt.numerics.importance_sampling_cython\
    import importance_sampling_simulations_2dim


# This is the most basic initial test, needs further refinement
class TestImportanceSamplingNDim(unittest.TestCase):
    def test_importance_sampling_ndim(self):
        # Although We'll only be simulating a 1D problem, we'll be doing it in
        # a multidim way. This is because this is purely a test of the code.

        # Let's test the boundaries using a simple random walk and no bias
        x_in = 0.
        y_in = 0.
        x_r = -5
        x_end = 5
        dt = 0.05
        t_f = 10000
        bias_amp = 0.
        num_runs = 500
        diffusion_amp = 1.
        drift_amp = 0.

        # The zone which is close to the end surface
        x_near_end = x_end - 2*diffusion_amp*(dt**0.5)

        def update_func(x, y, A, t, dt, dW):
            x += drift_amp*dt + diffusion_amp*dW[0] +\
                bias_amp*diffusion_amp*dt
            # Also need to apply reflective boundary here
            if direction*x > direction*x_r:
                x = 2*x_r - x
            A += bias_amp*(0.5*bias_amp*dt + dW[0])

            return x, y, A

        def end_cond(x, y, t):
            if x*direction <= x_end*direction:
                return 1
            elif x*direction <= x_near_end*direction:
                return -1
            else:
                return 0

        def expected_mean_FPT(x_in, x_r, x_end):
            # This is from Eq. (5.7) in arXiv:1707.00537v3, but generalised
            x = np.abs((np.diff((x_in, x_end)))/np.diff((x_end, x_r)))
            mu_sqaured = 2*(np.abs(np.diff((x_end, x_r)))/diffusion_amp)**2
            expectation = x*(1-x/2)*mu_sqaured
            return expectation

        direction = x_in - x_end

        ts, ws =\
            importance_sampling_simulations_2dim(x_in, y_in, 0., t_f, dt,
                                                 num_runs, end_cond,
                                                 update_func)
        ts = np.array(ts)
        # The ws should all be 1
        self.assertTrue(all(np.array(ws) == 1))
        # There is also an analytical expectation for the mean.
        expectation = expected_mean_FPT(x_in, x_r, x_end)
        differance = np.abs((expectation-np.mean(ts))/expectation)
        self.assertTrue(differance[0] < 0.2)

        # Let's now try swapping round the boundaries and starting at the
        # reflective boundary
        x_in = 5.
        x_r = 5
        x_end = -2
        # Remeber to redine the the near end surface value
        x_near_end = x_end + 2*diffusion_amp*(dt**0.5)
        direction = x_in - x_end
        ts, ws =\
            importance_sampling_simulations_2dim(x_in, y_in, 0., t_f, dt,
                                                 num_runs, end_cond,
                                                 update_func)
        ts = np.array(ts)
        # The ws should all be 1
        self.assertTrue(all(np.array(ws) == 1))
        # There is also an analytical expectation for the mean.
        expectation = expected_mean_FPT(x_in, x_r, x_end)
        differance = np.abs((expectation-np.mean(ts))/expectation)
        self.assertTrue(differance[0] < 0.2)

        # Now let's run a long simulation drift domiated simulation
        diffusion_amp = 0.00000001
        drift_amp = -1.
        num_runs = 10**1
        ts, ws =\
            importance_sampling_simulations_2dim(x_in, y_in, 0., t_f, dt,
                                                 num_runs, end_cond,
                                                 update_func)

        # All of these simulations should take the classical amount of time
        expectation = np.abs(np.diff((x_in, x_end))/drift_amp)
        differance = np.abs((expectation-ts))
        # The differance should be smaller than the step size
        self.assertTrue(np.all(differance < dt))

        # Finally, let's test if the bias is working as intended. The most
        # basic test we can do to test this is to see if a drift dominated sim
        # still reproduces expected value even when there is a bias
        # Now let's run a long simulation drift domiated simulation
        diffusion_amp = 0.01
        drift_amp = -1.
        num_runs = 10**3
        # Need to reduce the time step to be small than the diffusion
        dt = 0.01
        # Only using a small bias
        bias_amp = 1.0
        ts, ws =\
            importance_sampling_simulations_2dim(x_in, y_in, 0., t_f, dt,
                                                 num_runs, end_cond,
                                                 update_func)
        # The mean of the sample distrbution should be a few std from the
        # target mean.
        expectation_mean = np.abs(np.diff((x_in, x_end))/drift_amp)
        dist_from_targ_mean = (np.mean(ts)-7)/np.std(ts)
        self.assertTrue(dist_from_targ_mean > 2)
        # But it should still approximately reproduce the unbiased mean
        # Just using the classical value as small diffusion
        differance =\
            np.abs((expectation_mean - np.average(ts, weights=ws)) /
                   expectation_mean)
        self.assertTrue(differance < 0.05)


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
