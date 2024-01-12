import unittest
import numpy as np

from pyfpt.numerics import lognormality_check


class TestLognormalityCheck(unittest.TestCase):
    def test_lognormality_check(self):
        # Using mock bin data, to see if lognormality_check correctly
        # identifies lognormal and deviating bins, as well as checking it can
        # handle mutiple data inputs with sime zero place olders
        num_data_points = 10**4
        num_bins = 10

        # First looking at data array drawn from a lognormal distribution

        data_array = np.zeros((num_data_points, num_bins))
        # Want to produce mock distributions of weights within each bin
        for i in range(num_bins):
            # Need to draw the same random numbers each time, for consistent
            # tests
            np.random.seed(i)
            # Random number of sample of lognormal data points
            size = np.random.randint(num_data_points/2, high=num_data_points)
            # Random mean and standard deviation
            mean = np.random.uniform(low=-5, high=5)
            sigma = np.random.uniform(low=1, high=3)
            # Drawing from this random dist
            if i != 4:
                column_values =\
                    np.random.lognormal(mean=mean, sigma=sigma, size=size)
                # Store this
                data_array[:len(column_values), i] = column_values
            # So i=4 is empy, as an extra test
        # Now we have an array of mock lognomrally distributed data, let's see
        # if lognormality_check works as intended. Returns False if lognormally
        # distributed
        bins_true = np.linspace(0, 5, num_bins)  # Mock bins
        results_lognormal_case = lognormality_check(bins_true, data_array)
        self.assertFalse(results_lognormal_case)

        # Now let's test when one of the distributions is not normally
        # lognormall distributed. This deviation is achived by adding two
        # lognormal distributions together.

        data_array = np.zeros((num_data_points, num_bins))
        # Want to produce mock distributions of weights within each bin
        for i in range(num_bins):
            # Need to draw the same random numbers each time, for consistent
            # tests
            np.random.seed(i+10)
            # Random number of sample of lognormal data points
            size = np.random.randint(num_data_points/2, high=num_data_points)
            # Random mean and standard deviation
            mean = np.random.uniform(low=-5, high=5)
            sigma = np.random.uniform(low=1, high=3)
            # Drawing from this random dist
            column_values =\
                np.random.lognormal(mean=mean, sigma=sigma, size=size)
            if i != 4:
                # Store this
                data_array[:len(column_values), i] = column_values
            else:
                # This column will not be lognormally distributed, achived by
                # adding another distribution
                column_values2 =\
                    np.random.lognormal(mean=1.2*mean, sigma=0.8*sigma,
                                        size=size)
                data_array[:len(column_values), i] =\
                    column_values + column_values2
        bins_true = np.linspace(0, 5, num_bins)  # Mock bins
        results_lognormal_case =\
            lognormality_check(bins_true, data_array, display=False)
        self.assertTrue(results_lognormal_case)


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
