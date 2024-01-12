import unittest
import numpy as np

from pyfpt.numerics import histogram_data_truncation


class TestHistogramDataTruncation(unittest.TestCase):
    def test_histogram_data_truncation(self):
        num_data_points = 100000
        np.random.seed(1)
        # Using a normal distribution as mock data
        data = np.random.normal(size=num_data_points)
        # Also need weights, does not matter what values they have
        weights = np.random.normal(size=num_data_points)
        # Need to specify a threshold above which is truncated
        threshold = 3.5
        # The number of sub samples. This required when the data needs to be
        # evenly subdivided for resampling.
        num_sub_samples = 100

        # Testing unweighted data truncation

        truncated_data = histogram_data_truncation(data, threshold)
        # Check all the data is below the threshold
        self.assertTrue(any(truncated_data < threshold))
        # Now let's see if we can truncate the data so it can still be evenly
        # subdivided
        truncated_data =\
            histogram_data_truncation(data, threshold,
                                      num_sub_samples=num_sub_samples)
        self.assertTrue(any(truncated_data < threshold))
        # See if truncated a full subsamples worth
        self.assertIs(len(truncated_data) % num_sub_samples, 0)

        # Testing weighted data

        truncated_data, truncated_weights =\
            histogram_data_truncation(data, threshold, weights=weights)
        self.assertTrue(any(truncated_data < threshold))
        # Checking if weights also truncated, as same number should be removed
        self.assertEqual(len(truncated_data), len(truncated_weights))
        # Now let's see if we can truncate the data so it can still be evenly
        # subdivided
        truncated_data, truncated_weights =\
            histogram_data_truncation(data, threshold, weights=weights,
                                      num_sub_samples=num_sub_samples)
        self.assertTrue(any(truncated_data < threshold))
        # See if truncated a full subsamples worth
        self.assertIs(len(truncated_data) % num_sub_samples, 0)
        # Checking if weights also truncated, as same number should be removed
        self.assertEqual(len(truncated_data), len(truncated_weights))


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
