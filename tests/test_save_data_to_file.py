import unittest
import numpy as np
import pandas as pd
import os

from pyfpt.numerics import save_data_to_file


class TestSaveDataToFile(unittest.TestCase):
    def test_save_data_to_file(self):
        # Sythetic data

        # Need to draw the same random numbers each time, for consistent tests
        np.random.seed(1)
        data = np.random.multivariate_normal([0, -5], [[1, 0.8], [0.8, 1]],
                                             size=1000)
        # Need to make the y data into a lognormal by exponentiating
        data[:, 1] = np.exp(data[:, 1])
        x_i = 10**0.5
        num_data_points = len(data[:, 1])
        bias = 4.
        save_data_to_file(data[:, 0], data[:, 1], x_i, num_data_points, bias)

        # Now lets read the back
        raw_file_name = 'IS_data_phi_i_' + ('%s' % float('%.3g' % x_i)) +\
            '_iterations_' + str(num_data_points) + '_bias_' +\
            ('%s' % float('%.3g' % bias)) + '.csv'
        # Remembering to remove column numbering
        data_read = pd.read_csv(raw_file_name, index_col=0)
        data_read_column1 = np.array(data_read['N'])
        data_read_column2 = np.array(data_read['w'])

        # Rather than check all of the data is the same, just check if the mean
        # is the same. This is a similair method to data hashing.

        result1 = np.mean(data_read_column1)
        result2 = np.mean(data_read_column2)

        expected1 = np.mean(data[:, 0])
        expected2 = np.mean(data[:, 1])

        self.assertEqual(result1, expected1)
        self.assertEqual(result2, expected2)
        # Deleting this test file
        os.remove(raw_file_name)


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
