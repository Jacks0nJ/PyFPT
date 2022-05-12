import unittest

from pyfpt.analytics import reduced_potential_diff  # noqa: E402


class TestReducedPotentialDiff(unittest.TestCase):
    def test_reduced_potential_diff(self):
        m = 0.01

        def V_diff(phi):
            V = phi*m**2
            return V

        # As this is a small number, increasing it to round correctly
        func = reduced_potential_diff(V_diff)
        # There is some machine noise, so need to round result
        result = round(func(6)*10**6, 6)
        expected = 2.53303  # Taken from previously working code
        self.assertEqual(result, expected)


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
