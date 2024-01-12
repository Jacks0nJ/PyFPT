import unittest

from pyfpt.analytics import reduced_potential  # noqa: E402


class TestReducedPotential(unittest.TestCase):
    def test_reduced_potential(self):
        m = 0.01

        def V(phi):
            V = 0.5*(m*phi)**2
            return V

        # As this is a small number, increasing it to round correctly
        func = reduced_potential(V)
        # There is some machine noise, so need to round result
        result = round(func(6)*10**6, 6)
        expected = 7.599089  # Taken from previously working code
        self.assertEqual(result, expected)


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
