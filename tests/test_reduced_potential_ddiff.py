import unittest

from pyfpt.analytics import reduced_potential_ddiff  # noqa: E402


class TestReducedPotentialDDiff(unittest.TestCase):
    def test_reduced_potential_ddiff(self):
        m = 0.01

        def V_ddiff(phi):
            V = m**2
            return V

        # As this is a small number, increasing it to round correctly
        func = reduced_potential_ddiff(V_ddiff)
        # There is some machine noise, so need to round result
        result = round(func(6)*10**7, 6)
        expected = 4.221716  # Taken from previously working code
        self.assertEqual(result, expected)


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
