import unittest

from pyfpt.analytics import fourth_central_moment_efolds  # noqa: E402


class TestFourthCentralMomentEFolds(unittest.TestCase):
    def test_fourth_central_moment_efolds(self):
        m = 0.01
        N_starting = 10
        phi_end = 2**0.5
        phi_i = (4*N_starting+2)**0.5

        def V(phi):
            V = 0.5*(m*phi)**2
            return V

        def V_dif(phi):
            V_dif = (m**2)*phi
            return V_dif

        def V_ddif(phi):
            V_ddif = (m**2)
            return V_ddif
        # As this is a small number, increasing it to round correctly
        result =\
            fourth_central_moment_efolds(V, V_dif, V_ddif, phi_i,
                                         phi_end)*10**6
        # There is some machine noise, so need to round result
        result = round(result, 6)
        expected = 1.273763  # Taken from previously working code
        self.assertEqual(result, expected)


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
