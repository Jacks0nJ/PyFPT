import unittest
import numpy as np

from pyfpt.analytics import edgeworth_pdf  # noqa: E402


class TestEdgeworthPdf(unittest.TestCase):
    def test_edgeworth_pdf(self):
        m = 0.01
        N_starting = 10
        phi_end = 2**0.5
        phi_i = (4*N_starting+2)**0.5
        # Checking a range of N values
        N_values = np.linspace(9.9, 10.1, 5)

        def V(phi):
            V = 0.5*(m*phi)**2
            return V

        def V_dif(phi):
            V_dif = (m**2)*phi
            return V_dif

        def V_ddif(phi):
            V_ddif = (m**2)
            return V_ddif

        func = edgeworth_pdf(V, V_dif, V_ddif, phi_i, phi_end)
        # There is some machine noise, so need to round result
        # As this is a small number, increasing it to round correctly
        result = func(N_values)*10**3
        # Taken from previously working code
        expected = [6.50049510, 2282.31131, 15629.1737, 2306.95516, 8.09054141]
        for i in range(len(result)):
            result[i] = round(result[i], 5)
            self.assertEqual(result[i], expected[i])


# We need the following to execute the tests when we run the file in python
if __name__ == '__main__':
    unittest.main()
