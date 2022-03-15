from .fourth_central_moment_N_sto_limit import\
    fourth_central_moment_N_sto_limit
from .variance_N_sto_limit import variance_N_sto_limit

M_PL = 1


# Using the standard relation between the central moments and the kurtosis.
# Fisher is an optional argument
def kurtosis_N_sto_limit(V, V_dif, V_ddif, phi_int, phi_end, Fisher=True):
    # The excess kurtosis over the expected Gaussian amount
    fourth_moment = fourth_central_moment_N_sto_limit(V, V_dif, V_ddif,
                                                      phi_int, phi_end)
    var = variance_N_sto_limit(V, V_dif, V_ddif, phi_int, phi_end)
    if Fisher is True:
        return fourth_moment/var**2-3
    else:
        return fourth_moment/var**2
