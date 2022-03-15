from .mean_N_sto_limit import mean_N_sto_limit
from .variance_N_sto_limit import variance_N_sto_limit
from .skewness_N_sto_limit import skewness_N_sto_limit
from .kurtosis_N_sto_limit import kurtosis_N_sto_limit
from .edgeworth_pdf import edgeworth_pdf


# This returns a function which returns the Edgeworth expansion
def edgeworth_pdf_sto_limit(V, V_dif, V_ddif, phi_i, phi_end):

    mean = mean_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)
    std = variance_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)**0.5
    skewness = skewness_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)
    kurtosis = kurtosis_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)

    def edgeworth_function(N):
        return edgeworth_pdf(N, mean, std, skewness, kurtosis)

    return edgeworth_function
