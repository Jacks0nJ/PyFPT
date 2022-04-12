'''
Kurtosis of the Number of e-folds
---------------------------------
This module calculates the skewness of the number of e-folds in low diffusion
limit using equation 3.40 (from `Vennin--Starobinsky 2015`_) for the fourth
central moment and equation 3.33 for the variance.

.. _Vennin--Starobinsky 2015: https://arxiv.org/abs/1506.04732
'''


from .fourth_central_moment_N_sto_limit import\
    fourth_central_moment_N_sto_limit
from .variance_N_sto_limit import variance_N_sto_limit

M_PL = 1


# Using the standard relation between the central moments and the kurtosis.
# Fisher is an optional argument
def kurtosis_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end, Fisher=True):
    """Returns the kurtosis of the number of e-folds.

    Parameters
    ----------
    V : function
        The potential.
    V_dif : function
        The potential's first derivative.
    V_ddif : function
        The potential second derivative.
    phi_i : float
        The initial scalar field value
    phi_end : float
        The end scalar field value.
    Fisher : bool, optional
        If True, Fisher’s definition is used (normal ==> 0.0). If False,
        Pearson’s definition is used (normal ==> 3.0).
    Returns
    -------
    kurtosis_N : float
        the kurtosis of the number of e-folds.

    """
    # The excess kurtosis over the expected Gaussian amount
    fourth_moment = fourth_central_moment_N_sto_limit(V, V_dif, V_ddif,
                                                      phi_i, phi_end)
    var = variance_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)
    if Fisher is False:
        kurtosis_N = fourth_moment/var**2
    else:  # Defaults to Fisher definition
        kurtosis_N = fourth_moment/var**2-3

    return kurtosis_N
