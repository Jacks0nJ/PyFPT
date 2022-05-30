'''
Diffusion of Slow-Roll Inflation
---------------------------------
This module returns a function defining the diffusion in slow roll inflation.
To be used in numerics module, the function depends both on ``phi`` and ``N``.
'''


def slow_roll_diffusion(potential, potential_dif, planck_mass=1):
    """Returns the slow-roll diffusion as a function.

    Parameters
    ----------
    potential : function
        The potential of the slow-roll inflation simulated.
    potential_dif : function
        The first derivative of the potential of the slow-roll inflation
        simulated.
    planck_mass : scalar, optional
        The Planck mass used in the calculations. The standard procedure is to
        set it to 1. The default is 1.
    Returns
    -------
    drift_func : function
        A function dependent on ``(phi, N)`` which returns the slow-roll
        diffusion

    """
    def diffusion_func(phi, N):
        pi = 3.141592653589793
        hubble = (potential(phi)/3)**0.5
        return hubble/(2*pi)

    return diffusion_func
