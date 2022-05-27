'''
Drift of Slow-Roll Inflation
---------------------------------
This module returns a function defining the drift in slow roll inflation. To be
used in numerics module, the function depends both on ``phi`` and ``N``.
'''


planck_mass = 1


def slow_roll_drift(potential, potential_dif):
    """Returns the slow-roll drift as a function.

    Parameters
    ----------
    potential : function
        The potential of the slow-roll inflation simulated.
    potential_dif : function
        The first derivative of the potential of the slow-roll inflation
        simulated.
    Returns
    -------
    drift_func : function
        A function dependent on ``(phi, N)`` which returns the slow-roll drift

    """
    def drift_func(phi, N):
        return -potential_dif(phi)/potential(phi)

    return drift_func
