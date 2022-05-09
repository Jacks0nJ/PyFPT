'''
Importance Sampling Simulation
------------------------------
This is the main module of the PyFPT code, as it runs the simulations, post
processes and exports the data ready for plotting.
'''

from timeit import default_timer as timer
import multiprocessing as mp
from multiprocessing import Process, Queue
from scipy import integrate

import numpy as np

from .multi_processing_error import multi_processing_error
from .histogram_data_truncation import histogram_data_truncation
from .save_data_to_file import save_data_to_file
from .data_points_pdf import data_points_pdf

from .importance_sampling_sr_cython import\
    importance_sampling_simulations


def is_simulation(potential, potential_dif, potential_ddif, phi_i, phi_end,
                  num_runs, bias_amp, bins=50, delta_efolds=None,
                  min_bin_size=400, num_sub_samples=20, estimator='lognormal',
                  save_data=False, efolds_f=100, phi_uv=None):
    """Executes the simulation runs, then returns the histogram bin centres,
    heights and errors.

    Parameters
    ----------
    potential : function
        The potential.
    potential_dif : function
        The potential's first derivative.
    potential_ddif : function
        The potential second derivative.
    phi_i : float
        The initial scalar field value.
    phi_end : float
        The end scalar field value.
    num_runs : int
        The number of simulation runs.
    bias_amp : float
        The coefficent of the diffusion used define the bias. (In later
        versions this can also be a function).
    bins : int or sequence, optional
        If bins is an integer, it defines the number equal width bins for the
        first-passage times. If bins is a list or numpy array, it defines the
        bin edges, including the left edge of the first bin and the right edge
        of the last bin. The widths can vary. Defaults to 50 evenly spaced
        bins.
    delta_efolds : float or int, optional
        The step size in e-folds N. This should be a small enough to accurately
        bins the runs. Defaults to the standard deviation devided by the number
        of bins.
    min_bin_size : int, optional
        The minimum number of runs per bin to included in the data analysis.
        If a bin has less than this number, it is truncated. Defaults to 400.
    estimator : string, optional
        The estimator used to reconstruct the target distribution probability
        density from the importance sample. If ``'lognormal'``, it assumes the
        weights in each bin follow a lognomral distribution. If ``'naive'``, no
        assumption is made but more runs are required for convergance.
    num_sub_samples : int, optional
        The number of subsamples used in jackknife estimation of the errors
        used for the ``'naive'`` estimator. Defaults to 20 when ``estimator``
        is ``'naive'``.
    Save_data : bool, optional
        If ``True``, the first-passage times and the associated weights for
        each run is saved to a file.
    efolds_f : float, optional
        The maxiumum number of e-folds allowed per run. If this is exceded, the
        simulation run ends and returns ``efolds_f``, which can then be
        truncated. Defaults to 100 e-folds.
    phi_uv : float, optional
        The value of the reflective boundary. Must have a magntiude greater
        than the magnitude of ``phi_i``. Defaults to no reflective boundary.
    Returns
    -------
    bin_centres : list
        The centres of the histogram bins.
    heights : list
        The heights of the normalised histogram bars.
    errors : list
        The errors in estimating the heights.
    """
    # If no argument for delta_efolds is given, using the classical std to
    # define it
    if delta_efolds is None:
        if isinstance(bins, int) is True:
            std =\
                variance_efolds(potential, potential_dif, potential_ddif,
                                phi_i, phi_end)**0.5
            delta_efolds = std/bins
        elif isinstance(bins, int) is False:
            std =\
                variance_efolds(potential, potential_dif, potential_ddif,
                                phi_i, phi_end)
            num_bins = len(bins)-1
            delta_efolds = std/(num_bins)
    elif isinstance(delta_efolds, float) is not True\
            and isinstance(delta_efolds, int) is not True:
        raise ValueError('delta_efolds is not a number')

    # Check the user has provided a estimator
    if estimator != 'lognormal' and estimator != 'naive':
        print('Invalid estimator argument, defaulting to naive method')
        estimator = 'naive'

    # If no phi_uv argument is provided, default to infinie boundary
    if phi_uv is None:
        phi_uv = 10000*phi_i
    elif isinstance(phi_uv, float) is False:
        if isinstance(phi_uv, int) is True:
            if isinstance(phi_uv, bool) is True:
                raise ValueError('phi_uv is not a number')
            else:
                pass
        else:
            raise ValueError('phi_uv is not a number')
    elif np.abs(phi_uv) < np.abs(phi_i):
        raise ValueError('phi_uv is smaller than phi_i')

    if bias_amp == 0:
        estimator = 'naive'

    # The number of sims per core, so the total is correct
    num_runs_per_core = int(num_runs/mp.cpu_count())
    # Time how long the simulation runs take
    start = timer()

    # Using multiprocessing
    def multi_processing_func(phi_i, phi_uv, phi_end, efolds_i, efolds_f,
                              delta_efolds, bias_amp, num_runs, queue_efolds,
                              queue_ws, queue_refs):
        results =\
            importance_sampling_simulations(phi_i, phi_uv, phi_end, efolds_i,
                                            efolds_f, delta_efolds, bias_amp,
                                            num_runs, potential, potential_dif,
                                            potential_ddif,
                                            bias_type='diffusion',
                                            count_refs=False)
        efold_values = np.array(results[0][:])
        ws = np.array(results[1][:])
        queue_efolds.put(efold_values)
        queue_ws.put(ws)

    queue_efolds = Queue()
    queue_ws = Queue()
    queue_refs = Queue()
    cores = int(mp.cpu_count()/1)

    print('Number of cores used: '+str(cores))
    processes = [Process(target=multi_processing_func,
                         args=(phi_i, phi_uv,  phi_end, 0.0, efolds_f,
                               delta_efolds, bias_amp, num_runs_per_core,
                               queue_efolds, queue_ws, queue_refs))
                 for i in range(cores)]

    for p in processes:
        p.start()

    # More efficient to work with numpy arrays
    efolds_array = np.array([queue_efolds.get() for p in processes])
    ws_array = np.array([queue_ws.get() for p in processes])

    end = timer()
    print(f'The simulations took: {end - start} seconds')

    # Combine into columns into 1
    efold_values = efolds_array.flatten()
    w_values = ws_array.flatten()

    # Sort in order of increasing Ns
    sort_idx = np.argsort(efold_values)
    efold_values = efold_values[sort_idx]
    w_values = w_values[sort_idx]

    # Checking if multipprocessing error occured, by looking at correlation
    multi_processing_error(efold_values, w_values)

    # Truncating any data which did not reach phi_end
    efold_values, w_values =\
        histogram_data_truncation(efold_values, efolds_f, weights=w_values,
                                  num_sub_samples=num_sub_samples)
    # Saving the data
    if save_data is True:
        save_data_to_file(efold_values, w_values, phi_i, num_runs, bias_amp)

    # Now analysisng the data to creating the histogram/PDF data
    bin_centres, heights, errors, num_runs_used, bin_edges_untruncated =\
        data_points_pdf(efold_values, w_values, estimator, bins=bins,
                        min_bin_size=min_bin_size,
                        num_sub_samples=num_sub_samples)
    # Return data as lists
    return bin_centres.tolist(), heights.tolist(), errors.tolist()


# Equation 3.35 in Vennin 2015
def variance_efolds(potential, potential_dif, potential_ddif, phi_i, phi_end):
    pi = np.pi
    planck_mass = 1

    def v_func(phi):
        return potential(phi)/(24*pi**2)

    def v_dif_func(phi):
        return potential_dif(phi)/(24*pi**2)

    def v_ddif_func(phi):
        return potential_ddif(phi)/(24*pi**2)

    def integrand_calculator(phi):
        # Pre calculating values
        v = v_func(phi)
        v_dif = v_dif_func(phi)
        v_ddif = v_ddif_func(phi)
        non_classical = 6*v-np.divide(5*(v**2)*v_ddif, v_dif**2)
        constant_factor = 2/(planck_mass**4)

        integrand = constant_factor*np.divide(v**4, v_dif**3)*(1+non_classical)
        return integrand
    var_efolds, er = integrate.quad(integrand_calculator, phi_end, phi_i)
    return var_efolds
