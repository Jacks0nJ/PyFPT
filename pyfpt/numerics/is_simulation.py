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


def is_simulation(V, V_dif, V_ddif, phi_i, phi_end, num_runs, bias_amp,
                  bins=50, dN=None, min_bin_size=400, num_sub_samples=20,
                  estimator='lognormal', save_data=False, N_f=100,
                  phi_UV=None):
    """Executes the simulation runs, then returns the histogram bin centres,
    heights and errors.

    Parameters
    ----------
    V : function
        The potential.
    V_dif : function
        The potential's first derivative.
    V_ddif : function
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
    dN : float or int, optional
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
    N_f : float, optional
        The maxiumum number of e-folds allowed per run. If this is exceded, the
        simulation run ends and returns ``N_f``, which can then be truncated.
        Defaults to 100 e-folds.
    phi_UV : float, optional
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
    # If no argument for dN is given, using the classical std to define it
    if dN is None:
        if isinstance(bins, int) is True:
            std = variance_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)**0.5
            dN = std/bins
        elif isinstance(bins, int) is False:
            std = variance_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)
            num_bins = len(bins)-1
            dN = std/(num_bins)
    elif isinstance(dN, float) is not True and isinstance(dN, int) is not True:
        raise ValueError('dN is not a number')

    # Check the user has provided a estimator
    if estimator != 'lognormal' and estimator != 'naive':
        print('Invalid estimator argument, defaulting to naive method')
        estimator = 'naive'

    # If no phi_UV argument is provided, default to infinie boundary
    if phi_UV is None:
        phi_UV = 10000*phi_i
    elif isinstance(phi_UV, float) is False:
        if isinstance(phi_UV, int) is True:
            if isinstance(phi_UV, bool) is True:
                raise ValueError('phi_UV is not a number')
            else:
                pass
        else:
            raise ValueError('phi_UV is not a number')
    elif np.abs(phi_UV) < np.abs(phi_i):
        raise ValueError('phi_UV is smaller than phi_i')

    if bias_amp == 0:
        estimator = 'naive'

    # The number of sims per core, so the total is correct
    num_runs_per_core = int(num_runs/mp.cpu_count())
    # Time how long the simulation runs take
    start = timer()

    # Using multiprocessing
    def multi_processing_func(phi_i, phi_UV, phi_end, N_i, N_f, dN, bias_amp,
                              num_runs, queue_Ns, queue_ws, queue_refs):
        results =\
            importance_sampling_simulations(phi_i, phi_UV, phi_end, N_i, N_f,
                                            dN, bias_amp, num_runs, V, V_dif,
                                            V_ddif, bias_type='diffusion',
                                            count_refs=False)
        Ns = np.array(results[0][:])
        ws = np.array(results[1][:])
        queue_Ns.put(Ns)
        queue_ws.put(ws)

    queue_Ns = Queue()
    queue_ws = Queue()
    queue_refs = Queue()
    cores = int(mp.cpu_count()/1)

    print('Number of cores used: '+str(cores))
    processes = [Process(target=multi_processing_func,
                         args=(phi_i, phi_UV,  phi_end, 0.0, N_f, dN, bias_amp,
                               num_runs_per_core, queue_Ns, queue_ws,
                               queue_refs)) for i in range(cores)]

    for p in processes:
        p.start()

    # More efficient to work with numpy arrays
    Ns_array = np.array([queue_Ns.get() for p in processes])
    ws_array = np.array([queue_ws.get() for p in processes])

    end = timer()
    print(f'The simulations took: {end - start} seconds')

    # Combine into columns into 1
    sim_N_dist = Ns_array.flatten()
    w_values = ws_array.flatten()

    # Sort in order of increasing Ns
    sort_idx = np.argsort(sim_N_dist)
    sim_N_dist = sim_N_dist[sort_idx]
    w_values = w_values[sort_idx]

    # Checking if multipprocessing error occured, by looking at correlation
    multi_processing_error(sim_N_dist, w_values)

    # Truncating any data which did not reach phi_end
    sim_N_dist, w_values =\
        histogram_data_truncation(sim_N_dist, N_f, weights=w_values,
                                  num_sub_samples=num_sub_samples)
    # Saving the data
    if save_data is True:
        save_data_to_file(sim_N_dist, w_values, phi_i, num_runs, bias_amp)

    # Now analysisng the data to creating the histogram/PDF data
    bin_centres, heights, errors, num_runs_used, bin_edges_untruncated =\
        data_points_pdf(sim_N_dist, w_values, estimator, bins=bins,
                        min_bin_size=min_bin_size,
                        num_sub_samples=num_sub_samples)
    # Return data as lists
    return bin_centres.tolist(), heights.tolist(), errors.tolist()


# Equation 3.35 in Vennin 2015
def variance_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end):
    PI = np.pi
    M_PL = 1

    def v_func(phi):
        return V(phi)/(24*PI**2)

    def V_dif_func(phi):
        return V_dif(phi)/(24*PI**2)

    def V_ddif_func(phi):
        return V_ddif(phi)/(24*PI**2)

    def integrand_calculator(phi):
        # Pre calculating values
        v = v_func(phi)
        V_dif = V_dif_func(phi)
        V_ddif = V_ddif_func(phi)
        non_classical = 6*v-np.divide(5*(v**2)*V_ddif, V_dif**2)
        constant_factor = 2/(M_PL**4)

        integrand = constant_factor*np.divide(v**4, V_dif**3)*(1+non_classical)
        return integrand
    var_N, er = integrate.quad(integrand_calculator, phi_end, phi_i)
    return var_N
