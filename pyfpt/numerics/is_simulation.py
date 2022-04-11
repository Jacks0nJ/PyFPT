'''
Importance Sampling Simulation
---------------------------------
This module calculates the variance of the number of e-folds in low diffusion
limit using equation 3.35 from `Vennin-Starobinsky 2015`_.

.. _Vennin-Starobinsky 2015: https://arxiv.org/abs/1506.04732
'''

from timeit import default_timer as timer
import multiprocessing as mp
from multiprocessing import Process, Queue

import numpy as np

from .multi_processing_error import multi_processing_error
from .histogram_data_truncation import histogram_data_truncation
from .save_data_to_file import save_data_to_file
from .data_points_pdf import data_points_pdf

from ..analytics.variance_N_sto_limit import variance_N_sto_limit

from ..cython_code.importance_sampling_sr_cython12 import\
    many_simulations_importance_sampling


def is_simulation(V, V_dif, V_ddif, phi_i, phi_end, num_sims, bias, bins=50,
                  dN=None, min_bin_size=400, num_sub_samples=20,
                  reconstruction='lognormal', save_data=False, N_f=100,
                  phi_UV=None):
    """Returns the variance of the number of e-folds.

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

    Returns
    -------
    var_N : float
        the variance of the number of e-folds.
s
    """
    # If no argument for dN is given, using the classical std to define it
    if dN is None:
        if isinstance(bins, int) is True:
            std = variance_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)
            dN = std/(3*bins)
        elif isinstance(bins, int) is False:
            std = variance_N_sto_limit(V, V_dif, V_ddif, phi_i, phi_end)
            num_bins = len(bins)-1
            dN = std/(num_bins)
    elif isinstance(dN, float) is not True and isinstance(dN, int) is not True:
        raise ValueError('dN is not a number')

    if reconstruction != 'lognormal' and reconstruction != 'naive':
        print('Invalid reconstruction argument, defaulting to naive method')
        reconstruction = 'naive'

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

    # The number of sims per core, so the total is correct
    num_sims_per_core = int(num_sims/mp.cpu_count())

    start = timer()

    def multi_processing_func(phi_i, phi_UV, phi_end, N_i, N_f, dN, bias,
                              num_sims, queue_Ns, queue_ws, queue_refs):
        results =\
            many_simulations_importance_sampling(phi_i, phi_UV,
                                                 phi_end, N_i, N_f, dN,
                                                 bias, num_sims, V,
                                                 V_dif, V_ddif,
                                                 bias_type='diffusion',
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
                         args=(phi_i, phi_UV,  phi_end, 0.0, N_f, dN, bias,
                               num_sims_per_core, queue_Ns, queue_ws,
                               queue_refs)) for i in range(cores)]

    for p in processes:
        p.start()

    Ns_array = np.array([queue_Ns.get() for p in processes])
    ws_array = np.array([queue_ws.get() for p in processes])
    end = timer()
    print(f'The simulations took: {end - start}')

    # Combine into columns into 1
    sim_N_dist = Ns_array.flatten()
    w_values = ws_array.flatten()

    # Sort in order of increasing Ns
    sort_idx = np.argsort(sim_N_dist)
    sim_N_dist = sim_N_dist[sort_idx]
    w_values = w_values[sort_idx]

    # Checking if multipprocessing error occured, by looking at correlation
    multi_processing_error(sim_N_dist, w_values)
    # Truncating the data
    sim_N_dist, w_values =\
        histogram_data_truncation(sim_N_dist, N_f, weights=w_values,
                                  num_sub_samples=num_sub_samples)
    # Saving the data
    if save_data is True:
        save_data_to_file(sim_N_dist, w_values, phi_i, num_sims, bias=bias)

    # Now analysisng creating the PDF data
    bin_centres, heights, errors, num_sims_used, bin_edges_untruncated =\
        data_points_pdf(sim_N_dist, w_values, num_sub_samples,
                        reconstruction, bins=bins,
                        min_bin_size=min_bin_size, num_sims=num_sims)

    return bin_centres, heights, errors
