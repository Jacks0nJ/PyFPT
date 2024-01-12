'''
Importance Sampling Simulation 2-Dimensions
-------------------------------------------
This is an extension of the main module of PyFPT,
:ref:`Importance Sampling Simulation 1-Dimension<is_simulation_label>`, to
2 dimensions. It runs the simulations, post processes and exports the
data ready for plotting for a 2-dimensional FPT problem. As this problem is
more complex than the 1D case, much more of the functionality of PyFPT
must be done by the user.Therefore, the required inputs are significantly
different, e.g. the importance sampling is incorpered directly into ``update``.

If a reflection is required, this must be defined within the ``update``
function.
'''

from timeit import default_timer as timer
import multiprocessing as mp
from multiprocessing import Process, Queue

import numpy as np

from .multi_processing_error import multi_processing_error
from .histogram_data_truncation import histogram_data_truncation
from .save_data_to_file import save_data_to_file
from .data_points_pdf import data_points_pdf

from .importance_sampling_cython import\
    importance_sampling_simulations_2dim


def is_simulation_2dim(update, x_in, y_in, time_step, num_runs, end_cond,
                       bins=50, min_bin_size=400, num_sub_samples=20,
                       estimator='lognormal', save_data=False, t_in=0.,
                       t_f=100., display=True):
    """Executes the simulation runs by propagating position varaible ``x`` and
    ``y`` till the end condition is met ``num_runs`` times, then returns the
    histogram bin centres, heights and errors.

    Parameters
    ----------
    update : function
        Updates both the propagated variable ``x`` and the importance sampling
        variable ``A`` at each step. Must take both (array-type) ``x``, (float)
        ``t``, (float) ``dt`` and (array-type) ``dW`` as arguments in the
        format ``(x, t, dt, dW)``. Here ``dW`` is Weiner step, i.e. normally
        distributed random numbers of zero mean and variance ``dt``.
    x_in : float
        The initial position value as an array.
    y_in : float
        The initial position value as an array.
    time_step : float
        The time step. This should be at least smaller than the standard
        deviation of the FPTs.
    num_runs : int
        The number of simulation runs.
    end_cond : float or function
        If float, the simulation run ends when the first element in propagated
        variable ``x`` crosses ``end_cond``. If a function, which must take
        arguments ``(x, y, t)``, the run ends when it returns a value of ``1``
        as an integer and continues if a value of ``0`` is returned. A value of
        ``-1`` means the simulation is close to the end surface and smaller
        time steps will be used until ``0`` is returned or it ends with ``1``.
        This is to prevent the simulation run over-stepping the end surface
        and creating a small systematic error. This functionality is not
        available for the simpler float end surface.
    min_bin_size : int, optional
        The minimum number of runs per bin included in the data analysis.
        If a bin has less than this number, it is truncated. Defaults to 400.
    estimator : string, optional
        The estimator used to reconstruct the target distribution probability
        density from the importance sample. If ``'lognormal'``, it assumes the
        weights in each bin follow a lognormal distribution. If ``'naive'``, no
        assumption is made but more runs are required for convergence.
    num_sub_samples : int, optional
        The number of subsamples used in jackknife estimation of the errors
        used for the ``'naive'`` estimator. Defaults to 20 when ``estimator``
        is ``'naive'``.
    Save_data : bool, optional
        If ``True``, the first-passage times and the associated weights for
        each run is saved to a file.
    t_in : float, optional
        The initial time value of simulation Defaults to 0.
    t_f : float, optional
        The maximum FPT allowed per run. If this is exceeded, the
        simulation run ends and returns ``t_f``, which can then be
        truncated. Defaults to 100.
    display : bool, optional
        If True, p-value plots of both the real data, and the theoretical
        expectation if the underlying distribution is truly lognormal, are
        displayed using ``fpt.numerics.lognormality_check`` if a p-value is
        below the specified threshold.
    Returns
    -------
    bin_centres : np.ndarray
        The centres of the histogram bins.
    heights : np.ndarray
        The heights of the normalised histogram bars.
    errors : np.ndarray
        The errors in estimating the heights.
    """
    x_in = float(x_in)
    y_in = float(y_in)
    t_in = float(t_in)
    t_f = float(t_f)
    # Checking drift and diffusion are of the correct format
    if callable(update) is False:
        raise ValueError('Provided update is not a function')

    if isinstance(x_in, float) is not True:
        raise ValueError('x_in is not a float or int.')

    if isinstance(y_in, float) is not True:
        raise ValueError('y_in is not a float or int.')

    if isinstance(time_step, float) is not True:
        raise ValueError('time_step is not a float')

    # Check the user has provided a estimator
    if estimator != 'lognormal' and estimator != 'naive':
        print('Invalid estimator argument, defaulting to naive method')
        estimator = 'naive'

    # The number of sims per core, so the total is correct
    if num_runs % mp.cpu_count() == 0:
        num_runs_per_core = int(num_runs/mp.cpu_count())
    else:
        print("Provided run number can not be evenly divided between cores.")
        num_runs_per_core = int((num_runs - (num_runs % mp.cpu_count())) /
                                mp.cpu_count())
        num_runs = num_runs_per_core*mp.cpu_count()
        print("Using " + str(num_runs-num_runs % mp.cpu_count()) + " instead")

    # Time how long the simulation runs take
    start = timer()

    # Using multiprocessing
    def multi_processing_func(x_in, y_in, t_in, t_f, time_step, num_runs,
                              end_cond, update, queue_efolds, queue_ws):
        results =\
            importance_sampling_simulations_2dim(x_in, y_in, t_in, t_f,
                                                 time_step, num_runs, end_cond,
                                                 update)
        fpt_values = np.array(results[0][:])
        ws = np.array(results[1][:])
        queue_efolds.put(fpt_values)
        queue_ws.put(ws)

    queue_efolds = Queue()
    queue_ws = Queue()
    cores = int(mp.cpu_count()/1)

    print('Number of cores used: '+str(cores))
    processes = [Process(target=multi_processing_func,
                         args=(x_in, y_in, t_in, t_f, time_step, num_runs,
                               end_cond, update, queue_efolds, queue_ws))
                 for i in range(cores)]

    for p in processes:
        p.start()

    # More efficient to work with numpy arrays
    fpt_array = np.array([queue_efolds.get() for p in processes])
    ws_array = np.array([queue_ws.get() for p in processes])

    end = timer()
    print(f'The simulations took: {end - start} seconds')

    # Combine into columns into 1
    fpt_values = fpt_array.flatten()
    w_values = ws_array.flatten()

    # Sort in order of increasing Ns
    sort_idx = np.argsort(fpt_values)
    fpt_values = fpt_values[sort_idx]
    w_values = w_values[sort_idx]

    # Checking if importance sampling was not done, set to none
    if all(w_values == 1.):
        # Simply nullify if all 1, meaning no importance sampling achieved
        w_values = None
        # Truncating any data which did not reach x_end
        fpt_values =\
            histogram_data_truncation(fpt_values, t_f,
                                      num_sub_samples=num_sub_samples)
    else:
        # Checking if multipprocessing error occured, by looking at correlation
        _ = multi_processing_error(fpt_values, w_values)

        # Truncating any data which did not reach x_end
        fpt_values, w_values =\
            histogram_data_truncation(fpt_values, t_f, weights=w_values,
                                      num_sub_samples=num_sub_samples)
    # Saving the data
    if save_data is True:
        file_label = '_2dims'

        save_data_to_file(fpt_values, x_in, num_runs, weights=w_values,
                          extra_label=file_label)

    # Now analysisng the data to creating the histogram/PDF data
    bin_centres, heights, errors, num_runs_used, bin_edges_untruncated =\
        data_points_pdf(fpt_values, weights=w_values, estimator=estimator,
                        bins=bins, min_bin_size=min_bin_size,
                        num_sub_samples=num_sub_samples, display=display)
    # Return data as lists
    return bin_centres, heights, errors
