'''
Importance Sampling Simulation
------------------------------
This is the main module of the PyFPT code, as it runs the simulations, post
processes and exports the data ready for plotting.
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
    importance_sampling_simulations


def is_simulation(drift, diffusion, x_in, x_end,
                  num_runs, bias, time_step, bins=50, min_bin_size=400,
                  num_sub_samples=20, estimator='lognormal',
                  save_data=False, t_in=0., t_f=100, x_r=None, display=True):
    """Executes the simulation runs, then returns the histogram bin centres,
    heights and errors.

    Parameters
    ----------
    drift : function
        The drift term of the simulated Langevin equation. Must take both x and
        t as arguments in the format ``(x, t)``.
    diffusion : function
        The diffusion term of the simulated Langevin equation. Must take both
        x and t as arguments in the format ``(x, t)``.
    x_in : float
        The initial position value.
    x_end : float
        The end position value, i.e. the threshold which defines the FPT
        problem.
    num_runs : int
        The number of simulation runs.
    bias : scalar or function
        The bias used in the simulated Langevin equation to achieve importance
        sampling

        If a scalar (float or int), this the bias amplitude, i.e. a coefficient
        which multiplies the diffusion to define the bias.

        If a function, this simply defines the bias used. Must take arguments
        for both position and time in the format ``(x, t)``.
    bins : int or sequence, optional
        If bins is an integer, it defines the number equal width bins for the
        first-passage times. If bins is a list or numpy array, it defines the
        bin edges, including the left edge of the first bin and the right edge
        of the last bin. The widths can vary. Defaults to 50 evenly spaced
        bins.
    time_step : float or int, optional
        The time step. This should be at least smaller than the standard
        deviation of the FPTs.
    min_bin_size : int, optional
        The minimum number of runs per bin included in the data analysis.
        If a bin has less than this number, it is truncated. Defaults to 400.
    estimator : string, optional
        The estimator used to reconstruct the target distribution probability
        density from the importance sample. If ``'lognormal'``, it assumes the
        weights in each bin follow a lognomral distribution. If ``'naive'``, no
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
    x_r : float, optional
        The value of the reflective boundary. Must be compatible with the x_in
        and x_end chosen. Defaults to unreachable value, effectively no
        boundary.
    display : bool, optional
        If True, p-value plots of both the real data, and the theoretical
        expectation if the underlying distribution is truly lognormal, are
        displayed using ``fpt.numerics.lognormality_check`` if a p-value is
        below the specified threshold.
    Returns
    -------
    bin_centres : list
        The centres of the histogram bins.
    heights : list
        The heights of the normalised histogram bars.
    errors : list
        The errors in estimating the heights.
    """
    # Checking drift and diffusion are of the correct format
    if callable(drift) is True:
        if isinstance(drift(x_in, t_in), float) is True:
            pass
        else:
            ValueError('Provided drift is not the format (x, t)')
    else:
        ValueError('Provided drift is not a function')
    if callable(diffusion) is True:
        if isinstance(diffusion(x_in, t_in), float) is True:
            pass
        else:
            ValueError('Provided diffusion is not the format (x, t)')
    else:
        ValueError('Provided diffusion is not a function')

    # Make sure provided values are floats for Cython
    if isinstance(x_in, int) is True:
        x_in = 1.0*x_in
    if isinstance(x_end, int) is True:
        x_end = 1.0*x_end
    # Checking bias is of correct form
    if isinstance(bias, float) is True or isinstance(bias, float) is True:
        # If the bias argument is a scalar, use diffusion based bias
        bias_type = 'diffusion'
        if bias == 0:
            estimator = 'naive'
            print('As direct simulation, defaulting to naive estimator')
    elif callable(bias):
        # If a function is provided, check it is of the correct form
        if isinstance(bias(x_in, t_in), float) is True:
            bias_type = 'custom'
        else:
            ValueError('bias function must be of the form bias(x, t)')
    else:
        ValueError('Provided bias is not a number or function')

    if isinstance(time_step, float) is not True\
            and isinstance(time_step, int) is not True:
        raise ValueError('time_step is not a number')

    # Check the user has provided a estimator
    if estimator != 'lognormal' and estimator != 'naive':
        print('Invalid estimator argument, defaulting to naive method')
        estimator = 'naive'

    # If no x_r argument is provided, default to infinite boundary
    if x_r is None:
        # Set the reflective surface at an arbitrarily large value in the
        # opposite direction to propagation
        x_r = 10000*(x_in-x_end)
    elif isinstance(x_r, float) is False:
        if isinstance(x_r, int) is True:
            if isinstance(x_r, bool) is True:
                raise ValueError('x_r is not a number')
            else:
                pass
        else:
            raise ValueError('x_r is not a number')
    elif (x_r-x_in)*(x_in-x_end) < 0:
        raise ValueError('End and relfective surfaces not compatible with' +
                         ' initial value.')

    # The number of sims per core, so the total is correct
    num_runs_per_core = int(num_runs/mp.cpu_count())
    # Time how long the simulation runs take
    start = timer()

    # Using multiprocessing
    def multi_processing_func(x_in, x_r, x_end, t_in, t_f,
                              time_step, bias, num_runs, queue_efolds,
                              queue_ws, queue_refs):
        results =\
            importance_sampling_simulations(x_in, x_r, x_end, t_in,
                                            t_f, time_step, bias,
                                            num_runs, drift, diffusion,
                                            bias_type=bias_type,
                                            count_refs=False)
        fpt_values = np.array(results[0][:])
        ws = np.array(results[1][:])
        queue_efolds.put(fpt_values)
        queue_ws.put(ws)

    queue_efolds = Queue()
    queue_ws = Queue()
    queue_refs = Queue()
    cores = int(mp.cpu_count()/1)

    print('Number of cores used: '+str(cores))
    processes = [Process(target=multi_processing_func,
                         args=(x_in, x_r,  x_end, t_in, t_f,
                               time_step, bias, num_runs_per_core,
                               queue_efolds, queue_ws, queue_refs))
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

    # Checking if multipprocessing error occured, by looking at correlation
    _ = multi_processing_error(fpt_values, w_values)

    # Truncating any data which did not reach x_end
    fpt_values, w_values =\
        histogram_data_truncation(fpt_values, t_f, weights=w_values,
                                  num_sub_samples=num_sub_samples)
    # Saving the data
    if save_data is True:
        if bias_type == 'diffusion':
            save_data_to_file(fpt_values, w_values, x_in, num_runs, bias)
        else:
            # Label the file differently if custom bias is used.
            save_data_to_file(fpt_values, w_values, x_in, num_runs,
                              bias(x_in, 0), extra_label='_custom_bias')

    # Now analysisng the data to creating the histogram/PDF data
    bin_centres, heights, errors, num_runs_used, bin_edges_untruncated =\
        data_points_pdf(fpt_values, w_values, estimator, bins=bins,
                        min_bin_size=min_bin_size,
                        num_sub_samples=num_sub_samples, display=display)
    # Return data as lists
    return bin_centres.tolist(), heights.tolist(), errors.tolist()
