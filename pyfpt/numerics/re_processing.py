'''
Re-Processing
---------------
This module runs the same post-processing of data as the main simulation
modules. It is intended to be used if the simulation is run directly or to
re-analyse saved raw data.
'''

import numpy as np

from .multi_processing_error import multi_processing_error
from .histogram_data_truncation import histogram_data_truncation
from .data_points_pdf import data_points_pdf


def re_processing(data, weights=None, bins=50, min_bin_size=400,
                  num_sub_samples=20, estimator='lognormal', t_f=100,
                  display=True):
    """Runs the post-processing on the provided data and returns the histogram
    bin centres, heights and errors.

    Parameters
    ----------
    data : list or numpy.ndarray
        Input first-passage time data.
    weights : list or numpy.ndarray, optional
        Associated weights to the first-passage time data. Must be a one-to-one
        correspondence between them. Defaults to ``None``.
    bins : int or sequence, optional
        If bins is an integer, it defines the number equal width bins for the
        first-passage times. If bins is a list or numpy array, it defines the
        bin edges, including the left edge of the first bin and the right edge
        of the last bin. The widths can vary. Defaults to 50 evenly spaced
        bins.
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
    bin_centres : list
        The centres of the histogram bins.
    heights : list
        The heights of the normalised histogram bars.
    errors : list
        The errors in estimating the heights.
    """
    # If lists provided, change to numpy
    if isinstance(data, list):
        data = np.array(data)

    if isinstance(weights, list):
        weights = np.array(weights)

    if isinstance(weights, np.ndarray):  # Only true is weights given
        # Checking if multipprocessing error occured, by looking at correlation
        _ = multi_processing_error(data, weights)

        # Truncating any data which did not reach x_end
        data, weights =\
            histogram_data_truncation(data, t_f, weights=weights,
                                      num_sub_samples=num_sub_samples)
    # If unweighted data
    else:
        # Truncating any data which did not reach x_end
        data =\
            histogram_data_truncation(data, t_f,
                                      num_sub_samples=num_sub_samples)

    # Now analysisng the data to creating the histogram/PDF data
    bin_centres, heights, errors, num_runs_used, bin_edges_untruncated =\
        data_points_pdf(data, weights=weights, estimator=estimator, bins=bins,
                        min_bin_size=min_bin_size,
                        num_sub_samples=num_sub_samples, display=display)
    # Return data as lists
    return bin_centres.tolist(), heights.tolist(), errors.tolist()
