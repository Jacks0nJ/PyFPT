Welcome to PyFPT's documentation!
==============================================

This is the documentation for a Python/Cython package to run first-passage time (FPT) simulations using importance sampling.

This package will let you numerically investigate the tail of the probability density for first passage times, for a general 1D Langevin equation. See the :ref:`guide<getting_started>` section for how to install PyFPT, as well as a how-to on running your first simulation.

The tail of the probability density is investiated using the method of `importance sampling`_, where a bias increases the probability of large FPTs, resulting in a sample distribution, which are then weighted to reproduce the rare events of the target distribution. The :ref:`Numerics<numerics>` module both runs the simulations and performs the data analysis.

This package was orginally developed to solve FPT problems in stochastic slow-roll inflation, and as such it also comes with functionality to compare the numerical results with analytical expectations, see :ref:`Analytical Functions<analytics>`.  

.. _importance sampling: https://arxiv.org/abs/nucl-th/9809075
.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   getting_started
   analytics
   numerics



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
