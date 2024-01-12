Welcome to PyFPT's documentation!
==============================================

.. image:: /images/PyFPT_logo.png

This is the documentation for a Python/Cython package to run first-passage time (FPT) simulations using importance sampling. A FPT problem is about finding the time taken to cross some threshold during a stochastic process.

This package will let you numerically investigate the tail of the probability density for first passage times, for both general 1D and n-D Langevin equations. See the :ref:`guide<getting_started>` section for how to install PyFPT, as well as a how-to on running your first simulation. The n-D case is less efficient and requires more prior knowledge from the user.

The tail of the probability density is investigated using the method of `importance sampling`_, where a bias increases the probability of large FPTs, resulting in a sample distribution, which are then weighted to reproduce the rare events of the target distribution. The :ref:`Numerics<numerics>` module both runs the simulations and performs the data analysis.

This package was originally developed to solve FPT problems in `stochastic inflation`_, and as such it also comes with functionality to compare the numerical results with analytical expectations, see :ref:`Analytical Functions<analytics>`.  

.. _importance sampling: https://arxiv.org/abs/nucl-th/9809075
.. _stochastic inflation: https://arxiv.org/abs/1506.04732
.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   getting_started
   numerics
   analytics



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
