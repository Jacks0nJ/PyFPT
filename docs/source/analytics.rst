.. _analytics:

Analytical Functions
====================

This page gives an overview of the different analytical functions available in PyFPT. 

All but one of them are based on calculating the central moments of the FPTs using methods of `Vennin--Starobinsky 2015`_ in the low diffusion limit. The central moments not only give the mean, variance, skewness and kurtosis of the FPTs, but can also be used to add corrections to the Gaussian approximation for the probability density.

For diffusion dominated quadratic inflation, there are also :ref:`analytical expectations<quadratic_inflation_label>` for the probability density. Quadratic inflation is currently the only inflationary model included.


.. _Vennin--Starobinsky 2015: https://arxiv.org/abs/1506.04732

.. automodule:: analytics.classicality_criterion
    :members:
    
----

.. automodule:: analytics.mean_N
    :members:

----

.. automodule:: analytics.variance_N
    :members:
    
----

.. automodule:: analytics.skewness_N
    :members:
 
----

.. automodule:: analytics.kurtosis_N
    :members:
    
----

.. automodule:: analytics.third_central_moment_N
    :members:

----

.. automodule:: analytics.fourth_central_moment_N
    :members:

----

.. automodule:: analytics.reduced_potential
    :members:

----
  
.. automodule:: analytics.reduced_potential_diff
    :members:

----
  
.. automodule:: analytics.reduced_potential_ddiff
    :members:
    
----

.. automodule:: analytics.gaussian_pdf
    :members:
    
----

.. automodule:: analytics.edgeworth_pdf
    :members:

----

.. automodule:: analytics.gaussian_deviation
    :members:

----

.. _quadratic_inflation_label:
.. automodule:: analytics.quadratic_inflation_large_mass_pdf
    :members:
    
----

.. automodule:: analytics.quadratic_inflation_near_tail_pdf
    :members:

