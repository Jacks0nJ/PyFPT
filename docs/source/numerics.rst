.. _numerics:

Numerics
========

This page gives an overview of the functions which both run the importance sampling simulations and perform the data analysis.

The simulations are run using :ref:`Importance Sampling Simulation<is_simulation_label>`, and is the most import function. This function acts as a interface between Python and background Cython code, as well as running the data analysis to create to create the returned probability density. This is the only function which the user needs to interact with directly: the others are used by :ref:`Importance Sampling Simulation<is_simulation_label>`. 


.. _is_simulation_label:
.. automodule:: numerics.is_simulation
    :members:

----


.. automodule:: numerics.data_points_pdf
    :members:

----


.. automodule:: numerics.re_processing
    :members:
    
----

.. automodule:: numerics.histogram_normalisation
    :members:

----


.. automodule:: numerics.data_in_histogram_bins
    :members:

----

.. automodule:: numerics.histogram_data_truncation
    :members:

----


.. automodule:: numerics.jackknife_errors
    :members:
      
----

.. automodule:: numerics.save_data_to_file
    :members:

----

.. automodule:: numerics.multi_processing_error
    :members:
      
----

.. automodule:: numerics.log_normal_error
    :members:
      
----

.. automodule:: numerics.log_normal_height
    :members:
 
----

.. automodule:: numerics.log_normal_mean
    :members:
    
----

.. automodule:: numerics.lognormality_check
    :members:

