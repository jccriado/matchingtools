Integration
===========

.. currentmodule:: matchingtools.integration

.. note:: This section assumes that the classes and functions
   from :mod:`matchingtools.integration` that it uses are in the namespace.
   To import all the classes and functions that appear here do::

      from matchingtools.integration import (
          RealScalar, ComplexScalar, RealVector, ComplexVector,
	  VectorLikeFermion, MajoranaFermion, integrate)
	  
To integrate some heavy fields out of a previously constructed Lagrangian
the heavy fields should be specified first. The heavy fields should be objects
of any of the following classes:

* :class:`RealScalar`
* :class:`ComplexScalar`
* :class:`RealVector`
* :class:`ComplexVector`
* :class:`VectorLikeFermion`
* :class:`MajoranaFermion`

Create a heavy field using the constructors of these classes.

.. code:: python

  heavy_field = HeavyFieldClass("field_name", ...)

Then collect the heavy fields in a list and use the function
:func:`integrate` to perform the integration::

   heavy_fields = [heavy_field_1, heavy_field_2, ...]
   eff_lag = integrate(heavy_fields, int_lag, max_dim=...)
