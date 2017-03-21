Integration
===========

.. currentmodule:: efttools.integration

To integrate some heavy fields out of a previously constructed lagrangian
the heavy fields should be specified first. The heavy fields should be objects
of any of the following classes:

* :class:`RealScalar`
* :class:`ComplexScalar`
* :class:`RealVector`
* :class:`ComplexVector`
* :class:`VectorLikeFermion`
* :class:`MajoranaFermion`

Create a heavy field using the constructors of these classes::

  heavy_field = HeavyFieldClass("field_name", ...)

Then collect the heavy fields in a list and use the function
:func:`integrate` to perform the integration::

   heavy_fields = [heavy_field_1, heavy_field_2, ...]
   eff_lag = integrate(heavy_fields, int_lag, max_dim=...)
