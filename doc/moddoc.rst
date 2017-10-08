Modules documentation
*********************

The ``operators`` module
========================

.. automodule:: matchingtools.core

.. autoclass:: Tensor

.. autoclass:: Operator
	       
   .. automethod:: variation

   .. automethod:: replace_first

   .. automethod:: replace_all

   .. automethod:: match_first

   .. automethod:: __eq__

.. autoclass:: OperatorSum

   .. automethod:: variation

   .. automethod:: replace_all

.. autoclass:: TensorBuilder

   .. automethod:: __call__

.. autoclass:: FieldBuilder

   .. automethod:: __call__

.. autofunction:: D

.. autofunction:: D_op

.. autofunction:: Op

.. autofunction:: OpSum		  

.. autofunction:: number_op

.. autofunction:: symbol_op

.. autofunction:: tensor_op

.. autofunction:: flavor_tensor_op

.. autodata:: kdelta

.. autodata:: generic

.. autodata:: epsUp

.. autodata:: epsUpDot

.. autodata:: epsDown

.. autodata:: epsDownDot

.. autodata:: sigma4bar

.. autodata:: sigma4

The ``integration`` module
==========================

.. automodule:: matchingtools.integration

.. autoclass:: RealScalar

.. autoclass:: ComplexScalar

.. autoclass:: RealVector

.. autoclass:: ComplexVector

.. autoclass:: VectorLikeFermion

.. autoclass:: MajoranaFermion

.. autofunction:: integrate

The ``transformations`` module
==============================

.. automodule:: matchingtools.transformations

.. autofunction:: collect_numbers

.. autofunction:: collect_symbols

.. autofunction:: collect_numbers_and_symbols
		  
.. autofunction:: apply_rule

.. autofunction:: apply_rules

.. autofunction:: collect_by_tensors

.. autofunction:: collect
		  
The ``output`` module
=====================

.. automodule:: matchingtools.output

.. autoclass:: Writer

   .. automethod:: __init__

   .. automethod:: __str__

   .. automethod:: write_text_file

   .. automethod:: latex_code

   .. automethod:: write_latex

   .. automethod:: show_pdf

The ``extras`` package
======================

The ``matchingtools.extras`` package provides several modules with some
useful definitions of tensors related to the Lorentz group and the
groups :math:`SU(3)` and :math:`SU(2)`, as well as rules for
transforming some combinations of these tensors into others.

The definitions for the Standard Model tensors and fields, together
with the rules derived from their equations of motion for the
substitution of their covariant derivatives are provided. A basis
for the Standard Model effective Lagrangian up to dimension 6
is also given.

The ``extras.SU2`` module
-------------------------

.. automodule:: matchingtools.extras.SU2
   :members:

The ``extras.SU3`` module
-------------------------

.. automodule:: matchingtools.extras.SU3
   :members:

The ``extras.Lorentz`` module
-----------------------------

.. automodule:: matchingtools.extras.Lorentz
   :members:

The ``extras.SM`` module
------------------------

.. automodule:: matchingtools.extras.SM
   :members:
      
The ``extras.SM_dim_6_basis`` module
------------------------------------

.. automodule:: matchingtools.extras.SM_dim_6_basis
   :members:      

