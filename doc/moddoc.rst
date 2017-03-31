Modules documentation
*********************

The ``operators`` module
========================

.. automodule:: efttools.operators

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

.. automodule:: efttools.integration

.. autoclass:: RealScalar

.. autoclass:: ComplexScalar

.. autoclass:: RealVector

.. autoclass:: ComplexVector

.. autoclass:: VectorLikeFermion

.. autoclass:: MajoranaFermion

.. autofunction:: integrate

The ``transformations`` module
==============================

.. automodule:: efttools.transformations

.. autofunction:: collect_numbers

.. autofunction:: collect_symbols

.. autofunction:: collect_numbers_and_symbols
		  
.. autofunction:: apply_rule

.. autofunction:: apply_rules_until

.. autofunction:: collect_by_tensors

.. autofunction:: collect
		  
The ``output`` module
=====================

.. automodule:: efttools.output

.. autoclass:: Writer

   .. automethod:: __init__

   .. automethod:: __str__

   .. automethod:: write_text_file

   .. automethod:: latex_code

   .. automethod:: write_latex

   .. automethod:: show_pdf

The ``extras.SM`` module
========================

.. automodule:: efttools.extras.SM
   :members:

The ``extras.SU2`` module
=========================

.. automodule:: efttools.extras.SU2
   :members:


The ``extras.Lorentz`` module
=============================

.. automodule:: efttools.extras.Lorentz
   :members:

The ``extras.SM_dim_6_basis`` module
====================================

.. automodule:: efttools.extras.SM_dim_6_basis
   :members:      

