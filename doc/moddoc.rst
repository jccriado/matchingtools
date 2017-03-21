Modules documentation
*********************

The ``operators`` module
========================

.. automodule:: efttools.operators

.. autoclass:: efttools.operators.Tensor

.. autoclass:: efttools.operators.Operator
	       
   .. automethod:: variation

   .. automethod:: replace_first

   .. automethod:: replace_all

   .. automethod:: match_first

   .. automethod:: __eq__

.. autoclass:: efttools.operators.OperatorSum

   .. automethod:: variation

   .. automethod:: replace_all

.. autoclass:: efttools.operators.TensorBuilder

   .. automethod:: __call__

.. autoclass:: efttools.operators.FieldBuilder

   .. automethod:: __call__

.. autofunction:: efttools.operators.D

.. autofunction:: efttools.operators.D_op

.. autofunction:: efttools.operators.Op

.. autofunction:: efttools.operators.OpSum		  

.. autofunction:: efttools.operators.number_op

.. autofunction:: efttools.operators.symbol_op

.. autofunction:: efttools.operators.tensor_op

.. autofunction:: efttools.operators.flavor_tensor_op

.. autodata:: efttools.operators.kdelta

.. autodata:: efttools.operators.generic

.. autodata:: efttools.operators.epsUp

.. autodata:: efttools.operators.epsUpDot

.. autodata:: efttools.operators.epsDown

.. autodata:: efttools.operators.epsDownDot

.. autodata:: efttools.operators.sigma4bar

.. autodata:: efttools.operators.sigma4

The ``integration`` module
==========================

.. automodule:: efttools.integration

.. autoclass:: efttools.integration.RealScalar

.. autoclass:: efttools.integration.ComplexScalar

.. autoclass:: efttools.integration.RealVector

.. autoclass:: efttools.integration.ComplexVector

.. autoclass:: efttools.integration.VectorLikeFermion

.. autoclass:: efttools.integration.MajoranaFermion

.. autofunction:: efttools.integration.integrate

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
