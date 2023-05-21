Creation of models
==================

.. currentmodule:: matchingtools.core

.. note:: This section assumes that the classes and functions
   from :mod:`matchingtools.core` that it uses are in the namespace.
   To import all the classes and functions that appear here do::

      from matchingtools.core import (
	  Tensor, Operator, OperatorSum
          TensorBuilder, FieldBuilder, D, Op, OpSum,
	  number_op, power_op)
	     
The construction of a model is done in two steps: the creation of the
tensors and fields and the definition of the interaction Lagrangian.

The basic building block for a Lagrangian is tensor, an object of the
class :class:`Tensor`. Direct usage of the :class:`Tensor`
constructor obscures the code. There are two classes defined to make the
process of creating tensors easier and cleaner,
:class:`TensorBuilder` and
:class:`FieldBuilder`.

When a field has covariant derivatives applied to it, that is
represented internally in the attributes of its representative
:class:`Tensor` object. For aesthetical reasons and easeness of usage,
instead of manually modifying the attributes, it's better to use
the function :class:`D` to create covariant
derivatives of fields.

MatchingTools handles Lagrangians that are polynomials of the fields.
They are thus a sum of terms that are products of tensors. They are
represented as :class:`OperatorSum` objects, with only one
attribute: a list of its terms. Each term should be an operator, that is,
a product of tensors represented by an :class:`Operator`
object with only one attribute: a list of its factors.

Instead of using the constructors for both classes, the functions
:class:`OpSum` and :class:`Op` are
available to make the definitions clearer.

Minus signs are defined in the library for operators (
:meth:`Operator.__minus__`) and operator sums
(:meth:`OperatorSum.__minus__`). Multiplication ``*``
is defined for operators too (as the
concatenation of the tensors they contain,
see :meth:`Operator.__mul__`).

MatchingTools treats in a special way tensors whose name starts and
ends with square or curly brakets. A tensor name enclosed in square
brakets is understood as a (complex) number to be read from the name
using ``float(name[1:-1])``. The function :func:`number_op`
is to be used to create operators with such tensors inside.

When the name of a tensor starts and ends with curly brakets it's
it represents some symbolic constant that appears exponentiated.
The name should be of the form ``"{base^exponent}"``. Curly brakets
allow for the summation of the exponents of tensors that appear in
the same tensor and have the same base and indices. This is used
mainly to produce more readable results. The function designed to
create operators containing such tensors is :func:`power_op`.


Creation of tensors
-------------------

Create a tensor as::

  my_tensor = TensorBuilder("my_tensor")

and then use it inside an operator::
  
  Op(..., my_tensor(ind1, ind2, ...), ...)

with ``ind1``, ``ind2``, ... being integers.
   
Creation of fields
------------------

Create a field as::

  my_field = FieldBuilder("my_field", dimension, statistics)

where dimension (float) represents the energy dimensions of the field
and statistics is either equal to :data:`matchingtools.algebra.boson` or
:data:`matchingtools.algebra.fermion`. Then use it inside an operator::

  Op(..., my_field(ind1, ind2, ...), ...)

with ``ind1``, ``ind2``, ... being integers.

Definition of the Lagrangian
----------------------------

Define the interaction Lagrangian as an operator sum::

  int_lag = OpSum(op1, op2, ...)

Each argument to the function :func:`matchingtools.core.OpSum` should
be an operator defined as::

  op1 = Op(tens1(ind1, ind2, ...), field1(ind3, ind4, ...), ...)

The arguments of the function :func:`matchingtools.core.Op` are
tensors (and fields). Their indices are integer numbers. Negative
integers are reserved for free indices. Free indices are not meant
to be used in the operators appearing in the Lagrangian, but later
in the definition of their transformations.

Non-negative integers represent contracted indices. Contraction
is expressed by repetition of indices.

To introduce the covariant derivative with index ``ind`` of a tensor
``tens`` inside an operator, use the function
:func:`matchingtools.core.D` in the following way::

  D(ind, tens(ind1, ind2, ...))

If a numeric coefficient ``num`` is needed for some operator ``op``
it can be introduced as::

  number_op(num) * op

A symbolic constant ``s`` to some power ``p`` can multiply an operator as::

  power_op("s", p) * op
  
