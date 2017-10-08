Transformations of the effective Lagrangian
===========================================

.. currentmodule:: matchingtools.transformations

.. note:: This section assumes that the functions
   from :mod:`matchingtools.transformations` and
   :mod:`matchingtools.transformations` that it uses are in the namespace.
   To import all the functions that appear here do::

     from matchingtools.operators import tensor_op, flavor_tensor_op
     
     from matchingtools.transformations import (
         simplify, apply_rules)
		   
An effective Lagrangian obtained from integration of heavy fields
usually contains operators that aren't independent. Several
transformations can be applied to them to write the Lagrangian
in terms of a set of operators (a basis) that spans the space
of effective operators.

These transformations are such as Fierz identities or substitutions
of the equations of motion of the light particles. All of them consist
of the subsitution of operators or parts of them by sums of other
operators. The operations described in this section applied to
effective Lagrangians or to any other kind of operators sum.

The first step to simplify an effective Lagrangian is to collect and
multiply numeric coefficients and constant tensors that appear several
times inside the same operator. To do this, use::

  simplified_lag_1 = simplify(effective_lagrangian)

Then we can define a set of rules as a list of pairs
``(pattern, replacement)`` where ``pattern`` is an operator
and ``replacement`` is an operator sum::

  rules = [(Op(...), OpSum(...)), (Op(...), OpSum(...)), ...]

``pattern`` may contain tensors with negative indices corresponding
to indices that are not contracted inside ``pattern``. In that
case, the operators in replacement should also contain the same
negative indices. When ``pattern`` is substituted inside an operator
``op``, the indices in ``op`` outside ``pattern`` that are contracted
with indices inside pattern appear as contracted with the
corresponding ones in the operators of ``replacement``. For example,
to replace :math:`t_{ij}r_{ik}` by :math:`-s_{mnk}u_{nmj}` we would
write the rule::

  (Op(t(1, -1), r(1, -2)), OpSum(-Op(s(1, 2, -2), u(2, 1, -1))))

The operators of the basis should be represented by tensor with a name
identifing the operator. They can be defined using
:func:`matchingtools.operators.tensor_op` when they don't have
free indices and :func:`matchingtools.operators.flavor_tensor_op`
when they do. So we usually define::

  Op1 = tensor_op("Op1")
  Op2 = tensor_op("Op2")
  ...

  Opf1 = flavor_tensor_op("Opf1")
  Opf2 = flavor_tensor_op("Opf1")
  ...

and then specify how to identify them using rules::

  op_def_rules = [(Op(...), OpSum(Op1)),
                  (Op(...), OpSum(Op2)),
		  ...
                  (Op(...), OpSum(Opf1(i1, i2, ...))),
		  (Op(...), OpSum(Opf2(i1, i2, ...)))
		  ...]

Then we are ready to apply the rules using :func:`apply_rules`::

  simplified_lag_2 = apply_rules_until(
      simplified_lag_1, rules + op_def_rule, max_iter)

where ``max_iter`` is the maximum number of applications of all the
rules to each operator.
