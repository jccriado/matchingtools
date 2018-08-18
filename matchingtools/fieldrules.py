from matchingtools.indices import Index
from matchingtools.core import OperatorSum
from matchingtools.rules import Rule


class FieldRule(object):
    def __init__(self, field, replacement):
        self.field = field
        self.replacement = replacement

    def __str__(self):
        return str("({} -> {})".format(self.field, self.replacement))

    __repr__ = __str__

    def _rule(self, tensor):
        """
        A rule for substituting any tensor by the nth derivative of the
        replacement. The indices of these derivatives are given by the
        indices of the derivatives of the tensor.
        """

        new_indices_mapping = {
            old_index: Index(old_index.name)
            for old_index in tensor.derivatives_indices
        }
        field = self.field._replace_indices(new_indices_mapping)
        replacement = self.replacement._replace_indices(new_indices_mapping)

        if tensor.is_conjugated == self.field.is_conjugated:
            return Rule(
                field.nth_derivative(tensor.derivatives_indices),
                replacement.nth_derivative(tensor.derivatives_indices)
            )
        else:
            return Rule(
                field.nth_derivative(
                    tensor.derivatives_indices
                ).conjugate(),

                replacement.nth_derivative(
                    tensor.derivatives_indices
                ).conjugate()
            )

    def _substitute_in_operator(self, operator):
        """
        Replace the first tensor with name self.field_name by the nth
        derivative of the replacement.
        """
        for tensor in operator.tensors:
            if tensor.name == self.field.name:
                return self._rule(tensor).apply(operator)
        return operator._to_operator_sum()

    def substitute_once(self, target, max_dimension):
        """ Substitute once in each operator of target. """
        return sum(
            (
                self._substitute_in_operator(operator)
                .filter_by_max_dimension(max_dimension)
                for operator in target._to_operator_sum().operators
            ),
            OperatorSum()
        )

    def is_in(self, target):
        """
        Find whether there is a tensor in target with name self.field_name.
        """
        for operator in target.operators:
            for tensor in operator.tensors:
                if tensor.name == self.field.name:
                    return True

        return False
