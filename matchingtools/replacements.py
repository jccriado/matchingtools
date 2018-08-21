import uuid

from matchingtools.indices import Index
from matchingtools.core import Operator, OperatorSum
from matchingtools.rules import Rule


class Replacement(object):
    class DifferentiatedTensorError(Exception):
        def __init__(self, differentiated_tensor):
            error_msg = "Cannot replace tensor with derivatives: {}"
            super().__init__(
                error_msg.format(differentiated_tensor)
            )

    def __init__(self, tensor, replacement):
        if len(tensor.derivatives_indices) > 0:
            raise Replacement.DifferentiatedTensorError(tensor)

        self._id = uuid.uuid4()
        self.tensor = tensor
        self.replacement = replacement

    def mark(self, tensor):
        new_tensor = tensor.clone()
        new_tensor.name = self._id
        return new_tensor

    def __str__(self):
        return "{} -> {}".format(self.tensor, self.replacement)

    __repr__ = __str__

    def _generate_rule(self, tensor):
        """
        A rule for substituting any tensor by the nth derivative of the
        replacement. The indices of these derivatives are given by the
        indices of the derivatives of the tensor.
        """

        new_indices_mapping = {
            old_index: Index(old_index.name)
            for old_index in tensor.derivatives_indices
        }
        tensor = (
            self.mark(self.tensor)._replace_indices(new_indices_mapping)
            .nth_derivative(tensor.derivatives_indices)
        )
        replacement = (
            self.replacement._replace_indices(new_indices_mapping)
            .nth_derivative(tensor.derivatives_indices)
        )

        if tensor.is_conjugated == self.tensor.is_conjugated:
            return Rule(tensor, replacement)
        else:
            return Rule(tensor.conjugate(), replacement.conjugate())

    def _replace_in_operator(self, operator):
        """
        Replace the first tensor with name self.tensor.name by the nth
        derivative of the replacement.
        """
        for tensor in operator.tensors:
            if tensor.name == self._id:
                return self._generate_rule(tensor).apply(operator)
        return operator._to_operator_sum()

    def replace_once(self, target, max_dimension):
        """ Substitute once in each operator of target. """
        return sum(
            (
                self._replace_in_operator(operator)
                .filter_by_max_dimension(max_dimension)
                for operator in target._to_operator_sum().operators
            ),
            OperatorSum()
        )

    def is_in(self, target):
        """
        Find whether there is a tensor in target with name self.tensor.name.
        """
        for operator in target.operators:
            for tensor in operator.tensors:
                if tensor.name == self._id:
                    return True
        return False

    def replace_all_occurrences(self, target, max_dimension):
        """
        Replace all occurrences of self.tensor in the original target.
        """

        target = OperatorSum([
            Operator(
                [
                    self.mark(tensor)
                    if tensor.name == self.tensor.name
                    else tensor
                    for tensor in operator.tensors
                ],
                operator.coefficient
            )
            for operator in target.operators
        ])

        while self.is_in(target):
            target = self.replace_once(target, max_dimension)

        return target
