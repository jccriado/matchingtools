# coding: utf-8

from matchingtools.core import Operator, OperatorSum
from matchingtools.indices import Index
from matchingtools.matches import Match


class Rule(object):
    def __init__(self, pattern, replacement):
        self.pattern = pattern._to_operator()
        self.replacement = replacement._to_operator_sum()

    def _apply_to_operator(self, target):
        matches = Match.match_operators(self.pattern, target)

        if matches is None:
            return target

        match = next(matches)

        rest = target.tensors.copy()
        for pattern_tensor in self.pattern.tensors:
            rest.remove(match.tensors_mapping[pattern_tensor])

        # extend indices mapping
        for operator in self.replacement.operators:
            for tensor in operator.tensors:
                for index in tensor.all_indices:
                    is_mapped = index in match.indices_mapping
                    is_contracted = not operator.is_free_index(index)
                    name_clashes = any(
                        index in original_tensor
                        for original_tensor in rest
                    )

                    if not is_mapped and is_contracted and name_clashes:
                        # name could be repeated, but new_index is unique
                        new_index = Index(index.name)
                        match.indices_mapping[index] = new_index

        adapted_replacement = OperatorSum([
            Operator(
                [
                    tensor._replace_indices(match.indices_mapping)
                    for tensor in operator.tensors
                ],
                operator.coefficient
                * target.coefficient
                / self.pattern.coefficient
            )
            for operator in self.replacement.operators
        ])

        return match.sign * adapted_replacement * Operator(rest)

    def apply(self, target):
        target = target._to_operator_sum()

        return sum(
            (
                self._apply_to_operator(operator)
                for operator in target.operators
            ),
            OperatorSum()
        )
