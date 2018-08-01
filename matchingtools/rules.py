# coding: utf-8

from matchingtools.core import (
    Operator, OperatorSum, Statistics
)
from matchingtools.indices import Index
from matchingtools.matches import Match
from matchingtools.utils import Permutation


class Rule(object):
    def __init__(self, pattern, replacement):
        self.pattern = pattern._to_operator()
        self.replacement = replacement._to_operator_sum()

    def _apply_to_operator(self, target):
        matches = Match.match_operators(self.pattern, target)

        if matches is None:
            return target

        match = next(matches)

        # TODO: It seems that there's a bug here: target_tensor not in matched
        # causes the problem.
        #
        # matched = [
        #     match.tensors_mapping[pattern_tensor]
        #     for pattern_tensor in self.pattern.tensors
        # ]
        # rest = [
        #     target_tensor for target_tensor in target.tensors
        #     if target_tensor not in matched
        # ]
        #
        # Using this code instead:
        matched = []
        rest = target.tensors.copy()
        for pattern_tensor in self.pattern.tensors:
            mapped = match.tensors_mapping[pattern_tensor]
            matched.append(mapped)
            rest.remove(mapped)

        # compute the new operator tensors list
        reordered_target = matched + rest

        # check if a sign change is needed because of the reordering
        original_fermions = [
            tensor for tensor in target.tensors
            if tensor.statistics == Statistics.FERMION
        ]
        reordered_fermions = [
            tensor for tensor in reordered_target
            if tensor.statistics == Statistics.FERMION
        ]

        fermions_permutation = Permutation.compare(
            original_fermions, reordered_fermions
        )
        sign = fermions_permutation.parity

        # extend indices mapping
        for operator in self.replacement.operators:
            for tensor in operator.tensors:
                for index in tensor.all_indices:
                    is_mapped = index in match.indices_mapping
                    is_contracted = not operator.is_free_index(index)
                    name_clashes = any(
                        index in original_tensor for original_tensor in rest
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
        
        return sign * adapted_replacement * Operator(rest)

    def apply(self, target):
        target = target._to_operator_sum()

        return sum(
            (
                self._apply_to_operator(operator)
                for operator in target.operators
            ),
            OperatorSum()
        )
