# coding: utf-8

from functools import reduce
import itertools
from operator import mul

from core import Constant, Operator, OperatorSum, RealMixin, Statistics
from indices import Index


class Rule(object):
    def __init__(self, pattern, replacement):
        self.pattern = pattern._to_operator()
        self.replacement = replacement._to_operator_sum()

    def _apply_to_operator(self, target):
        match = Match.match_operator(self.pattern, target)

        if match is None:
            return target

        matched = [
            match.tensors_mapping[pattern_tensor]
            for pattern_tensor in self.pattern.tensors
        ]
        rest = [
            target_tensor for target_tensor in target.tensors
            if target_tensor not in matched
        ]
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
                for index in tensor.indices:
                    is_mapped = index in match.indices_mapping
                    is_contracted = not operator.is_free_index(index)
                    name_clashes = any(
                        index in original_tensor for original_tensor in rest
                    )

                    if not is_mapped and is_contracted and name_clashes:
                        # TODO new_index might already be mapped (!)
                        new_index = Index(index.name + "'")
                        match.indices_mapping[index] = new_index

        adapted_replacement = OperatorSum([
            Operator([
                tensor._replace_indices(match.indices_mapping)
                for tensor in operator.tensors
            ])
            for operator in self.replacement.operators
        ])

        return sign * adapted_replacement * Operator(rest)

    def apply(self, target):
        target = target._to_operator_sum()

        return sum(
            self._apply_to_operator(operator) for operator in target.operators
        )


class Match(object):  # TODO make sure things that don't match don't match
    """
    Type representing an operator target - rule pattern match.
    For this to happen:
        - Every tensor in the pattern must appear in the target (with exact
          multiplicity). Order is not considered except for the parity of
          the permutation.
        - Let Î be the set of indices of the pattern and Ĵ the one of the
          target, f: Î -> Ĵ an injection, T(I) a tensor in the pattern and
          T(J) its associate in the target, I and J the indices of the
          aforementioned tensors ordered and with multiplicity.
          Then |I| = L = |J| and f(I[k]) = J[k] for every 1 <= k <= L.

    In this fashion, such a match is given by the correspondence between
    pattern's and target's tensors and the correspondence between their
    indices. Note that the parity of the permutation can be computed from
    the tensors correspondence.
    """
    def __init__(self, tensors_mapping, indices_mapping):
        self.tensors_mapping = tensors_mapping
        self.indices_mapping = indices_mapping

    @staticmethod
    def tensors_do_match(t1, t2):
        return (
            t1.name == t2.name
            and t1.statistics == t2.statistics
            and t1.is_conjugated == t2.is_conjugated
            and isinstance(t1, Constant) == isinstance(t2, Constant)
            and isinstance(t1, RealMixin) == isinstance(t2, RealMixin)
        )

    @staticmethod
    def _map_tensors(pattern_operator, target_operator):
        associates = {
            pattern_tensor: [] for pattern_tensor in pattern_operator.tensors
        }

        for target_tensor in target_operator.tensors:
            for pattern_tensor in pattern_operator.tensors:
                if Match.tensors_do_match(pattern_tensor, target_tensor):
                    associates[pattern_tensor].append(target_tensor)

        return (
            {
                tensor: associate
                for tensor, associate in zip(
                    pattern_operator.tensors, mapping
                )
            } for mapping in itertools.product(
                *[associates[tensor] for tensor in pattern_operator.tensors]
            )
        )


    @staticmethod
    def _map_tensor_indices(I, J):
        mapping = {}

        for i, j in zip(I, J):
            if i in mapping and mapping[i] != j:
                # unable to match the indices: incoherence reached
                return None
            else:
                # unknown index yet. try with the obvious mapping and
                # wait for an incoherence later on.
                mapping[i] = j

        return mapping

    @staticmethod
    def _map_operator_indices(tensor_mapping):
        indices_mapping = {}

        for tensor, associate in tensor_mapping.items():
            if len(tensor.indices) != len(associate.indices):
                return None

            local_indices_mapping = Match._map_tensor_indices(
                tensor.indices, associate.indices
            )

            if local_indices_mapping is None:
                return None

            for tensor_index, associate_index in local_indices_mapping.items():
                incoherent = (
                    tensor_index in indices_mapping  # known
                    and indices_mapping[tensor_index] != associate_index
                )

                if incoherent:
                    return None

            # coherent mapping for the moment
            indices_mapping.update(local_indices_mapping)

        return indices_mapping

    @staticmethod
    def match_operators(pattern, target):
        tensors_mappings = Match._map_tensors(pattern, target)

        for tensor_mapping in tensors_mappings:
            """
            Testing if a certain possible tensor mapping leads to a viable
            indices mapping. If there is any inconsistency in the indices
            mapping we can discard it right away.
            """
            indices_mapping = Match._map_operator_indices(tensor_mapping)

            if indices_mapping is None:
                continue

            return Match(tensor_mapping, indices_mapping)

        return None


class Permutation(object):
    def __init__(self, indices_transformation):
        self.indices_transformation = indices_transformation

    @staticmethod
    def compare(original, transformed):
        if len(original) != len(transformed):
            raise ValueError(
                "Unable to build permutation: samples have different sizes"
            )

        try:
            new_positions = [transformed.index(item) for item in original]
        except ValueError as e:
            raise ValueError(
                "Unable to build permutation: elements do not match"
            ) from e

        return Permutation(new_positions)

    @property
    def parity(self):
        return reduce(
            mul,
            [cycle.parity for cycle in self.get_cycles()]
        )

    def get_cycles(self):
        cycles = [[]]
        current_cycle = cycles[-1]
        current_index = 0
        used = [False] * len(self.indices_transformation)

        while not all(used):
            _next = self.indices_transformation[current_index]

            if _next in current_cycle:  # cycled
                cycles.append([])
                current_index = used.index(False)
            else:
                used[_next] = True
                current_cycle.append(_next)
                current_index = _next

        return [Cycle(cycle) for cycle in cycles]


class Cycle(object):
    def __init__(self, cycle):
        if len(cycle) == 1:
            raise ValueError("Unable to build cycle from a singleton list")

        self.cycle = cycle

    def __str__(self):
        return str(self.cycle)

    __repr__ = __str__

    @property
    def parity(self):
        if not self.cycle:  # identity
            return 1

        def odd(x):
            return (x % 2) == 1

        return 1 if odd(len(self.cycle)) else -1
