# coding: utf-8

import itertools
from functools import reduce
from operator import mul

from matchingtools.utils import groupby, Permutation
from matchingtools.statistics import Statistics


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
    class IncoherenceError(Exception):
        def __init__(self, base, offendant):
            error_msg = "Unable to extend {} by {}: incoherent overlapping"
            super().__init__(
                error_msg.format(base, offendant)
            )

    def __init__(self, tensors_mapping, indices_mapping, sign, rest):
        self.tensors_mapping = tensors_mapping
        self.indices_mapping = indices_mapping
        self.sign = sign
        self.rest = rest

    @staticmethod
    def extend_dict(base, addition):
        for key, value in addition.items():
            incoherent = (
                (key in base and base[key] != value)
            )

            if incoherent:
                raise Match.IncoherenceError(base, addition)

        base.update(addition)

    @staticmethod
    def tensors_do_match(tensor, other):
        return tensor._match_attributes() == other._match_attributes()

    @staticmethod
    def _map_tensors(pattern_operator, target_operator):
        pattern_eqclasses = groupby(
            pattern_operator.tensors,
            Match.tensors_do_match
        )

        associates = {
            pattern_eqclass[0]: []
            for pattern_eqclass in pattern_eqclasses
        }

        for target_tensor in target_operator.tensors:
            for pattern_representative in associates:
                if Match.tensors_do_match(
                        pattern_representative,
                        target_tensor
                ):
                    associates[pattern_representative].append(target_tensor)
                    break

        partial_mappings = [
            [
                list(
                    zip(
                        pattern_eqclass,
                        image
                    )
                ) for base_image in itertools.combinations(
                    associates[pattern_eqclass[0]],
                    len(pattern_eqclass)
                ) for image in itertools.permutations(base_image)
            ] for pattern_eqclass in pattern_eqclasses
        ]

        mappings = (
            dict(
                itertools.chain.from_iterable(
                    local_mappings_choice
                )
            ) for local_mappings_choice in itertools.product(*partial_mappings)
        )

        return mappings

    @staticmethod
    def _map_tensor_indices(I, J):
        if len(I) != len(J):
            return None

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
            local_indices_mapping = Match._map_tensor_indices(
                tensor.indices, associate.indices
            )

            if local_indices_mapping is None:
                return None

            local_derivatives_mapping = Match._map_tensor_indices(
                tensor.derivatives_indices, associate.derivatives_indices
            )

            if local_derivatives_mapping is None:
                return None

            # check for local incoherence

            try:
                Match.extend_dict(
                    local_indices_mapping, local_derivatives_mapping
                )
            except Match.IncoherenceError:
                return None

            # check for incoherence when merging back

            try:
                Match.extend_dict(indices_mapping, local_indices_mapping)
            except Match.IncoherenceError:
                return None

        return indices_mapping

    @staticmethod
    def _match_operators_iterable(tensors_mappings, pattern, target):
        for tensor_mapping in tensors_mappings:
            # Compute all possible combinations of allowed permutations of the
            # indices of the target tensors in the candidate match
            index_permuted_mappings = itertools.product(*[
                [
                    (tensor, permuted_assoc, sign)
                    for permuted_assoc, sign in associate.index_permutations()
                ]
                for tensor, associate in tensor_mapping.items()
            ])

            for index_permuted_mapping in index_permuted_mappings:
                """
                Testing if a certain possible tensor mapping leads to a viable
                indices mapping. If there is any inconsistency in the indices
                mapping we can discard it right away.
                """
                # Compute the sign due to the permutation of the indices
                indices_sign = reduce(
                    mul,
                    (sign for _, _, sign in index_permuted_mapping),
                    1
                )

                # Compute the actual mapping of tensors, with each associate
                # with its indices permuted
                permuted_tensor_mapping = {
                    tensor: associate
                    for tensor, associate, _ in index_permuted_mapping
                }

                # Try to match the indices for this permutation
                indices_mapping = Match._map_operator_indices(
                    permuted_tensor_mapping
                )

                if indices_mapping is None:
                    continue

                # ------------------------------------------------------------

                # Identifying pairs of indices in the pattern that map to the
                # same index in the target, where one of the indices in the
                # pair are not free
                def non_injective_pair(fst_index, snd_index):
                    return (
                        indices_mapping[fst_index]
                        == indices_mapping[snd_index]
                        and not (
                            pattern.is_free_index(fst_index)
                            and pattern.is_free_index(snd_index)
                        )
                    )

                non_injective = any(
                    non_injective_pair(first_index, second_index)
                    for first_index, second_index
                    in itertools.combinations(indices_mapping.keys(), 2)
                )

                if non_injective:
                    continue

                # Check that pre-image of a free index should be free
                free_preimage_is_free = all(
                    not target.is_free_index(target_index)
                    or pattern.is_free_index(pattern_index)
                    for pattern_index, target_index in indices_mapping.items()
                )

                if not free_preimage_is_free:
                    continue

                # ------------------------------------------------------------
                # Compute the sign due to the fermion permutation
                matched = []
                rest = target.tensors.copy()
                for pattern_tensor in pattern.tensors:
                    mapped = tensor_mapping[pattern_tensor]
                    matched.append(mapped)
                    rest.remove(mapped)

                # Compute the new operator tensors list
                reordered_target = matched + rest

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
                fermion_sign = fermions_permutation.parity

                yield Match(
                    tensor_mapping,
                    indices_mapping,
                    fermion_sign * indices_sign,
                    rest
                )

    @staticmethod
    def match_operators(pattern, target):
        tensors_can_match = all(
            any(
                Match.tensors_do_match(pattern_tensor, target_tensor)
                for target_tensor in target.tensors
            )
            for pattern_tensor in pattern.tensors
        )

        if not tensors_can_match:
            return None

        tensors_mappings = list(Match._map_tensors(pattern, target))

        if tensors_mappings is None:
            return None

        try:
            next(Match._match_operators_iterable(
                tensors_mappings,
                pattern,
                target
            ))
            return Match._match_operators_iterable(
                tensors_mappings,
                pattern,
                target
            )
        except StopIteration:
            return None
