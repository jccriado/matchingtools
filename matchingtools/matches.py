# coding: utf-8

import itertools

from matchingtools.utils import groupby


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

    def __init__(self, tensors_mapping, indices_mapping):
        self.tensors_mapping = tensors_mapping
        self.indices_mapping = indices_mapping

    @staticmethod
    def extend_dict(base, addition):
        for key, value in addition.items():
            incoherent = (
                key in base
                and base[key] != value
            )

            if incoherent:
                raise Match.IncoherenceError(base, addition)

        base.update(addition)

    @staticmethod
    def _map_tensors(pattern_operator, target_operator):
        Tensor = type(pattern_operator.tensors[0])  # not true, but good enough
        pattern_eqclasses = groupby(
            pattern_operator.tensors,
            Tensor.does_match
        )

        associates = {
            pattern_eqclass[0]: []
            for pattern_eqclass in pattern_eqclasses
        }

        for target_tensor in target_operator.tensors:
            for pattern_representative in associates:
                if pattern_representative.does_match(target_tensor):
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
    def match_operators(pattern, target):
        tensors_mappings = Match._map_tensors(pattern, target)

        if tensors_mappings is None:
            return None

        for tensor_mapping in tensors_mappings:
            """
            Testing if a certain possible tensor mapping leads to a viable
            indices mapping. If there is any inconsistency in the indices
            mapping we can discard it right away.
            """
            indices_mapping = Match._map_operator_indices(tensor_mapping)

            if indices_mapping is None:
                continue

            yield Match(tensor_mapping, indices_mapping)

        return None
