from matchingtools.core import Operator
from matchingtools.shortcuts import D


def integration_by_parts_results(operator):
    results = [operator._to_operator_sum()]
    for pos, tensor in enumerate(operator.tensors):
        if len(tensor.derivatives_indices) > 0:
            new_tensor = tensor.clone()
            new_tensor.derivatives_indices = tensor.derivatives_indices[:-1]
            rest = operator.tensors[:pos] + operator.tensors[pos+1:]
            results.append(
                - new_tensor
                * D(tensor.derivatives_indices[-1], Operator(rest))
            )
    return results
