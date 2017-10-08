"""
Module dedicated to the transformation and simplification of
operators and operator sums. Defines functions for collecting
numeric and symbolic factors.

Implements the function :func:`apply_rules_until` for the
systematic substitution of patterns inside operators in an 
operator sum until writing the sum it in a given basis of 
operators; and the function :func:`collect` for collecting the
coefficients of each operator in a given list.
"""

import sys

from matchingtools.core import (
    Operator, OperatorSum,
    number_op, symbol_op, tensor_op, kdelta, generic)

def collect_numbers(operator):
    """
    Collect all the tensors representing numbers into a single one.

    A tensor is understood to represent a number when its name
    is of the form ``"[number]"``.
    """
    new_tensors = []
    number = 1
    for tensor in operator.tensors:
        if tensor.name[0] == "[" and tensor.name[-1] == "]":
            number *= complex(tensor.name[1:-1])
        else:
            new_tensors.append(tensor)

    # Remove coefficients 1
    if number == 1:
        return Operator(new_tensors)
    
    return number_op(number) * Operator(new_tensors)

def collect_symbols(operator):
    """
    Collect all the tensors representing each symbolic constant
    with exponent into a single one.

    A tensor is understood to represent a symbolic constant with
    exponent when its name is of the form ``"{base^exponent}"``.
    """
    new_tensors = []
    symbols = {}
    for tensor in operator.tensors:
        if tensor.name[0] == "{" and tensor.name[-1] == "}":
            name = tensor.name[1:tensor.name.index("^")]
            exponent = float(tensor.name[tensor.name.index("^")+1:-1])

            # Previusly collected exponent for same base and indices
            prev_exponent = symbols.get((name, tuple(tensor.indices)), 0)

            # The exponents of a product are added
            symbols[(name, tuple(tensor.indices))] = exponent + prev_exponent
        else:
            new_tensors.append(tensor)

    # Remove tensors with exponent 0
    new_op = Operator([])
    for (symbol, inds), exponent in symbols.items():
        if exponent != 0:
            new_op *= symbol_op(symbol, exponent, indices=inds)
            
    return new_op * Operator(new_tensors)

def collect_numbers_and_symbols(op_sum):
    """ 
    Collect all the tensors representing all numeric factors and each
    symbolic constant into a single one for all the operators in the sum.
    """
    return OperatorSum([collect_numbers(collect_symbols(op))
                        for op in op_sum.operators])

def remove_kdeltas(operator):
    """
    Remove all the Kroneker deltas, substituting them by the
    corresponding index contraction.
    """
    for pos, tensor in enumerate(operator.tensors):
        if tensor.name == "kdelta":
            new_op = OperatorSum([operator.remove_tensor(pos)])
            n = new_op.operators[0].max_index + 1
            new_op = Operator([generic(n, n)]).replace_first("generic", new_op)
            return remove_kdeltas(new_op.operators[0])
    return operator

def apply_rule(operator, pattern, replacement):
    """
    Replace the first occurrence of ``pattern`` by ``replacement``
    in ``operator``
    """
    new_op = operator.match_first(pattern)
    if new_op is None:
        return None
    return new_op.replace_first("generic", replacement)

def apply_rules_aux(op_sum, rules):
    """
    Auxiliary function for :func:`apply_rules`. 

    Do the actual computations for each iteration.
    """
    for pattern, replacement in rules:
        new_op_sum = OperatorSum()
        for operator in op_sum.operators:
            operator = remove_kdeltas(operator)
            new_ops = apply_rule(operator, pattern, replacement)
            if new_ops is not None:
                new_op_sum += new_ops
            else:
                new_op_sum += OperatorSum([operator])
        op_sum = new_op_sum
    return new_op_sum

def apply_rules(op_sum, rules, max_iterations, verbose=True):
    """
    Apply all the given rules to the operator sum.

    With the adecuate set of rules this function can be used to express
    an effective lagrangian in a specific basis of operators

    Args:
        op_sum (:class:`matchingtools.operator.OperatorSum`): to which the rules
            should be applied.
        rules (list of pairs (:class:`matchingtools.operators.Operator`,  :class:`matchingtools.operators.OperatorSum`)): The first element
            of each pair represents a pattern to be subtituted in each
            operator by the second element using :func:`apply_rule`.
        max_iterations (int): maximum number of application of rules to
            each operator.
        verbose (bool): specifies whether to print messages signaling
            the start and end of the integration process

    Return:
        OperatorSum containing the result of the application of rules.
    """
    if verbose:
        sys.stdout.write("Applying rules... ")
        sys.stdout.flush()
    
    for i in range(0, max_iterations):
        op_sum =  apply_rules_aux(op_sum, rules)

    if verbose:
        sys.stdout.write("done.\n")
        
    return op_sum


def sum_numbers(op_sum):
    """
    Collect operators that are equal except for a numeric coefficient
    and sum the numbers to get one.
    """
    collection = []
    for op in op_sum.operators:
        # Strip numeric coefficient off
        collected = False
        num = 1
        n_removed = 0
        op = collect_numbers(op)
        if op.tensors[0].name[0] == "[" and op.tensors[0].name[-1] == "]":
            num = complex(op.tensors[0].name[1:-1])
            new_op = Operator(op.tensors[1:])
        else:
            num = 1
            new_op = op

        # Sum the numbers of equal operators
        for i, (collected_op, collected_num) in enumerate(collection):
            if collected_op == new_op:
                collected = True
                collection[i] = (new_op, num + collected_num)
                break
        if not collected:
            collection.append((new_op, num))
    return [(o, num) for o, num in collection if abs(num) > 10**(-10)]

def collect_by_tensors(op_sum, tensor_names):
    """
    Collect the coefficients of the given tensors.

    Usually, these tensors represent operators of a basis
    in which the effective lagrangian is expressed.

    Args:
        op_sum (OperatorSum): whose terms are to be collectd
        tensor_names (list of strings): names of the tensors
            whose coefficients will be obtained

    Return:
       A pair (collection, rest) where collection is a list of
       pairs (name, coef) where name is the name of each of the
       tensors and coef is an OperatorSum representing its
       coefficients; and where rest is an OperatorSum with the
       operators that didn't contain a tensor with one of the
       given names.
    """
    collection = {}
    rest = []
    for op in op_sum.operators:
        for pos, tensor in enumerate(op.tensors):
            if tensor.name in tensor_names:
                n = len(tensor.indices)
                collected = collection.get((tensor.name, n), OperatorSum())
                new_ops = OperatorSum([op.remove_tensor(pos)])
                collection[(tensor.name, n)] = collected + new_ops
                break
        else:
            rest.append(op)
    pair_collection = []
    for key in collection.keys():
        s = sum_numbers(collection[key])
        if s:
            pair_collection.append((key, s))
    rest = sum_numbers(OperatorSum(rest))
    return sorted(pair_collection, key=(lambda x: x[0])), rest

def sum_collection(collection):
    op_sum = OperatorSum()
    for (op_name, n_inds), coef in collection:
        for op, num in coef:
            op_sum += OperatorSum([
                number_op(num) * tensor_op(op_name, list(range(n_inds)))
                * op])
    return op_sum

def simplify(op_sum):
    op_sum = collect_numbers_and_symbols(op_sum)
    return OperatorSum([number_op(n) * op
                        for op, n in sum_numbers(op_sum)])

def collect(op_sum, tensor_names, verbose=True):
    """
    Simplify the numeric and exponentiated symbolic tensors
    (using :func:`collect_numbers_and_symbols`) and collect 
    the coefficients of the given tensors (using
    :func:`collect_by_tensors`).

    Args:
        op_sum (OperatorSum): whose terms are to be collectd
        tensor_names (list of strings): names of the tensors
            whose coefficients will be obtained
        verbose (bool): specify whether to write messages
            signaling the start and end of the computation.
    """
    if verbose:
        sys.stdout.write("Collecting...")
        sys.stdout.flush()
    op_sum = collect_numbers_and_symbols(op_sum)
    collection, rest = collect_by_tensors(op_sum, tensor_names)
    if verbose:
        sys.stdout.write("done.\n")
    return collection, rest
