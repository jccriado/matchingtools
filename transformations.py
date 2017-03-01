from operators import (
    Tensor, Operator, OperatorSum,
    TensorBuilder, FieldBuilder, D,
    apply_derivatives, concat,
    number_op, symbol_op, kdelta, generic)

def collect_numbers(operator):
    new_tensors = []
    number = 1
    for tensor in operator.tensors:
        if tensor.name[0] == "[" and tensor.name[-1] == "]":
            number *= complex(tensor.name[1:-1])
        else:
            new_tensors.append(tensor)
    if number == 1:
        return Operator(new_tensors)
    return number_op(number) * Operator(new_tensors)

def collect_symbols(operator):
    new_tensors = []
    symbols = {}
    for tensor in operator.tensors:
        if tensor.name[0] == "{" and tensor.name[-1] == "}":
            name = tensor.name[1:tensor.name.index("^")]
            exponent = float(tensor.name[tensor.name.index("^")+1:-1])
            prev_exponent = symbols.get((name, tuple(tensor.indices)), 0)
            symbols[(name, tuple(tensor.indices))] = exponent + prev_exponent
        else:
            new_tensors.append(tensor)
    new_op = Operator([])
    for (symbol, inds), exponent in symbols.items():
        if exponent != 0:
            new_op *= symbol_op(symbol, exponent, indices=inds)
    return new_op * Operator(new_tensors)

def collect_numbers_and_symbols(op_sum):
    return OperatorSum([collect_numbers(collect_symbols(op))
                        for op in op_sum.operators])

def remove_kdeltas(operator):
    for pos, tensor in enumerate(operator.tensors):
        if tensor.name== "kdelta":
            new_op = OperatorSum([operator.remove_tensor(pos)])
            n = new_op.operators[0].max_index + 1
            new_op = Operator([generic(n, n)]).replace_first("generic", new_op)
            return remove_kdeltas(new_op.operators[0])
    return operator

def apply_rule(operator, pattern, replacement):
    new_op = operator.match_first(pattern)
    if new_op is None:
        return None
    return new_op.replace_first("generic", replacement)

def apply_rules_until_aux(op_sum, rules, final_tensor_names, done):
    new_op_sum = OperatorSum()
    for operator in op_sum.operators:
        operator = remove_kdeltas(operator)
        for i, (pattern, replacement) in enumerate(rules):
            new_ops = apply_rule(operator, pattern, replacement)
            if new_ops is not None:
                new_op_sum += new_ops
                break
        else:
            new_op_sum += OperatorSum([operator])
    remaining = []
    for operator in new_op_sum.operators:
        if any(operator.contains(name) for name in final_tensor_names):
            done.append(operator)
        else:
            remaining.append(operator)
    return done, remaining

def apply_rules_until(op_sum, rules, final_tensor_names, max_iterations):
    remaining = op_sum.operators
    done = []
    for i in range(0, max_iterations):
        done, remaining = apply_rules_until_aux(
                OperatorSum(remaining), rules, final_tensor_names, done)
    return OperatorSum(done + remaining)

def sum_numbers(op_sum):
    collection = []
    for op in op_sum.operators:
        collected = False
        num = 1
        n_removed = 0
        for pos, tensor in enumerate(op.tensors):
            if tensor.name[0] == "[" and tensor.name[-1] == "]":
                op = op.remove_tensor(pos - n_removed)
                n_removed += 1
                num *= complex(tensor.name[1:-1])
        for i, (collected_op, collected_num) in enumerate(collection):
            if collected_op == op:
                collected = True
                collection[i] = (op, num + collected_num)
                break
        if not collected:
            collection.append((op, num))
    return [(op, num) for op, num in collection if num != 0]

def collect_by_tensors(op_sum, tensor_names):
    collection = {}
    rest = []
    for op in op_sum.operators:
        for pos, tensor in enumerate(op.tensors):
            if tensor.name in tensor_names:
                collected = collection.get(tensor.name, OperatorSum())
                new_ops = OperatorSum([op.remove_tensor(pos)])
                collection[tensor.name] = collected + new_ops
                break
        else:
            rest.append(op)
    pair_collection = []
    for key in collection.keys():
        s = sum_numbers(collection[key])
        if s:
            pair_collection.append((key, s))
    return sorted(pair_collection, key=(lambda x: x[0])), rest

def group_op_sum(op_sum):
    op_sum = collect_numbers_and_symbols(op_sum)
    return OperatorSum([number_op(n) * op
                        for op, n in sum_numbers(op_sum)])
