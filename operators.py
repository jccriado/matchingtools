import itertools
import permutations

def concat(lst):
    return list(itertools.chain(*lst))

class Tensor(object):
    def __init__(self, name, indices, is_field=False,
                 num_of_der=0, dimension=0, statistics=True):
        self.name = name
        self.indices = indices
        self.is_field = is_field
        self.num_of_der = num_of_der
        self.dimension = dimension
        self.statistics = statistics
   
    def __str__(self):
        output = ""
        for index in self.indices[:self.num_of_der]:
            output += "D(" + str(index) + ")"
        inds_str =  ",".join(map(str, self.indices[self.num_of_der:]))
        return output + self.name + "(" + inds_str + ")"

    __repr__ = __str__

    def __eq__(self, other):
        return (self.name == other.name and
                self.indices == other.indices and
                self.is_field == other.is_field and
                self.num_of_der == other.num_of_der and
                self.dimension == other.dimension and
                self.statistics == other.statistics)

    @property
    def der_indices(self):
        return self.indices[:self.num_of_der]

    @property
    def non_der_indices(self):
        return self.indices[self.num_of_der:]

    def change_indices(self, new_indices):
        return Tensor(self.name, new_indices,
                      is_field=self.is_field,
                      num_of_der=self.num_of_der,
                      dimension=self.dimension,
                      statistics=self.statistics)

class Operator(object):
    def __init__(self, tensors):
        self.tensors = tensors

    def __str__(self):
        return "".join(map(str, self.tensors))

    __repr__ = __str__

    def __mul__(self, other):
        return Operator(self.tensors + other.tensors)

    def __neg__(self):
        return number_op(-1) * self
    
    @property
    def dimension(self):
        return sum([tensor.dimension + tensor.num_of_der
                    for tensor in self.tensors])

    @property
    def max_index(self):
        indices = concat([tensor.indices for tensor in self.tensors])
        if type(indices[0]) is not int:
            raise Exception(str(self) + " " + str(indices))
        if indices:
            return max(indices)
        return 0

    def remaining_tensors(self, position):
        return self.tensors[:position] + self.tensors[position+1:]

    def contains(self, names):
        return any(tensor.name in names for tensor in self.tensors)
    
    def derivative(self, index):
        return OperatorSum(list(leibniz_rule(index, self)))
                
    def remove_tensor(self, position):
        target_indices = self.tensors[position].non_der_indices
        new_tensors = []
        for tensor in self.remaining_tensors(position):
            new_indices = remove_indices(tensor.indices, target_indices)
            new_tensors.append(tensor.change_indices(new_indices))
        return Operator(new_tensors)
    
    def variation(self, field_name, statistics):
        result = OperatorSum()
        for pos, tensor in enumerate(self.tensors):
            if tensor.name == field_name:
                der_inds = remove_indices(tensor.der_indices, tensor.non_der_indices)
                inside_op = self.remove_tensor(pos)
                number_of_fermions = len([1 for t in self.tensors[:pos]
                                          if not t.statistics])
                sign = (-1) ** len(der_inds) * ((-1) ** number_of_fermions
                                                if not statistics else 1)
                if sign == -1:
                    inside_op *= Operator([Tensor("[-1]", [])])
                inside_ops = OperatorSum([inside_op])
                result += apply_derivatives(list(reversed(der_inds)), inside_ops)
        return result

    def prepare_indices(self, incr, free_indices):
        new_tensors = []
        for tensor in self.tensors:
            new_indices = increase_and_bind_indices(
                tensor.indices, incr, free_indices)
            new_tensors.append(tensor.change_indices(new_indices))
        return Operator(new_tensors)

    def replace_at_position(self, position, operator):
        target_tensor = self.tensors[position]
        free_indices = target_tensor.non_der_indices
        if not isinstance(operator, Operator):
            raise Exception(str(operator))
        subs_op = operator.prepare_indices(self.max_index + 1, free_indices)
        der_subs_ops = apply_derivatives(target_tensor.der_indices,
                                         OperatorSum([subs_op]))
        tensors_left = self.tensors[:position]
        tensors_right = self.tensors[position+1:]
        return OperatorSum([Operator(tensors_left + op.tensors + tensors_right)
                            for op in der_subs_ops.operators])

    def replace_first(self, field_name, operator_sum):
        for pos, tensor in enumerate(self.tensors):
            if tensor.name == field_name:
                return sum([self.replace_at_position(pos, op)
                            for op in operator_sum.operators],
                           OperatorSum())

    def replace_all(self, substitutions, max_dim):
        for field_name, subs in substitutions.items():
            new_ops = self.replace_first(field_name, subs)
            if new_ops is not None:
                return new_ops.replace_all(substitutions, max_dim)
        return OperatorSum()

    def match_first(self, pattern):
        fermions = tuple(pos for pos, tensor in enumerate(self.tensors)
                         if not tensor.statistics)
        for match in match_tensor_lists(self.tensors, pattern.tensors):
            candidate = [self.tensors[pos] for pos in match]
            matches, free_indices = match_indices(candidate, pattern.tensors)
            if matches:
                fermion_reorder = tuple(fermions.index(pos)
                                        for pos in match if pos in fermions)
                sign = permutations.permutations(len(fermions))[fermion_reorder]
                if sign == -1: candidate.append(Tensor("[-1]", []))
                return Operator([generic(*free_indices)] +
                                candidate[len(pattern.tensors):])

    def __eq__(self, other):
        if len(self.tensors) != len(other.tensors):
            return False
        fermions = tuple(pos for pos, tensor in enumerate(self.tensors)
                         if not tensor.statistics)
        for match in match_tensor_lists(self.tensors, other.tensors):
            candidate = [self.tensors[pos] for pos in match]
            matches, free_indices = match_indices(candidate, other.tensors)
            if matches:
                fermion_reorder = tuple(fermions.index(pos)
                                        for pos in match if pos in fermions)
                sign = permutations.permutations(len(fermions))[fermion_reorder]
                if sign == -1:
                    return False
                same_free_indices = True
                for i, index in enumerate(free_indices):
                    if i != -1 - index:
                        same_free_indices = False
                if same_free_indices:
                    return True
        return False

class OperatorSum(object):
    def __init__(self, operators=None):
        if operators is None:
            operators = []
        self.operators = operators

    def __str__(self):
        return " + ".join(map(str, self.operators))

    __repr__ = __str__

    def __add__(self, other):
        return OperatorSum(self.operators + other.operators)

    def __mul__(self, other):
        return OperatorSum([self_op * other_op
                            for self_op in self.operators
                            for other_op in other.operators])

    def __neg__(self):
        return OperatorSum([-op for op in self.operators])

    def append(self, operator):
        return OperatorSum(self.operators + [operator])

    def derivative(self, index):
        return OperatorSum(concat([op.derivative(index).operators
                                   for op in self.operators]))

    def variation(self, field_name, statistics):
        return sum([op.variation(field_name, statistics)
                    for op in self.operators],
                   OperatorSum())
    
    def replace_all(self, substitutions, max_dim):
        field_names = substitutions.keys()
        changed = False
        result = OperatorSum()
        for operator in self.operators:
            if (operator.dimension <= max_dim and
                not operator.contains(field_names)):
                result = result.append(operator)
            elif operator.dimension < max_dim:
                changed = True
                result += operator.replace_all(substitutions, max_dim)
        if changed:
            return result.replace_all(substitutions, max_dim)
        else:
            return result

def rest(lst, index):
    return lst[:index] + lst[index+1:]

def enum_product(*iters):
    return itertools.product(*map(enumerate, iters))

def remove_indices(indices, inds_to_be_rm):
    return [ind if ind not in inds_to_be_rm
            else -inds_to_be_rm.index(ind)-1
            for ind in indices]

def increase_and_bind_indices(indices, incr, free_indices):
    return [ind + incr if ind >= 0
            else free_indices[-ind - 1]
            for ind in indices]
    
def leibniz_rule(index, operator):
    for i, tensor in enumerate(operator.tensors):
        if tensor.is_field:
            new_indices = [index] + tensor.indices
            new_tensor = Tensor(tensor.name, new_indices, is_field=True,
                                num_of_der=tensor.num_of_der + 1,
                                dimension=tensor.dimension,
                                statistics=tensor.statistics)
            yield Operator(operator.tensors[:i] + [new_tensor] +
                           operator.tensors[i+1:])

def apply_derivatives(indices, target):
    for index in reversed(indices):
        target = target.derivative(index)
    return target

def match_tensor_lists_aux(lst, pattern):
    if not pattern:
        return [tuple(pos for pos, _ in lst)]
    results = []
    for i, (pos, tensor) in enumerate(lst):
        if (tensor.name == pattern[0].name and
            tensor.num_of_der == pattern[0].num_of_der):
            match_rest = match_tensor_lists_aux(rest(lst, i), pattern[1:])
            results += [(pos,) + prev for prev in match_rest]
    return results

def match_tensor_lists(lst, pattern):
    return match_tensor_lists_aux(list(enumerate(lst)), pattern)

def match_indices(lst, pattern):
    for (pos1, t1), (pos2, t2) in enum_product(pattern, pattern):
        for (i, i1), (j, i2) in enum_product(t1.indices, t2.indices):
            li1 = lst[pos1].indices[i]
            li2 = lst[pos2].indices[j]
            if not ((i1 == i2) == (li1 == li2)):
                return False, None
    free_indices = [0] * sum([1 for t in pattern for i in t.indices if i < 0])
    for pos, tensor in enumerate(pattern):
        for i, index in enumerate(tensor.indices):
            if index < 0:
                free_indices[-index-1] = lst[pos].indices[i]
    return True, free_indices

class TensorBuilder(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, *indices):
        return Tensor(self.name, list(indices))

class FieldBuilder(object):
    def __init__(self, name, dimension, statistics):
        self.name = name
        self.dimension = dimension
        self.statistics = statistics

    def __call__(self, *indices):
        return Tensor(self.name, list(indices), is_field=True,
                      num_of_der=0, dimension=self.dimension,
                      statistics=self.statistics)

def D_op(index, *tensors):
    return Operator(list(tensors)).derivative(index)

def D(index, tensor):
    return Operator([tensor]).derivative(index).operators[0].tensors[0]

def Op(*tensors):
    return Operator(list(tensors))

def OpSum(*operators):
    if not operators:
        return OperatorSum()
    return OperatorSum(list(operators))

def number_op(number):
    return Operator([Tensor("[" + str(number) + "]", [])])

def symbol_op(symbol, exponent):
    return Operator([Tensor("{" + symbol + "^" + str(exponent) + "}", [])])

def tensor_op(name):
    return Operator([Tensor(name, [])])

def flavor_tensor_op(name):
    def f(*indices):
        return Op(Tensor(name, list(indices)))
    return f

kdelta = TensorBuilder("kdelta")
generic = TensorBuilder("generic")

boson = True
fermion = False

epsUp = TensorBuilder("epsUp")
epsUpDot = TensorBuilder("epsUpDot")
epsDown = TensorBuilder("epsDown")
epsDownDot = TensorBuilder("epsDownDot")
sigma4bar = TensorBuilder("sigma4bar")
sigma4 = TensorBuilder("sigma4")
