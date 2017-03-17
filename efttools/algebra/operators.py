"""
Definitions of the basic building blocks: the classes Tensor, Operator,
and OperatorSum. Implementation of the Leibniz rule for derivatives and
the algorithms for matching and replacing as well as functional 
derivatives.

Interfaces for the creation of tensors, fields, operators and operator 
sums: the functions TensorBuilder, FieldBuilder, Op and OpSum; 
interface for creating derivatives of single tensors: the function D;
interface for creating special single-tensor operators associated to
(complex) numbers and powers of constants.

Definition of the Lorentz tensors eps[Up/Down](Dot) and sigma4(bar).
"""

import permutations

from lsttools import concat, enum_product

class Tensor(object):
    """
    Building block for operators.

    An object of the class Tensor represents a tensor, which may be
    anything from structure constants to fields, flavor matrices or
    just coupling constants. This tensor might have some derivatives
    applied to it. The indices correponding to the derivatives
    correspond to the first indices in the list of indices of the
    Tensor.

    Attributes:
        name (string): string identifier
        indices ([int]): indices of the derivatives and the tensor
        is_field (bool): false iff the tennsor is constant
        num_of_der (int): number of derivatives acting on the tensor
        dimension (int): energy dimensions of the tensor
        statistics (bool): true for bosons, false for fermions
    """
    
    def __init__(self, name, indices, is_field=False,
                 num_of_der=0, dimension=0, statistics=True):
        self.name = name
        self.indices = indices
        self.is_field = is_field
        self.num_of_der = num_of_der
        self.dimension = dimension
        self.statistics = statistics
   
    def __str__(self):
        """
        Returns a string of the form:
            D(i[0])D(i[1])...D(i[m])T(i[m+1],i[m+2],...,i[n-1])
        for a Tensor with indices i[0], i[1], ..., i[n-1] and 
        num_of_der equal to m
        """
        der_str = ("D({})" * self.num_of_der).format(*self.der_indices)
        ten_str = "{}({})".format(self.name,
                                  ",".join(map(str, self.non_der_indices)))
        return der_str + ten_str

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
    """
    Container for a list of tensors with their indices contracted.

    Indices repeated in different tensors mean contraction. Contracted
    indices should be positive. Negative indices represent free indices
    and should appear in order: -1, -2, ...

    The methods include the basic derivation, matching and replacing 
    operations, as well as the implementation of functional derivatives.

    Attributes:
        tensors ([Tensor]): list of the tensors contained
    """
    
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
        if indices:
            return max(indices)
        return 0

    def contains(self, name):
        return any(tensor.name == name for tensor in self.tensors)

    def contains_symbol(self, name):
        for tensor in self.tensors:
            if tensor.name[0] == "{" and tensor.name[-1] == "}":
                if tensor.name[1:tensor.name.index("^")] == name:
                    return True
        return False
    
    def derivative(self, index):
        return leibniz_rule(index, self)
                
    def remove_tensor(self, position):
        """
        Return an operator obtained from self by removing the
        tensor at the given position and changing the correponding
        contracted indices to free ones.

        The derivatives acting on the tensor aren't taken into account.
        """
        target_indices = self.tensors[position].non_der_indices
        new_tensors = []
        for tensor in rest(self.tensors, position):
            new_indices = remove_indices(tensor.indices, target_indices)
            new_tensors.append(tensor.change_indices(new_indices))
        return Operator(new_tensors)
    
    def variation(self, field_name, statistics):
        """
        Take functional derivative of the spacetime integral of self.

        Args:
            field_name (string): the name of the field with respect to
                                 which the functional derivative is taken
            statistics (bool): statistics of the field
        """
        result = OperatorSum()
        for pos, tensor in enumerate(self.tensors):
            if tensor.name == field_name:
                inside_op = self.remove_tensor(pos)
                der_inds = remove_indices(tensor.der_indices,
                                          tensor.non_der_indices)
                
                # Compute the sign taking into account the number of
                # derivatives that act on the field and the number of
                # fermions before it in the fermionic case
                minus_sign = len(der_inds) % 2 == 1
                if statistics:
                    number_of_fermions = len([1 for t in self.tensors[:pos]
                                              if not t.statistics])
                    minus_sign = minus_sign != (number_of_fermions % 2 == 1)
                if minus_sign:
                    inside_op *= Operator([Tensor("[-1]", [])])

                # Apply the derivatives with the correponding indices
                inside_ops = OperatorSum([inside_op])
                result += apply_derivatives(list(reversed(der_inds)),
                                            inside_ops)
        return result

    def prepare_indices(self, incr, free_indices):
        """
        Increase all contracted indices by incr and set all free
        indices to free_indices in the order given:
            -1 -> free_indices[0], -2 -> free_indices[1], ...
        """
        new_tensors = []
        for tensor in self.tensors:
            new_indices = increase_and_bind_indices(
                tensor.indices, incr, free_indices)
            new_tensors.append(tensor.change_indices(new_indices))
        return Operator(new_tensors)

    def replace_at_position(self, position, operator):
        """Replace the tensor at the given position by the given operator"""
        # Prepare the replacement operator
        target_tensor = self.tensors[position]
        free_indices = target_tensor.non_der_indices
        subs_op = operator.prepare_indices(self.max_index + 1, free_indices)

        # Apply the corresponding derivatives
        der_subs_ops = apply_derivatives(target_tensor.der_indices,
                                         OperatorSum([subs_op]))

        # Insert in the corresponding position
        tens_left = self.tensors[:position]
        tens_right = self.tensors[position+1:]
        return OperatorSum([Operator(tens_left + op.tensors + tens_right)
                            for op in der_subs_ops.operators])

    def replace_first(self, field_name, operator_sum):
        """
        Replace the first ocurrence of a field.

        Args:
            field_name (string): name of the field to be replaced
            operator_sum (OperatorSum): replacement

        Return:
            An OperatorSum resulting from replacing the first ocurrence of
            the field by its replacement
        """
        for pos, tensor in enumerate(self.tensors):
            if tensor.name == field_name:
                return sum([self.replace_at_position(pos, op)
                            for op in operator_sum.operators],
                           OperatorSum())

    def replace_all(self, substitutions, max_dim):
        """
        Replace all ocurrences of several fields.

        Args:
            substitutions ([(string, OperatorSum)]): 
                list of pairs with the first element of the pair being the
                name of a field to be replaced and the second being the
                replacement.
            max_dim (int): maximum dimension of the operators in the result

        Return:
            An OperatorSum resulting from replacing every ocurrence of the
            fields by their replacements
        """
        for field_name, subs in substitutions.items():
            new_ops = self.replace_first(field_name, subs)
            if new_ops is not None:
                return new_ops.replace_all(substitutions, max_dim)
        return OperatorSum()

    def match_first(self, pattern):
        """
        Match the first ocurrence of a pattern

        Args:
            pattern (Operator): contains the tensors and index structure
                                to be matched

        Return:
            if the matching succeeds: an Operator with the first occurrence
                                      of the pattern substituted by a
                                      "generic" tensor (with a sign change
                                      if needed)
            otherwise: None
        """
        fermions = tuple(pos for pos, tensor in enumerate(self.tensors)
                         if not tensor.statistics)
        for match in match_tensor_lists(self.tensors, pattern.tensors):
            candidate = [self.tensors[pos] for pos in match]
            matches, free_indices = match_indices(candidate, pattern.tensors)
            if matches:
                # Compute change of sign due to fermion permutation
                fermion_reorder = tuple(fermions.index(pos)
                                        for pos in match if pos in fermions)
                sign = permutations.permutations(len(fermions))[fermion_reorder]
                if sign == -1: candidate.append(Tensor("[-1]", []))

                # Replace the matched part by a "generic" tensor
                return Operator([generic(*free_indices)] +
                                candidate[len(pattern.tensors):])

    def __eq__(self, other):
        """
        Match self with other operator. All tensors and index contractions
        should match. No sign differences allowed. All free indices should
        be equal.
        """
        if len(self.tensors) != len(other.tensors):
            return False
        fermions = tuple(pos for pos, tensor in enumerate(self.tensors)
                         if not tensor.statistics)
        for match in match_tensor_lists(self.tensors, other.tensors):
            candidate = [self.tensors[pos] for pos in match]
            matches, free_indices = match_indices(candidate, other.tensors)
            if matches:
                # If there has been an odd permutation of fermions,
                # equality fails
                fermion_reorder = tuple(fermions.index(pos)
                                        for pos in match if pos in fermions)
                sign = permutations.permutations(len(fermions))[fermion_reorder]
                if sign == -1: return False

                # Checking that all free indices are equal
                same_free_indices = True
                for i, index in enumerate(free_indices):
                    if i != -1 - index:
                        same_free_indices = False
                if same_free_indices:
                    return True
        return False

class OperatorSum(object):
    """
    Container for lists of operators representing their sum.

    The methods perform the basic operations defined for operators
    generalized for sums of them 
    
    Attributes:
        operators ([Operator]): the operators whose sum is represented
    """
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

    def derivative(self, index):
        """Takes the derivative with the given index"""
        return OperatorSum(concat([op.derivative(index).operators
                                   for op in self.operators]))

    def variation(self, field_name, statistics):
        """
        Take functional derivative of the spacetime integral of self.

        Args:
            field_name (string): the name of the field with respect to
                                 which the functional derivative is taken
            statistics (bool): statistics of the field
        """
        return sum([op.variation(field_name, statistics)
                    for op in self.operators],
                   OperatorSum())
    
    def replace_all(self, substitutions, max_dim):
        """
        Replace all ocurrences of several fields.

        Args:
            substitutions ([(string, OperatorSum)]): 
                list of pairs with the first element of the pair being the
                name of a field to be replaced and the second being the
                replacement.
            max_dim (int): maximum dimension of the operators in the result

        Return:
            An OperatorSum resulting from replacing every ocurrence of the
            fields by their replacements
        """
        field_names = substitutions.keys()
        changed = False
        result = []
        for operator in self.operators:
            if (operator.dimension <= max_dim and
                not any(operator.contains(name) for name in field_names)):
                result.append(operator)
            elif operator.dimension < max_dim:
                changed = True
                result += operator.replace_all(substitutions,
                                               max_dim).operators
        if changed:
            return OperatorSum(result).replace_all(substitutions, max_dim)
        else:
            return OperatorSum(result)

def rest(lst, index):
    return lst[:index] + lst[index+1:]

def remove_indices(indices, inds_to_be_rm):
    """
    Convert some bound indices to free ones.

    The conversion goes as: inds_to_be_rm[i] -> -1 - i

    Args:
        indices ([int]): list in which the conversion is to be done
        inds_to_be_rm ([int]): list of the indices to be made free in order

    Return:
       A list of the changed indices.
    """
    return [ind if ind not in inds_to_be_rm
            else -inds_to_be_rm.index(ind)-1
            for ind in indices]

def increase_and_bind_indices(indices, incr, free_indices):
    """
    Increase all indices greater than or equal to zero by some constant 
    amount and change the free indices to the ones given as:
        i -> free_indices[-i - 1]

    Args:
        indices ([int]): list in which the operation is to be done
        incr (int): constant amount to increase the non-negative indices
        free_indices ([int]): list of new free_indices

    Return:
        A list of the changed indices.
    """
    return [ind + incr if ind >= 0
            else free_indices[-ind - 1]
            for ind in indices]
    
def leibniz_rule(index, operator):
    """
    Take the derivative of an operator and apply the Leibniz rule to it

    Args:
        index (int): index of the derivative
        operator (Operator): operator to which the derivative is to be applied

    Return:
        An OperatorSum resulting from the Leibniz rule
    """
    result = []
    for i, tensor in enumerate(operator.tensors):
        if tensor.is_field:
            new_indices = [index] + tensor.indices
            new_tensor = Tensor(tensor.name, new_indices, is_field=True,
                                num_of_der=tensor.num_of_der + 1,
                                dimension=tensor.dimension,
                                statistics=tensor.statistics)
            result.append(Operator(operator.tensors[:i] + [new_tensor] +
                                   operator.tensors[i+1:]))
    return OperatorSum(result)

def apply_derivatives(indices, target):
    """Applies any number of derivatives to an Operator or OperatorSum"""
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
    """
    Match the names of a pattern list of tensors in another list of tensors

    Reorderings of the main list are considered.

    Args:
        lst ([Tensor]): list inside which the pattern is to be found
        pattern ([Tensor]): pattern to find

    Return:
        A list of tuples of integers. Each tuple represents a reordering
        of the positions of the tensors in lst that matches the names of
        the pattern in its first elements.
    """
    return match_tensor_lists_aux(list(enumerate(lst)), pattern)

def match_indices(lst, pattern):
    """
    Match the index structure of a pattern list of tensors in another
    list of tensors without reorderings

    Args:
        lst ([Tensor]): list inside which the pattern is to be found
        pattern ([Tensor]): pattern to find

    Return:
        if they match: a pair with the first element being True and the
                       second one being a list of the indices of lst that
                       correspond to the free indices of pattern
        otherwise: a pair (False, None)
    """
    # Check the index structure
    for (pos1, t1), (pos2, t2) in enum_product(pattern, pattern):
        for (i, i1), (j, i2) in enum_product(t1.indices, t2.indices):
            li1 = lst[pos1].indices[i]
            li2 = lst[pos2].indices[j]
            if not ((i1 == i2) == (li1 == li2)):
                return False, None

    # Find the indices of lst correponding to the free ones of pattern
    free_indices = [0] * sum([1 for t in pattern for i in t.indices if i < 0])
    for pos, tensor in enumerate(pattern):
        for i, index in enumerate(tensor.indices):
            if index < 0:
                free_indices[-index-1] = lst[pos].indices[i]
    return True, free_indices

class TensorBuilder(object):
    """
    Interface for the creation of constant tensors.

    To be used as follows. For the creation of a tensor, do:
    > name = TensorBuilder("name")
    To use it later inside an operator, do:
    > Op(..., name(ind1, ind2, ...), ...)

    Attributes:
        name (string): the identifier
    """
    def __init__(self, name):
        self.name = name

    def __call__(self, *indices):
        return Tensor(self.name, list(indices))

class FieldBuilder(object):
    """
    Interface for the creation of fields.

    To be used as follows. For the creation of a field, do:
    > name = FieldBuilder("name", dimension, statistics)
    To use it later inside an operator, do:
    > Op(..., name(ind1, ind2, ...), ...)

    Attributes:
        name (string): the identifier
        dimension (int): the energy (mass, inverse length) dimensions
        statistics (bool): True for bosons and False for fermion. 
                           The usage the variables boson and fermion
                           defined in this module is recommended
    """
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
    """
    Interface for the creation of tensors with derivatives applied.

    For ten an object of the classes TensorBuilder of FieldBuilder, the
    recommended notation to define a tensor with derivatives inside
    an operator is:
    > Op(..., D(index, ten(ind1, ind2, ...)), ...)
    """
    return Operator([tensor]).derivative(index).operators[0].tensors[0]

def Op(*tensors):
    """Interface for the creation of operators"""
    return Operator(list(tensors))

def OpSum(*operators):
    """Interface for the creation of operator sums"""
    if not operators:
        return OperatorSum()
    return OperatorSum(list(operators))

def number_op(number):
    """
    Create an operator correponding to a number.

    Square brackets around the name of a tensor mean that it should be 
    treated as a (complex) number.
    """
    return Operator([Tensor("[{}]".format(number), [])])

def symbol_op(symbol, exponent, indices=None):
    """
    Create an operator corresponding to a tensor exponentiated to some power.

    Curly braces around the name of a tensor mean that what's inside is of
    the form "base^exponent" and that it may be collected together with 
    others with the same base and indices, adding the exponents.
    """
    if indices is None:
        indices = []
    return Operator([Tensor("{{{}^{}}}".format(symbol, exponent), indices)])

def tensor_op(name):
      return Operator([Tensor(name, [])])

def flavor_tensor_op(name):
    """Interface for the creation of one-tensor operators with indices"""
    def f(*indices):
        return Op(Tensor(name, list(indices)))
    return f

# To be used to specify the statistics of fields
boson = True
fermion = False

# Basic Lorentz group related tensors.
#
# * The eps- tensors represent the antisymmetric epsilon symbols with
#   two-component spinor indices, with Up/Down denoting the position of
#   the two indices and the sufix -Dot meaning that both indices are dotted.
#
# * The sigma- tensors represent sigma matrices, defined in the usual way
#   using the three Pauli matrices and the 2x2 identity. The Lorentz vector
#   index is the first, the two two-component spinor indices are the second
#   and third.
epsUp = TensorBuilder("epsUp")
epsUpDot = TensorBuilder("epsUpDot")
epsDown = TensorBuilder("epsDown")
epsDownDot = TensorBuilder("epsDownDot")
sigma4bar = TensorBuilder("sigma4bar")
sigma4 = TensorBuilder("sigma4")

# Kronecker delta. To be replaced by the correponding
# index contraction appearing instead (module transformations)
kdelta = TensorBuilder("kdelta")

# Generic tensor to be used for intermediate steps in calculations
# and in the output of matching
generic = FieldBuilder("generic", 0, boson)


