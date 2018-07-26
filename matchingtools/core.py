"""
Core module with the definitions of the basic building blocks: the
classes :class:`Tensor`, :class:`Operator`, and :class:`OperatorSum`.
Implements the Leibniz rule for derivatives and the algorithms for
matching and replacing as well as functional derivatives.

Defines interfaces for the creation of tensors, fields, operators and
operator sums: :class:`TensorBuilder`, :class:`FieldBuilder`,
:func:`Op` and :func:`OpSum`; the interface for creating derivatives
of single tensors: the function :func:`D`; and interfaces for creating
special single-tensor associated to (complex) numbers and powers of
constants.

Defines the Lorentz tensors :data:`epsUp`, :data:`epsUpDot`,
:data:`epsDown`, :data:`epsDownDot`, :data:`sigma4` and
:data:`sigma4bar`.
"""

from abc import ABCMeta, abstractmethod
from collections import Counter
from enum import Enum
from functools import partial

import rules


class Statistics(Enum):
    BOSON = 1
    FERMION = 2


class Conjugable(object, metaclass=ABCMeta):
    is_conjugated = False

    @abstractmethod
    def conjugate(self):
        pass


class Differentiable(object, metaclass=ABCMeta):
    @abstractmethod
    def differentiate(self, index):
        pass


class Convertible(object, metaclass=ABCMeta):
    class DemoteError(Exception):
        def __init__(self, what, src, dest):
            error_msg = "Unable to demote {what} from {src} to {dest}"
            super().__init__(
                error_msg.format(
                    what=what,
                    src=src.__name__,
                    dest=dest.__name__
                )
            )

    @abstractmethod
    def _to_tensor(self):
        pass

    @abstractmethod
    def _to_operator(self):
        pass

    @abstractmethod
    def _to_operator_sum(self):
        pass


class RealMixin(object):
    def conjugate(self):
        return self


class ComplexMixin(object):
    def conjugate(self):
        conjugated = self.clone()
        conjugated.is_conjugated = not self.is_conjugated
        return conjugated


class Tensor(Conjugable, Convertible, Differentiable):
    """
    Basic building block for operators.

    A tensor might have some derivatives applied to it. The
    indices correponding to the derivatives are given by the first
    indices in the list of indices of the Tensor.

    Attributes:
        name (string): identifier
        indices (list of ints): indices of the tensor and the
                                derivatives applied to it
        dimension (int): energy dimensions
        statistics (Statistics): Either BOSON or FERMION
    """

    def __init__(
            self, name, indices, derivatives_indices,
            dimension=0, statistics=Statistics.BOSON
    ):
        self.name = name
        self.indices = indices
        self.derivatives_indices = derivatives_indices
        self.dimension = dimension
        self.statistics = statistics

    def __str__(self):
        """
        Returns a string of the form:
            D(di[0])D(di[1])...D(di[m-1])T(i[0], i[1], ..., i[n-1])
        for a Tensor with indices i[0], i[1], ..., i[n-1] and
        derivatives_indices di[0], di[1], ..., di[m-1]
        """
        derivatives_str = ''.join(
            'D({})'.format(index) for index in self.derivatives_indices
        )
        indices_str = ', '.join(map(str, self.indices))

        return '{derivatives}{name}({indices})'.format(
            derivatives=derivatives_str,
            name=self.name,
            indices=indices_str
        )

    __repr__ = __str__

    def __add__(self, other):
        return self._to_operator_sum() + other

    __radd__ = __add__

    def __mul__(self, other):
        return self._to_operator_sum() * other

    __rmul__ = __mul__

    def __eq__(self, other):
        if(not isinstance(other, Tensor)):
            return False

        return (
            self.name == other.name
            and self.indices == other.indices
            and self.derivatives_indices == other.derivatives_indices
            and self.dimension == other.dimension
            and self.statistics == other.statistics
        )

    def __hash__(self):
        return hash(self.name)

    def __contains__(self, index):
        return index in self.all_indices

    def _to_tensor(self):
        return self

    def _to_operator(self):
        return Operator([self])

    def _to_operator_sum(self):
        return self._to_operator()._to_operator_sum()

    def _replace_indices(self, indices_mapping):
        new_tensor = self.clone()
        new_tensor.indices = [
            indices_mapping.get(index, index) for index in self.indices
        ]
        new_tensor.derivatives_indices = [
            indices_mapping.get(index, index)
            for index in self.derivatives_indices
        ]

        return new_tensor

    @property
    def all_indices(self):
        return self.derivatives_indices + self.indices

    @abstractmethod
    def clone(self):
        pass

    @classmethod
    def make(cls, *names):
        def builder(name, indices):
            return cls(name=name, indices=indices, derivatives_indices=[])

        return [partial(builder, name) for name in names]


class Constant(Tensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.derivatives_indices = []

    def clone(self):
        return Constant(
            name=self.name,
            indices=[index for index in self.indices],
            dimension=self.dimension,
            statistics=self.statistics
        )

    def differentiate(self, index):
        return 0


class Field(Tensor):
    def clone(self):
        return Field(
            name=self.name,
            indices=[index for index in self.indices],
            derivatives_count=self.derivatives_count,
            dimension=self.dimension,
            statistics=self.statistics
        )

    def differentiate(self, index):
        diff = self.clone()
        diff.derivatives_count += 1
        diff.indices = [index] + diff.indices

        return diff


class RealConstant(RealMixin, Constant):
    pass


class RealField(RealMixin, Field):
    pass


class ComplexConstant(ComplexMixin, Constant):
    pass


class ComplexField(ComplexMixin, Field):
    pass


class Kdelta(RealConstant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            assert len(self.indices) == 2
        except AssertionError:
            raise ValueError(
                "A Kdelta tensor takes exactly 2 indices ({} given)".format(
                    len(self.indices)
                )
            )


class Operator(Conjugable, Convertible, Differentiable):
    """
    Container for a list of tensors with their indices contracted.

    The methods include the basic derivation, matching and replacing
    operations, as well as the implementation of functional derivatives.

    Attributes:
        coefficient (Number): the numeric coefficient of the term
        tensors ([Tensor]): list of the tensors contained
    """

    def __init__(self, tensors=None, coefficient=1):
        if tensors is None:
            tensors = []
        self.tensors = tensors
        self.coefficient = coefficient

        self._simplify()

    def __str__(self):
        tensors = " ".join(map(str, self.tensors))

        if self.coefficient == 1:
            return tensors

        return "{} {}".format(self.coefficient, tensors)

    __repr__ = __str__

    def __add__(self, other):
        return self._to_operator_sum() + other

    __radd__ = __add__

    def __mul__(self, other):
        return self._to_operator_sum() * other

    __rmul__ = __mul__

    def __neg__(self):
        return self * (-1)

    def __hash__(self):
        return hash(tuple(self.tensors))

    def _to_tensor(self):
        if len(self.tensors) == 1:
            return self.tensors[0]
        else:
            raise Convertible.DemoteError(self, Operator, Tensor)

    def _to_operator(self):
        return self

    def _to_operator_sum(self):
        return OperatorSum([self])

    def _simplify(self):
        if self.coefficient == 0:
            self.tensors = []

        kdeltas = []
        rest = []

        for tensor in self.tensors:
            if isinstance(tensor, Kdelta):
                kdeltas.append(tensor)
            else:
                rest.append(tensor)

        # TODO fix this shit
        for tensor in rest:
            for pos, index in enumerate(tensor.indices):
                for kdelta in kdeltas:
                    i, j = kdelta.indices
                    if index == i:
                        tensor.indices[pos] = j
                        kdeltas.remove(kdelta)
                    elif index == j:
                        tensor.indices[pos] = i
                        kdeltas.remove(kdelta)

    @property
    def dimension(self):
        return sum([
            tensor.dimension + tensor.derivatives_count
            for tensor in self.tensors
        ])

    def __contains__(self, tensor):
        return tensor in self.tensors

    def is_free_index(self, index):
        return len([
            1 for tensor in self.tensors for tensor_index in tensor.indices
            if tensor_index == index
        ]) == 1

    @property
    def free_indices(self):
        counter = Counter(
            index for tensor in self.tensors for index in tensor.indices
        )

        return [
            index for index, multiplicity in counter.items()
            if multiplicity == 1
        ]

    def clone(self):
        return Operator(
            [tensor.clone() for tensor in self.tensors],
            self.coefficient
        )

    def conjugate(self):
        return Operator([
            tensor.conjugate() for tensor in self.tensors
        ], coefficient=self.coefficient)

    def differentiate(self, index):
        acc = 0

        for pos, tensor in enumerate(self.tensors):
            left = Operator(self.tensors[:pos])
            diff = self.tensors[pos].differentiate(index)
            right = Operator(self.tensors[pos+1:])

            acc += left * diff * right

        return acc

    def variation(self, field_name, statistics):
        """
        Take functional derivative of the spacetime integral of self:

        Args:
            field_name (string): the name of the field with respect to
                                 which the functional derivative is taken
            statistics (Statistics): statistics of the field
        """

        raise NotImplementedError()
        result = OperatorSum()
        for pos, tensor in enumerate(self.tensors):
            if tensor.name == field_name:
                inside_op = self.remove_tensor(pos)
                der_inds = remove_indices(tensor.derivatives_indices,
                                          tensor.non_derivatives_indices)

                # Compute the sign taking into account the number of
                # derivatives that act on the field and the number of
                # fermions before it in the fermionic case
                minus_sign = len(der_inds) % 2 == 1

                if statistics == fermion:
                    number_of_fermions = len([1 for t in self.tensors[:pos]
                                              if not t.statistics])
                    minus_sign = minus_sign != (number_of_fermions % 2 == 1)
                if minus_sign:
                    inside_op *= number_op(-1)

                # Apply the derivatives with the correponding indices
                inside_ops = OperatorSum([inside_op])
                result += apply_derivatives(list(reversed(der_inds)),
                                            inside_ops)
        return result

    def __eq__(self, other):
        """
        Match self with other operator. All tensors and index contractions
        should match. No sign differences allowed. All free indices should
        be equal. Reorderings allowed.
        """
        if not isinstance(other, Convertible):
            return False

        try:
            other = other._to_operator()
        except Convertible.DemoteError:
            return False

        if abs(self.coefficient) != abs(other.coefficient):
            return False

        if len(self.tensors) != len(other.tensors):
            return False

        match = rules.Match.match_operators(self, other)

        if match is None:
            return False

        for index, associate in match.indices_mapping.items():
            if index != associate:
                return False

        sign = rules.Permutation.compare(
            self.tensors,
            [match.tensor_mapping[tensor] for tensor in self.tensors]
        ).parity

        return self.coeffient * sign == other.coefficient


class OperatorSum(Conjugable, Convertible, Differentiable):
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
        if not isinstance(other, Convertible):
            if other == 0:
                return self
            return self + OperatorSum([Operator([], coefficient=other)])

        if not isinstance(other, OperatorSum):
            return self + other._to_operator_sum()

        return OperatorSum(self.operators + other.operators)

    __radd__ = __add__

    def __mul__(self, other):
        if not isinstance(other, Convertible):
            if other == 0:
                return OperatorSum()

            if other == 1:
                return self

            return self * OperatorSum([Operator([], coefficient=other)])

        if not isinstance(other, OperatorSum):
            return self + other._to_operator_sum()

        return OperatorSum([
                Operator(
                    self_op.tensors + other_op.tensors,
                    coefficient=self_op.coefficient * other_op.coefficient
                )
                for self_op in self.operators
                for other_op in other.operators
        ])

    __rmul__ = __mul__

    def __neg__(self):
        return OperatorSum([-op for op in self.operators])

    def __eq__(self, other):
        if len(self.operators) != len(other.operators):
            return False

        return all(
            self_operator in other.operators
            for self_operator in self.operators
        )

    def _to_tensor(self):
        return self._to_operator()._to_tensor()

    def _to_operator(self):
        if len(self.operators) == 1:
            return self.operators[0]
        else:
            raise Convertible.DemoteError(self, OperatorSum, Operator)

    def _to_operator_sum(self):
        return self

    def _simplify(self):
        coefficients = {}

        for operator in self.operators:
            coefficients.setdefault(operator, 0)
            coefficients[operator] += operator.coefficient

        self.operators = [
            Operator(
                operator.tensors,
                coefficient
            )
            for operator, coefficient in coefficients.items()
        ]

    def conjugate(self):
        return OperatorSum([
            operator.conjugate() for operator in self.operators
        ])

    def differentiate(self, index):
        return OperatorSum([
            operator.differentiate(index) for operator in self.operators
        ])

    def variation(self, field_name, statistics):
        """
        Take functional derivative of the spacetime integral of self.

        Args:
            field_name (string): the name of the field with respect to
                                 which the functional derivative is taken
            statistics (bool): statistics of the field
        """
        return sum([
            operator.variation(field_name, statistics)
            for operator in self.operators
        ])


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
                                derivatives_count=tensor.derivatives_count + 1,
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


def TensorBuilder(name):
    def builder(*indices):
        """
        Creates a Tensor object with the given indices and default attributes
        """
        return Tensor(name, list(indices))

    return builder


def FieldBuilder(name, dimension, statistics):
    """
    Creates a Tensor object with the given indices and
    the following attributes: ``is_field = True``,
    ``derivatives_count = 0``
    """
    def builder(*indices):
        return Tensor(
            name=name, indices=list(indices), is_field=True,
            derivatives_count=0, dimension=dimension,
            statistics=statistics
        )

    return builder


def D_op(index, *tensors):
    """
    Interface for the creation of operator sums obtained from
    the application of derivatives to producs of tensors.
    """
    return Operator(list(tensors)).derivative(index)


def D(index, tensor):
    """
    Interface for the creation of tensors with derivatives applied.
    """
    return Operator([tensor]).derivative(index).operators[0].tensors[0]


def Op(*tensors):
    """Interface for the creation of operators"""
    return Operator(list(tensors))


def OpSum(*operators):
    """Interface for the creation of operator sums"""
    return OperatorSum(list(operators))


def number_op(number):
    """
    Create an operator correponding to a number.
    """
    return Op(Tensor("$number", [], content=number, exponent=1))


i_op = Op(Tensor("$i", [], exponent=1))
"""Operator representing the imaginary unit."""


def power_op(name, exponent, indices=None):
    """
    Create an operator corresponding to a tensor exponentiated to some power.
    """
    if indices is None:
        indices = []
    return Operator([Tensor(name, indices, exponent=exponent)])


def tensor_op(name, indices=None):
    if indices is None:
        indices = []
    return Operator([Tensor(name, indices)])


def flavor_tensor_op(name):
    """Interface for the creation of one-tensor operators with indices"""
    def f(*indices):
        return Op(Tensor(name, list(indices)))
    return f


# To be used to specify the statistics of fields

boson = Statistics.BOSON
fermion = Statistics.FERMION

kdelta = TensorBuilder("kdelta")
"""
Kronecker delta. To be replaced by the correponding
index contraction appearing instead (module transformations).
"""
generic = FieldBuilder("generic", 0, boson)
"""
Generic tensor to be used for intermediate steps in calculations
and in the output of matching.
"""

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
"""
Totally anti-symmetric tensor with two two-component spinor
 undotted superindices
"""

epsUpDot = TensorBuilder("epsUpDot")
"""
Totally anti-symmetric tensor with two two-component spinor
 dotted superindices
"""

epsDown = TensorBuilder("epsDown")
"""
Totally anti-symmetric tensor with two two-component spinor
 undotted subindices
"""

epsDownDot = TensorBuilder("epsDownDot")
"""
Totally anti-symmetric tensor with two two-component spinor
 dotted subindices
"""

sigma4bar = TensorBuilder("sigma4bar")
"""
Tensor with one lorentz index, one two-component spinor dotted
superindex and one two-component spinor undotted superindex.
Represents the four-vector of 2x2 matrices built out identity
and minus the Pauli matrices.
"""

sigma4 = TensorBuilder("sigma4")
"""
Tensor with one lorentz index, one two-component spinor undotted
subindex and one two-component spinor dotted subindex.
Represents the four-vector of 2x2 matrices built out identity
and the Pauli matrices.
"""
