"""
Core module with the definitions of the basic building blocks: the
classes :class:`Tensor`, :class:`Operator`, and :class:`OperatorSum`.
Implements the Leibniz rule for derivatives and the algorithms for
matching and replacing as well as functional derivatives.

Defines the Lorentz tensors :data:`epsUp`, :data:`epsUpDot`,
:data:`epsDown`, :data:`epsDownDot`, :data:`sigma4` and
:data:`sigma4bar`.
"""

from abc import ABCMeta, abstractmethod
from collections import Counter
from enum import Enum
from fractions import Fraction
from itertools import permutations
from operator import add, __eq__

from matchingtools.lsttools import LookUpTable
from matchingtools.matches import Match
from matchingtools.utils import Permutation


class Statistics(Enum):
    BOSON = 1
    FERMION = 2


class Conjugable(object, metaclass=ABCMeta):
    @abstractmethod
    def conjugate(self):
        pass


class Differentiable(object, metaclass=ABCMeta):
    @abstractmethod
    def differentiate(self, index):
        pass

    def nth_derivative(self, indices):
        differentiated = self
        # TODO: decide whether to use indices of reversed(indices)
        for index in indices:
            differentiated = differentiated.differentiate(index)

        return differentiated


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


class Tensor(Conjugable, Convertible, Differentiable):
    """
    Basic building block for operators.

    A tensor might have some derivatives applied to it. The
    indices correponding to the derivatives are given by the first
    indices in the list of indices of the Tensor.

    Attributes:
        name (string): identifier
        indices (list of Index): indices of the tensor
        derivatives_indices (list of Index): indices of its derivatives
        dimension (float): energy dimensions
        statistics (Statistics): Either BOSON or FERMION
    """

    def __init__(
            self, name, indices, derivatives_indices,
            dimension=0, statistics=Statistics.BOSON,
            is_conjugated=False
    ):
        self.name = name
        self.indices = indices
        self.derivatives_indices = derivatives_indices
        self._tensor_dimension = Fraction(round(2 * dimension), 2)
        self.statistics = statistics
        self.is_conjugated = is_conjugated

    @property
    def dimension(self):
        return self._tensor_dimension + len(self.derivatives_indices)

    def __str__(self):
        """
        Returns a string of the form:
            D(di[0])D(di[1])...D(di[m-1])T(i[0], i[1], ..., i[n-1])
        for a Tensor with indices i[0], i[1], ..., i[n-1] and
        derivatives_indices di[0], di[1], ..., di[m-1]
        """
        derivatives_str = ''.join(
            'D({})'.format(index)
            for index in reversed(self.derivatives_indices)
        )
        indices_str = ', '.join(map(str, self.indices))
        conjugate_str = '.c' if self.is_conjugated else ''

        return '{derivatives}{name}{conjugate}({indices})'.format(
            derivatives=derivatives_str,
            name=self.name,
            conjugate=conjugate_str,
            indices=indices_str
        )

    __repr__ = __str__

    def __add__(self, other):
        return self._to_operator_sum() + other

    __radd__ = __add__

    def __neg__(self):
        return -self._to_operator()

    def __mul__(self, other):
        return self._to_operator_sum() * other

    __rmul__ = __mul__

    def __eq__(self, other):
        return (
            self.does_match(other)
            and self.indices == other.indices
            and self.derivatives_indices == other.derivatives_indices
            and self._tensor_dimension == other._tensor_dimension
            and self.dimension == other.dimension
            and self.statistics == other.statistics
        )

    def does_match(self, other):
        if not isinstance(other, Tensor):
            return False

        return (
            self.name == other.name
            and self.dimension == other.dimension
            and self.statistics == other.statistics
            and self.is_conjugated == other.is_conjugated
            and len(self.indices) == len(other.indices)
            and len(self.derivatives_indices) == len(other.derivatives_indices)
            and isinstance(self, Constant) == isinstance(other, Constant)
            and isinstance(self, RealMixin) == isinstance(other, RealMixin)
        )

    def __hash__(self):
        return hash(self.name)

    def __contains__(self, index):
        return index in self.all_indices

    def _to_tensor(self):
        return self

    def _to_operator(self):
        return Operator(tensors=[self])

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

    def clone(self):
        # TODO: .copy() has been included for indices lists. Is it necessary?
        return type(self)(
            name=self.name,
            indices=self.indices.copy(),
            derivatives_indices=self.derivatives_indices.copy(),
            dimension=self._tensor_dimension,
            statistics=self.statistics,
            is_conjugated=self.is_conjugated
        )

    @classmethod
    def make(cls, *names, **kwargs):
        return [Builder(name, cls, kwargs) for name in names]


class Constant(Tensor):
    # TODO: enforce derivatives_indices==[]?
    def differentiate(self, index):
        return 0


class Field(Tensor):
    def differentiate(self, index):
        diff = self.clone()
        diff.derivative_indices.append(index)

        return diff


class RealMixin(object):
    def conjugate(self):
        return self


class ComplexMixin(object):  # TODO: inherit from Tensor?
    def conjugate(self):
        conjugated = self.clone()
        conjugated.is_conjugated = not self.is_conjugated
        return conjugated


class RealConstant(RealMixin, Constant):
    pass


class RealField(RealMixin, Field):
    pass


class ComplexConstant(ComplexMixin, Constant):
    pass


class ComplexField(ComplexMixin, Field):
    pass


class Kdelta(RealConstant):
    def __init__(self, *indices):
        indices_list = list(indices)
        indices_count = len(indices_list)

        if indices_count != 2:
            raise ValueError(
                "A Kdelta tensor takes exactly 2 indices ({} given)".format(
                    indices_count
                )
            )

        super().__init__(
            name="Kdelta",
            indices=list(indices),
            derivatives_indices=[],
            dimension=0,
            statistics=Statistics.BOSON
        )

    def clone(self):
        return Kdelta(*self.indices)


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
            # TODO: understand this, sometimes:
            # type(tensor) == Kdelta AND isinstance(tensor, Kdelta) == False
            if tensor.name == "Kdelta":
                kdeltas.append(tensor)
            else:
                rest.append(tensor.clone())

        # TODO: Refactor this?
        remaining_kdeltas = []
        for pos, kdelta in enumerate(kdeltas):
            found_index = False
            for tensor in kdeltas[pos+1:] + rest:
                for pos, index in enumerate(tensor.indices):
                    i, j = kdelta.indices
                    if index == i:
                        tensor.indices[pos] = j
                        found_index = True
                        break
                    elif index == j:
                        tensor.indices[pos] = i
                        found_index = True
                        break
                if found_index:
                    break
            if not found_index:
                remaining_kdeltas.append(kdelta)

        self.tensors = rest + remaining_kdeltas

    @property
    def dimension(self):
        return sum([tensor.dimension for tensor in self.tensors])

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
        return Operator(
            [tensor.conjugate() for tensor in self.tensors],
            coefficient=self.coefficient.conjugate()
        )

    def differentiate(self, index):
        acc = 0

        for pos, tensor in enumerate(self.tensors):
            left = Operator(self.tensors[:pos])
            diff = self.tensors[pos].differentiate(index)
            right = Operator(self.tensors[pos+1:])

            acc += left * diff * right

        return acc

    def __eq__(self, other):
        """
        Match self with other operator. All tensors and index contractions
        should match. No sign differences allowed. All free indices should
        be equal. Reorderings allowed.
        """

        # TODO: understand this, sometimes:
        # type(other) == Operator AND isinstance(other, Convertible) == False
        # ---
        # When this is solved, uncomment the following two lines:
        #   if not isinstance(other, Convertible):
        #     return False

        try:
            other = other._to_operator()
        except Convertible.DemoteError:
            return False

        if abs(self.coefficient) != abs(other.coefficient):
            return False

        if len(self.tensors) != len(other.tensors):
            return False

        matches = Match.all_matches(self, other)

        if len(matches) == 0:
            return False

        # TODO: make sure this previous code isn't the right solution:
        #   for index, associate in match.indices_mapping.items():
        #     if index != associate:
        #       return False
        # ---
        # it's been substituted by this:
        for match in matches:
            free_indices_coincide = True
            for tensor in self.tensors:
                for index in tensor.indices:
                    is_free = self.is_free_index(index)
                    if is_free and index in match.indices_mapping:
                        if index != match.indices_mapping[index]:
                            free_indices_coincide = False
            if free_indices_coincide:
                break
        else:
            return False            

        own_fermions = [
            tensor for tensor in self.tensors
            if tensor.statistics == Statistics.FERMION
        ]
        sign = Permutation.compare(
            own_fermions,
            [match.tensors_mapping[tensor] for tensor in own_fermions]
        ).parity

        return self.coefficient * sign == other.coefficient

    def with_unit_coefficient(self):
        new_operator = self.clone()
        new_operator.coefficient = 1
        return new_operator


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
                
        self.operators = [operator._to_operator() for operator in operators]

        self._simplify()

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

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if not isinstance(other, Convertible):
            if other == 0:
                return OperatorSum()

            if other == 1:
                return self

            return self * OperatorSum([Operator([], coefficient=other)])

        if not isinstance(other, OperatorSum):
            return self * other._to_operator_sum()

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
        return sum(-op for op in self.operators)

    def __eq__(self, other):
        if len(self.operators) != len(other.operators):
            return False

        # TODO: is there a more efficient version of this?
        return any(
            all(map(__eq__, self.operators, other_permutation))
            for other_permutation in permutations(other.operators)
        )
        # Does the following work always?
        #   return (
        #       all(op in other.operators for op in self.operators)
        #       and
        #       all(op in self.operators for op in other.operators)
        #   )

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
        coefficients = LookUpTable()

        for operator in self.operators:
            coefficients.update(
                operator.with_unit_coefficient(),
                operator.coefficient,
                add
            )

        self.operators = [
            Operator(
                operator.tensors,
                coefficient
            )
            for operator, coefficient in coefficients.items
        ]

    def conjugate(self):
        return sum(
            operator.conjugate() for operator in self.operators
        )

    def differentiate(self, index):
        return sum(
            operator.differentiate(index) for operator in self.operators
        )

    def filter_by_max_dimension(self, max_dimension):
        return OperatorSum([
            operator for operator in self.operators
            if operator.dimension <= max_dimension
        ])


def D(index, tensor):
    """
    Interface for the creation of tensors with derivatives applied.
    """
    return tensor.differentiate(index)


class Builder(object):
    def __init__(self, name, cls, kwargs):
        self.name = name
        self.cls = cls
        self.kwargs = kwargs

    def __call__(self, *indices):
        return self.cls(
            name=self.name,
            indices=list(indices),
            derivatives_indices=[],
            **self.kwargs
        )

    def c(self, *indices):
        return self(*indices).conjugate()


# To be used to specify the statistics of fields

# TODO: decide whether we want to keep these aliases or not

boson = Statistics.BOSON
fermion = Statistics.FERMION

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

# TODO: Rewrite these
'''
epsUp = TensorBuilder("epsUp")
epsUpDot = TensorBuilder("epsUpDot")
epsDown = TensorBuilder("epsDown")
epsDownDot = TensorBuilder("epsDownDot")
sigma4bar = TensorBuilder("sigma4bar")
sigma4 = TensorBuilder("sigma4")
'''
