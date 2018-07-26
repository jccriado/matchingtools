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
from enum import Enum
from collections import Counter

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
        indices (list of Index): indices of the tensor
        derivatives_indices (list of Index): indices of its derivatives
        dimension (float): energy dimensions
        statistics (Statistics): Either BOSON or FERMION
    """

    def __init__(
            self, name, indices, derivatives_indices,
            _tensor_dimension=0, statistics=Statistics.BOSON
    ):
        self.name = name
        self.indices = indices
        self.derivatives_indices = derivatives_indices
        self._tensor_dimension = _tensor_dimension
        self.statistics = statistics

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

    @abstractmethod
    def clone(self):
        pass


class Constant(Tensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.derivatives_indices = []

    def clone(self):
        return Constant(
            name=self.name,
            indices=self.indices.copy(),
            # TODO: What is derivatives_indices here?
            derivatives_indices=[],
            dimension=self.dimension,
            statistics=self.statistics
        )

    def differentiate(self, index):
        return 0


class Field(Tensor):
    def clone(self):
        return Field(
            name=self.name,
            indices=self.indices.copy()
            derivatives_indices=self.derivatives_indices.copy()
            dimension=self.dimension,
            statistics=self.statistics
        )

    def differentiate(self, index):
        diff = self.clone()
        diff.derivative_indices.append(index)

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

        for tensor in rest:
            for pos, index in enumerate(tensor.indices):
                for kdelta in kdeltas.copy():
                    i, j = kdelta.indices
                    if index == i:
                        tensor.indices[pos] = j
                        kdeltas.remove(kdelta)
                    elif index == j:
                        tensor.indices[pos] = i
                        kdeltas.remove(kdelta)

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
            coefficient=self.coefficient
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

        # TODO: Do want to generate new bound indices for other_op to avoid clashes?
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
