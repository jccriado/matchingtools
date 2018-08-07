from abc import ABCMeta, abstractmethod
from itertools import combinations

from matchingtools.core import (
    RealConstant, Operator, OperatorSum,
    TotallySymmetricMixin, TotallyAntiSymmetricMixin
)
from matchingtools.utils import merge_dicts


class InvertibleMatrix(object, metaclass=ABCMeta):
    @abstractmethod
    def inverse_matrix(self, first_index, second_index):
        pass


class InvertibleScalar(object, metaclass=ABCMeta):
    @abstractmethod
    def inverse_scalar(self):
        pass


class MassMatrix(TotallySymmetricMixin, RealConstant, InvertibleMatrix):
    def __init__(self, field_name, first_index, second_index, exponent=1):
        super().__init__(
            name="M_"+field_name+"^"+str(exponent),
            indices=[first_index, second_index],
            derivatives_indices=[]
        )

        self.field_name = field_name
        self.exponent = exponent

    def clone(self):
        return MassMatrix(
            field_name=self.field_name,
            first_index=self.indices[0],
            second_index=self.indices[1],
            exponent=self.exponent
        )

    def inverse_matrix(self, first_index, second_index):
        return MassMatrix(
            field_name=self.field_name,
            first_index=first_index,
            second_index=second_index,
            exponent=-self.exponent
        )

    def __pow__(self, number):
        new_mass = self.clone()
        new_mass.exponent *= number
        return new_mass

    def _match_attributes(self):
        return merge_dicts(super()._match_attributes(), {
            'exponent': self.exponent,
            'field_name': self.field_name
        })

    @staticmethod
    def _simplify(convertible):
        return OperatorSum([
            MassMatrix._simplify_product(operator)
            for operator in convertible._to_operator_sum().operators
        ])

    def shares_index(self, other):
        return any(own_index in other.indices for own_index in self.indices)

    def free_index_pair(self, other):
        for index in self.indices:
            if index not in other.indices:
                first = index
                break
        for index in other.indices:
            if index not in self.indices:
                second = index
                break
        return [first, second]

    @staticmethod
    def _simplify_product(operator):
        masses = [
            tensor for tensor in operator.tensors
            if isinstance(tensor, MassMatrix)
        ]
        rest = [
            tensor for tensor in operator.tensors
            if not isinstance(tensor, MassMatrix)
        ]

        while masses:
            for mass_1, mass_2 in combinations(masses, 2):
                names_match = mass_1.field_name == mass_2.field_name
                index_shared = mass_1.shares_index(mass_2)
                if (names_match and index_shared):
                    first_index, second_index = mass_1.free_index_pair(mass_2)
                    masses.append(
                        MassMatrix(
                            field_name=mass_1.field_name,
                            first_index=first_index,
                            second_index=second_index,
                            exponent=mass_1.exponent+mass_2.exponent
                        )
                    )
                    masses.remove(mass_1)
                    masses.remove(mass_2)
                    break
            else:
                break

        return Operator(masses + rest, operator.coefficient)


class MassScalar(RealConstant, InvertibleScalar):
    def __init__(self, field_name, exponent=1):
        super().__init__(
            name="M_"+field_name+"^"+str(exponent),
            indices=[],
            derivatives_indices=[]
        )

        self.field_name = field_name
        self.exponent = exponent

    def clone(self):
        return MassScalar(
            field_name=self.field_name,
            exponent=self.exponent
        )

    def inverse_scalar(self):
        return MassScalar(
            field_name=self.field_name,
            exponent=-self.exponent
        )

    def __pow__(self, number):
        new_mass = self.clone()
        new_mass.exponent *= number
        return new_mass

    def _match_attributes(self):
        return merge_dicts(super()._match_attributes(), {
            'exponent': self.exponent,
            'field_name': self.field_name
        })

    @staticmethod
    def _simplify(convertible):
        return OperatorSum([
            MassScalar._simplify_product(operator)
            for operator in convertible._to_operator_sum().operators
        ])

    @staticmethod
    def _simplify_product(operator):
        masses = [
            tensor for tensor in operator.tensors
            if isinstance(tensor, MassScalar)
        ]
        rest = [
            tensor for tensor in operator.tensors
            if not isinstance(tensor, MassScalar)
        ]

        while masses:
            for mass_1, mass_2 in combinations(masses, 2):
                if (mass_1.field_name == mass_2.field_name):
                    masses.append(
                        MassScalar(
                            field_name=mass_1.field_name,
                            exponent=mass_1.exponent+mass_2.exponent
                        )
                    )
                    masses.remove(mass_1)
                    masses.remove(mass_2)
                    break
            else:
                break

        return Operator(masses + rest, operator.coefficient)


class EpsilonUp(TotallySymmetricMixin, RealConstant, InvertibleMatrix):
    def __init__(self, first_index, second_index):
        super().__init__(
            name="EpsilonUp",
            indices=[first_index, second_index]
        )

    def clone(self):
        return EpsilonUp(*self.indices)

    def inverse_matrix(self, first_index, second_index):
        return EpsilonDown(first_index, second_index)


class EpsilonDown(TotallyAntiSymmetricMixin, RealConstant, InvertibleMatrix):
    def __init__(self, first_index, second_index):
        super().__init__(
            name="EpsilonDown",
            indices=[first_index, second_index]
        )

    def clone(self):
        return EpsilonDown(*self.indices)

    def inverse_matrix(self, first_index, second_index):
        return EpsilonUp(first_index, second_index)
