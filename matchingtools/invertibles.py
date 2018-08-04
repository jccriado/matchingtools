from abc import ABCMeta, abstractmethod

from matchingtools.core import RealConstant


class InvertibleMatrix(object, metaclass=ABCMeta):
    @abstractmethod
    def inverse_matrix(self, first_index, second_index):
        pass


class InvertibleScalar(object, metaclass=ABCMeta):
    @abstractmethod
    def inverse_scalar(self):
        pass


class MassMatrix(RealConstant, InvertibleMatrix):
    def __init__(self, field_name, first_index, second_index, exponent=1):
        super().__init__(
            name="M_"+field_name+"^"+self.exponent,
            indices=[first_index, second_index]
        )

        self.field_name = field_name
        self.exponent = 1

    def clone(self):
        return MassMatrix(
            field_name=self.field_name,
            indices=self.indices,
            exponent=self.exponent
        )

    def inverse_matrix(self, first_index, second_index):
        return MassMatrix(
            field_name=self.field_name,
            indices=[first_index, second_index],
            exponent=-self.exponent
        )


class MassScalar(RealConstant, InvertibleScalar):
    def __init__(self, field_name, exponent=1):
        super().__init__(
            name="M_"+field_name+"^"+self.exponent,
            indices=[]
        )

        self.field_name = field_name
        self.exponent = 1

    def clone(self):
        return MassScalar(
            field_name=self.field_name,
            exponent=self.exponent
        )

    def inverse_scalar(self, first_index, second_index):
        return MassScalar(
            field_name=self.field_name,
            indices=[first_index, second_index],
            exponent=-self.exponent
        )


class EpsilonUp(RealConstant, InvertibleMatrix):
    def __init__(self, first_index, second_index):
        super().__init__(
            name="EpsilonUp",
            indices=[first_index, second_index]
        )

    def clone(self):
        return EpsilonUp(*self.indices)

    def inverse_matrix(self, first_index, second_index):
        return EpsilonDown(first_index, second_index)


class EpsilonDown(RealConstant, InvertibleMatrix):
    def __init__(self, first_index, second_index):
        super().__init__(
            name="EpsilonDown",
            indices=[first_index, second_index]
        )

    def clone(self):
        return EpsilonDown(*self.indices)

    def inverse_matrix(self, first_index, second_index):
        return EpsilonUp(first_index, second_index)
