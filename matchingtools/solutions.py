from operator import mul
from functools import reduce

from matchingtools.indices import Index
from matchingtools.matches import Match
from matchingtools.core import OperatorSum
from matchingtools.rules import Rule
from matchingtools.invertibles import InvertibleMatrix, InvertibleScalar


class LinearTerm(object):
    def __init__(
            self, field, invertible_matrices, invertible_scalars, coefficient
    ):
        self.field = field
        self.invertible_matrices = invertible_matrices
        self.invertible_scalars = invertible_scalars
        self.coefficient = coefficient

    @staticmethod
    def split(operator, fields):
        found_field = False
        invertible_matrices = []
        invertible_scalars = []

        for tensor in operator.tensors:
            if any(Match.tensors_do_match(tensor, field) for field in fields):
                if found_field:
                    return None
                else:
                    found_field = True
                    return_field = tensor
            elif isinstance(tensor, InvertibleMatrix):
                invertible_matrices.append(tensor)
            elif isinstance(tensor, InvertibleScalar):
                invertible_scalars.append(tensor)
            else:
                return None

        if not found_field:
            return None

        return LinearTerm(
            return_field,
            invertible_matrices,
            invertible_scalars,
            operator.coefficient
        )


class Equation(object):
    def __init__(self, equation, unknowns):
        self.equation = equation
        self.unknowns = unknowns

    def split_linear(self):
        found_linear_term = False
        rest = OperatorSum()

        for pos, operator in enumerate(self.equation.operators):
            linear_term_splitting = LinearTerm.split(operator, self.unknowns)
            if linear_term_splitting is None:
                rest += operator
            else:
                if found_linear_term:
                    return None
                else:
                    found_linear_term = True
                    linear_term = linear_term_splitting

        if not found_linear_term:
            return None

        return linear_term, rest

    def solve(self):
        splitting = self.split_linear()

        if splitting is None:
            # TODO: improve
            raise Exception("Can't solve equation: " + str(self.equation))

        linear_term, rest = splitting

        factor_scalars = reduce(
            mul,
            (scalar.inverse_scalar()
             for scalar in linear_term.invertible_scalars),
            1
        )

        new_indices_mapping = {
            index: Index(index.name)
            for index in linear_term.field.indices
        }

        factor_matrices = reduce(
            mul,
            (matrix.inverse_matrix(
                new_indices_mapping[matrix.indices[1]],
                matrix.indices[0])
             for matrix in linear_term.invertible_matrices),
            1
        )

        return EquationSolution(
            linear_term.field._replace_indices(new_indices_mapping),
            -(
                1/linear_term.coefficient
                * factor_scalars
                * factor_matrices
                * rest._replace_indices(new_indices_mapping)
            )
        )


class System(object):
    def __init__(self, equations, unknowns):
        self.equations = equations
        self.unknowns = unknowns

    def solve(self, max_dimension):
        pre_solution = SystemSolution([
            Equation(equation, self.unknowns).solve()
            for equation in self.equations
        ])
        return pre_solution.solve(max_dimension)


class EquationSolution(object):
    def __init__(self, field, replacement):
        self.field = field
        self.replacement = replacement

    def __str__(self):
        return str("({} := {})".format(self.field, self.replacement))

    __repr__ = __str__

    def _rule(self, tensor):
        """
        A rule for substituting any tensor by the nth derivative of the
        replacement. The indices of these derivatives are given by the
        indices of the derivatives of the tensor.
        """
        new_indices_mapping = {
            old_index: Index(old_index.name)
            for old_index in tensor.derivatives_indices
        }
        field = self.field._replace_indices(new_indices_mapping)
        replacement = self.replacement._replace_indices(new_indices_mapping)

        if tensor.is_conjugated == self.field.is_conjugated:
            return Rule(
                field.nth_derivative(tensor.derivatives_indices),
                replacement.nth_derivative(tensor.derivatives_indices)
            )
        else:
            return Rule(
                field.nth_derivative(
                    tensor.derivatives_indices
                ).conjugate(),

                replacement.nth_derivative(
                    tensor.derivatives_indices
                ).conjugate()
            )

    def _substitute_in_operator(self, operator):
        """
        Replace the first tensor with name self.field_name by the nth
        derivative of the replacement.
        """
        for tensor in operator.tensors:
            if tensor.name == self.field.name:
                return self._rule(tensor).apply(operator)
        return operator._to_operator_sum()

    def substitute_once(self, target, max_dimension):
        """ Substitute once in each operator of target. """
        return sum(
            (
                self._substitute_in_operator(operator)
                .filter_by_max_dimension(max_dimension)
                for operator in target._to_operator_sum().operators
            ),
            OperatorSum()
        )

    def is_in(self, target):
        """
        Find whether there is a tensor in target with name self.field_name.
        """
        for operator in target.operators:
            for tensor in operator.tensors:
                if tensor.name == self.field.name:
                    return True

        return False


class SystemSolution(object):
    def __init__(self, solutions):
        self.solutions = solutions

    def __str__(self):
        return str(self.solutions)

    __repr__ = __str__

    def substitute(self, target, max_dimension):
        """
        Substitute everywhere iteratively, until there are not any
        remaining substitutions to be made
        """
        for solution in self.solutions:
            target = solution.substitute_once(target, max_dimension)

        if any(solution.is_in(target) for solution in self.solutions):
            return self.substitute(target, max_dimension)

        return target

    def solve(self, max_dimension):
        """
        Converts a system of solutions and to a system in which they have
        been used to remove any occurrence of their field names in their
        replacements.
        """
        new_solution = SystemSolution([
            EquationSolution(
                target_solution.field,
                self.substitute(target_solution.replacement, max_dimension)
                .filter_by_max_dimension(max_dimension)
            )
            for target_solution in self.solutions
        ])

        if any(solution.is_in(target_solution.replacement)
               for solution in self.solutions
               for target_solution in self.solutions):
            return new_solution.solve(max_dimension)

        return new_solution
