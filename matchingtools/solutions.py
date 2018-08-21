from operator import mul
from functools import reduce

from matchingtools.indices import Index
from matchingtools.matches import Match
from matchingtools.core import Field, OperatorSum
from matchingtools.replacements import Replacement
from matchingtools.invertibles import InvertibleMatrix, InvertibleScalar


class Equation(object):
    class LinearInvertibleTerm(object):
        class NonInvertibleError(Exception):
            def __init__(self, operator, non_invertible_factor):
                error_msg = (
                    "Unable to indentify invertible term: "
                    + "non-invertible factor {} in operator {}"
                )
                super().__init__(
                    error_msg.format(non_invertible_factor, operator)
                )

        class NonLinearError(Exception):
            def __init__(self, operator):
                error_msg = "Unable to identify linear term: {}"
                super().__init__(
                    error_msg.format(operator)
                )

        NonLinearInvertibleError = (NonLinearError, NonInvertibleError)

        def __init__(
            self, field, invertible_matrices, invertible_scalars, coefficient
        ):
            """
            A LinearInvertibleTerm represents an operator of the form:
            coefficient
            * invertible_matrix_1 * invertible_matrix_2 * ...
            * invertible_scalar_1 * invertible_matrix_2 * ...
            * field
            """

            self.field = field
            self.invertible_matrices = invertible_matrices
            self.invertible_scalars = invertible_scalars
            self.coefficient = coefficient

        @staticmethod
        def from_operator(operator, valid_fields):
            """
            Split an operator into a LinearInvertibleTerm whose field is one
            of the valid_fields.
            """

            found_field = False
            invertible_matrices = []
            invertible_scalars = []

            for tensor in operator.tensors:
                is_valid = any(
                    Match.tensors_do_match(tensor, field)
                    for field in valid_fields
                )

                if is_valid:
                    if found_field:
                        raise (
                            Equation.LinearInvertibleTerm
                            .NonLinearError(operator)
                        )
                    else:
                        found_field = True
                        return_field = tensor
                elif isinstance(tensor, InvertibleMatrix):
                    invertible_matrices.append(tensor)
                elif isinstance(tensor, InvertibleScalar):
                    invertible_scalars.append(tensor)
                elif isinstance(tensor, Field):
                    raise (
                        Equation.LinearInvertibleTerm
                        .NonLinearError(operator)
                    )
                else:
                    raise (
                        Equation.LinearInvertibleTerm
                        .NonInvertibleError(operator, tensor)
                    )

            if not found_field:
                raise Equation.LinearInvertibleTerm.NonLinearError(operator)

            return Equation.LinearInvertibleTerm(
                return_field,
                invertible_matrices,
                invertible_scalars,
                operator.coefficient
            )

    class NoLinearTermsError(Exception):
        def __init__(self, equation):
            error_msg = (
                "Unable find a 'linear invertible term' in equation {}\n" +
                "with unknowns {}."
            )
            super().__init__(
                error_msg.format(equation.equation, equation.unknowns)
            )

    class MultipleLinearTermsError(Exception):
        def __init__(self, equation):
            error_msg = (
                "Found several 'linear invertible terms' in equation {}\n" +
                "with unknowns {}."
            )
            super().__init__(
                error_msg.format(equation.equation, equation.unknowns)
            )

    def __init__(self, equation, unknowns):
        self.equation = equation
        self.unknowns = unknowns

    def find_linear(self):
        """ Look for a unique 'linear invertible term' in self.equation. """

        found_linear_term = False
        rest = OperatorSum()

        for pos, operator in enumerate(self.equation.operators):
            try:
                linear_term = (
                    Equation.LinearInvertibleTerm
                    .from_operator(operator, self.unknowns)
                )

                if found_linear_term:
                    raise Equation.MultipleLinearTermsError(self)
                else:
                    found_linear_term = True
            except Equation.LinearInvertibleTerm.NonLinearInvertibleError:
                rest += operator

        if not found_linear_term:
            raise Equation.NoLinearTermsError(self)

        return linear_term, rest

    def solve(self):
        linear_term, rest = self.find_linear()

        scalars_product = reduce(
            mul,
            (scalar.inverse_scalar()
             for scalar in linear_term.invertible_scalars),
            1
        )

        new_indices_mapping = {
            index: Index(index.name)
            for index in linear_term.field.indices
        }

        matrices_product = reduce(
            mul,
            (matrix.inverse_matrix(
                new_indices_mapping[matrix.indices[1]],
                matrix.indices[0])
             for matrix in linear_term.invertible_matrices),
            1
        )

        return Replacement(
            linear_term.field._replace_indices(new_indices_mapping),
            -(
                1/linear_term.coefficient
                * scalars_product
                * matrices_product
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


class SystemSolution(object):
    def __init__(self, replacements):
        self.replacements = replacements

    def __str__(self):
        return str(self.replacements)

    __repr__ = __str__

    def replace(self, target, max_dimension):
        """
        Substitute everywhere iteratively, until there are not any
        remaining substitutions to be made
        """
        for replacement in self.replacements:
            target = replacement.replace_all_occurrences(target, max_dimension)

        if any(replacement.is_in(target) for replacement in self.replacements):
            return self.replace(target, max_dimension)

        return target

    def solve(self, max_dimension):
        """
        Converts a system of solutions and to a system in which they have
        been used to remove any occurrence of their field names in their
        replacements.
        """
        new_replacements = SystemSolution([
            Replacement(
                target_replacement.tensor,
                self.replace(target_replacement.replacement, max_dimension)
                .filter_by_max_dimension(max_dimension)
            )
            for target_replacement in self.replacements
        ])

        if any(replacement.is_in(target_replacement.replacement)
               for replacement in self.replacements
               for target_replacement in self.replacements):
            return new_replacements.solve(max_dimension)

        return new_replacements
