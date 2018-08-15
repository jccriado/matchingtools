from operator import mul
from functools import reduce

from matchingtools.indices import Index
from matchingtools.matches import Match
from matchingtools.core import Field, OperatorSum
from matchingtools.rules import Rule
from matchingtools.invertibles import InvertibleMatrix, InvertibleScalar


class Equation(object):
    class LinearInvertibleTerm(object):
        class NonInvertible(Exception):
            def __init__(self, operator, non_invertible_factor):
                error_msg = "Non-invertible factor {} in operator {}"
                super().__init__(
                    error_msg.format(non_invertible_factor, operator)
                )

        class NonLinear(Exception):
            def __init__(self, operator):
                error_msg = "Non-linear operator {}"
                super().__init__(
                    error_msg.format(operator)
                )

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
                        raise Equation.LinearInvertibleTerm.NonLinear(operator)
                    else:
                        found_field = True
                        return_field = tensor
                elif isinstance(tensor, InvertibleMatrix):
                    invertible_matrices.append(tensor)
                elif isinstance(tensor, InvertibleScalar):
                    invertible_scalars.append(tensor)
                elif isinstance(tensor, Field):
                    raise Equation.LinearInvertibleTerm.NonLinear(operator)
                else:
                    raise Equation.LinearInvertibleTerm.NonInvertible(
                        operator, tensor
                    )

            if not found_field:
                return Equation.LinearInvertibleTerm.NonLinear(operator)

            return Equation.LinearInvertibleTerm(
                return_field,
                invertible_matrices,
                invertible_scalars,
                operator.coefficient
            )

    class NoLinearTerms(Exception):
        def __init__(self, equation):
            error_msg = (
                "Unable find a 'linear invertible term' in equation {}\n" +
                "with unknowns {}."
            )
            super().__init__(
                error_msg.format(equation.equation, equation.unknowns)
            )

    class MultipleLinearTerms(Exception):
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
                    raise Equation.MultipleLinearTerms(self)
                else:
                    found_linear_term = True
            except Equation.LinearInvertibleTerm.NonInvertible:
                rest += operator
            except Equation.LinearInvertibleTerm.NonLinear:
                rest += operator

        if not found_linear_term:
            raise Equation.NoLinearTerms(self)

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

        return FieldRule(
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


class FieldRule(object):
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
    def __init__(self, field_rules):
        self.field_rules = field_rules

    def __str__(self):
        return str(self.field_rules)

    __repr__ = __str__

    def substitute(self, target, max_dimension):
        """
        Substitute everywhere iteratively, until there are not any
        remaining substitutions to be made
        """
        for field_rule in self.field_rules:
            target = field_rule.substitute_once(target, max_dimension)

        if any(field_rule.is_in(target) for field_rule in self.field_rules):
            return self.substitute(target, max_dimension)

        return target

    def solve(self, max_dimension):
        """
        Converts a system of solutions and to a system in which they have
        been used to remove any occurrence of their field names in their
        replacements.
        """
        new_rules = SystemSolution([
            FieldRule(
                target_rule.field,
                self.substitute(target_rule.replacement, max_dimension)
                .filter_by_max_dimension(max_dimension)
            )
            for target_rule in self.field_rules
        ])

        if any(field_rule.is_in(target_rule.replacement)
               for field_rule in self.field_rules
               for target_rule in self.field_rules):
            return new_rules.solve(max_dimension)

        return new_rules
