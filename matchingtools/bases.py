from collections import Counter
from operator import add
import itertools

from matchingtools.rules import Rule
from matchingtools.core import Constant, Field, OperatorSum
from matchingtools.uniquefields import _UniqueField
from matchingtools.byparts import integration_by_parts_results
from matchingtools.redefinitions import Redefinition
from matchingtools.lsttools import LookUpTable


class FieldContent(object):
    def __init__(self, field_counter, derivatives_count):
        self.field_counter = field_counter
        self.derivatives_count = derivatives_count

    def __str__(self):
        return str((self.field_counter, self.derivatives_count))

    __repr__ = __str__

    def __eq__(self, other):
        return (
            self.field_counter == other.field_counter
            and self.derivatives_count == other.derivatives_count
        )

    @staticmethod
    def from_operator(operator):
        return FieldContent(
            Counter(
                (tensor.name, tensor.is_conjugated)
                for tensor in operator.tensors
                if isinstance(tensor, Field)
            ),
            sum(
                len(tensor.derivatives_indices)
                for tensor in operator.tensors
            )
        )

    @staticmethod
    def classify(target):
        target = target._to_operator_sum()
        classes = LookUpTable()

        for operator in target.operators:
            classes.update(
                FieldContent.from_operator(operator),
                [operator],
                add
            )

        return classes.values


class Basis(object):
    def __init__(self, named_operators, fields):
        self.named_operators = {
            name: operator._to_operator()
            for name, operator in named_operators.items()
        }

        self.fields = fields
        self.max_dimension = self._compute_max_dimension()

    @staticmethod
    def extract_constant_factor(target, operator):
        unique_field = _UniqueField.from_operator(operator)

        factor = (
            Rule(operator, unique_field)
            .apply(target)
            .functional_derivative(unique_field)
        )

        constant_factor = sum(
            operator for operator in factor.operators
            if all(isinstance(tensor, Constant) for tensor in operator.tensors)
        )

        return constant_factor

    def wilson_coefficients(self, target):
        coefficients = {
            name: Basis.extract_constant_factor(target, operator)
            for name, operator in self.named_operators.items()
        }

        rest = target - sum(
            coefficients[name] * self.named_operators[name]
            for name in self.named_operators
        )

        return coefficients, rest

    def replace_kinetic_term_structures(self, target):
        for field in self.fields:
            target = Redefinition.replace_kinetic_term_structure(
                target,
                field,
                self.max_dimension
            )

        return target

    def _compute_max_dimension(self):
        return max(
            operator.dimension
            for operator in self.named_operators.values()
        )

    def recover(self, coefficients):
        return sum(
            (
                coefficient * self.named_operators[name]
                for name, coefficient in coefficients.items()
            ),
            OperatorSum()
        )

    def use_integral_by_parts(self, lagrangian, structures):
        field_content_classes = FieldContent.classify(lagrangian)
        coefficients = {}
        rest = OperatorSum()

        def number_of_remaining_operators(pair):
            result = pair[1]
            for field in self.fields:
                result = Redefinition.remove_structure(
                    result,
                    structures[field]
                )
            return len(result.operators)

        for field_content_class in field_content_classes:
            possibilities = list(itertools.product(
                *[
                    integration_by_parts_results(operator)
                    for operator in field_content_class
                ]
            ))

            possible_wilson_coefficients = [
                self.wilson_coefficients(sum(possibility, OperatorSum()))
                for possibility in possibilities
            ]

            new_coefficients, new_rest = min(
                possible_wilson_coefficients,
                key=number_of_remaining_operators
            )

            coefficients = Basis.merge_coefficients(
                coefficients,
                new_coefficients
            )
            rest += new_rest

        return coefficients, rest

    def apply_identity(self, lagrangian, identity):
        def number_of_remaining_operators(pair):
            return len(pair[1].operators)

        possible_outcomes = [
            self.wilson_coefficients(rule.apply(lagrangian))
            for rule in identity.rules()
        ]

        return min(possible_outcomes, key=number_of_remaining_operators)

    @staticmethod
    def merge_coefficients(first_coefficients, second_coefficients):
        merged_names = set(
            list(first_coefficients.keys())
            + list(second_coefficients.keys())
        )

        return {
            name:
            first_coefficients.get(name, OperatorSum())
            + second_coefficients.get(name, OperatorSum())
            for name in merged_names
        }

    def compute_wilson_coefficients(
            self, lagrangian, identities
    ):
        structures = {
            field: Redefinition.find_kinetic_term_structure(lagrangian, field)
            for field in self.fields
        }

        coefficients, rest = self.wilson_coefficients(lagrangian)
        for _ in range(5):
            # Use "generalized equations of motion" (== redefinitions)
            lagrangian = self.recover(coefficients) + rest
            lagrangian = self.replace_kinetic_term_structures(lagrangian)
            coefficients, rest = self.wilson_coefficients(lagrangian)

            # Use integral by parts
            # lagrangian = self.recover(coefficients) + rest
            # coefficients, rest = self.use_integral_by_parts(lagrangian)

            new_coefficients, rest = self.use_integral_by_parts(
                rest,
                structures
            )
            coefficients = Basis.merge_coefficients(
                coefficients,
                new_coefficients
            )

            # Apply identities
            for identity in identities:
                new_coefficients, rest = self.apply_identity(
                    rest,
                    identity
                )
                coefficients = Basis.merge_coefficients(
                    coefficients,
                    new_coefficients
                )

        return coefficients, rest


class WilsonCoefficient(object):
    def __init__(self, operator_name, coefficient):
        self.operator_name = operator_name
        self.coefficient = coefficient
