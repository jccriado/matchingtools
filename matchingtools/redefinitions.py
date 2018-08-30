from matchingtools.core import Field, OperatorSum
from matchingtools.rules import Rule
from matchingtools.uniquefields import _UniqueField
from matchingtools.matches import Match


class Redefinition(object):
    def __init__(self, kinetic_term_structure, replacement):
        self.kinetic_term_structure = kinetic_term_structure
        self.replacement = replacement

        self.rule = Rule(kinetic_term_structure, replacement)
        self.conjugate_rule = Rule(
            kinetic_term_structure.conjugate(),
            replacement.conjugate()
        )

    def apply(self, target, max_dimension):
        target = target._to_operator_sum()

        return sum(
            self.rule.apply(operator)
            if Match.match_operators(self.kinetic_term_structure, operator)
            else self.conjugate_rule.apply(operator)
            for operator in target.operators
        ).filter_by_max_dimension(max_dimension)

    @staticmethod
    def extract_factor(target, operator):
        operator = operator._to_operator()
        unique_field = _UniqueField.from_operator(operator)

        return (
            Rule(operator, unique_field)
            .apply(target)
            .functional_derivative(unique_field)
        )

    @staticmethod
    def find_field(target, with_derivatives):
        for operator in target.operators:
            found_field = False
            for tensor in operator.tensors:
                if isinstance(tensor, Field):
                    if found_field:
                        break
                    else:
                        found_field = True
                        has_derivatives = len(tensor.derivatives_indices) > 0
            else:
                if found_field and (has_derivatives == with_derivatives):
                    return operator
        return OperatorSum()

    @staticmethod
    def find_kinetic_term(lagrangian, field):
        def is_kinetic_term(operator):
            field_count = 0
            derivative_count = 0
            for tensor in operator.tensors:
                if isinstance(tensor, Field):
                    if field_count == 2:
                        return False
                    if tensor.name != field.name:
                        return False
                    field_count += 1
                    derivative_count += len(tensor.derivatives_indices)
            return field_count == 2 and 0 < derivative_count < 3

        for operator in lagrangian.operators:
            if is_kinetic_term(operator):
                return operator

    @staticmethod
    def find_kinetic_term_structure(lagrangian, field):
        return Redefinition.find_field(
            target=lagrangian._to_operator_sum().variation(field.conjugate()),
            with_derivatives=True
        )

    @staticmethod
    def replace_kinetic_term_structure(lagrangian, field, max_dimension):
        kinetic_term = Redefinition.find_kinetic_term(lagrangian, field)
        kinetic_term_structure = Redefinition.find_kinetic_term_structure(
            kinetic_term,
            field
        )

        # TODO: + corrections
        replacement = (
            lagrangian.variation(field.conjugate()) - kinetic_term_structure
        )

        redefinition = Redefinition(kinetic_term_structure, replacement)

        return redefinition.apply(lagrangian, max_dimension)

    @staticmethod
    def remove_structure(lagrangian, structure):
        def contains_structure(operator):
            return not (
                Match.match_operators(structure, operator) is None
                and
                Match.match_operators(structure.conjugate(), operator) is None
            )

        return sum(
            (
                operator for operator in lagrangian.operators
                if not contains_structure(operator)
            ),
            OperatorSum()
        )
