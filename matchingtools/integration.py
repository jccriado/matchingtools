from abc import ABCMeta, abstractmethod

from matchingtools.indices import Index
from matchingtools.core import RealConstant, RealField, D, Statistics, Operator
from matchingtools.eomsolutions import EOMSolution, EOMSolutionSystem
from matchingtools.lsttools import concat, iterate

class Mass(RealConstant):
    # Masses are tensors that are associated with some field. They can have
    # zero or one indices, and they can be exponentiated to some integer
    # power. Products of powers of masses may be simplified.
    
    def __init__(self, field_name, index=None, exponent=1):

        # Compute the empty or one-element list of indices
        indices = [] if index is None else [index]
        
        super().__init__(
            name="M_"+field_name,
            indices=indices,
            derivatives_indices=[],
            dimension=0,
            statistics=Statistics.BOSON
        )

        # Extra attributes that do not appear in Tensor
        self.field_name = field_name
        self.exponent = exponent
        self.index = index

    def __str__(self):
        exponent_str = '' if self.exponent == 1 else '^' + str(self.exponent)
        indices_str = ', '.join(map(str, self.indices))

        return "{name}{exponent}({indices})".format(
            name=self.name,
            exponent=exponent_str,
            indices=indices_str
        )

    def __eq__(self, other):
        return (
            isinstance(other, Mass)
            and self.index == other.index
            and self.exponent == other.exponent
        )

    def __hash__(self):
        return hash(self.name)
    
    def clone(self):
        return Mass(
            field_name=self.field_name,
            index=self.index,
            exponent=self.exponent
        )
        
    def __pow__(self, number):
        new_mass = self.clone()
        new_mass.exponent *= number
        return new_mass

    @staticmethod
    def _simplify(convertible):
        return [
            Mass._simplify_product(operator)
            for operator in convertible._to_operator_sum().operators
        ]

    @staticmethod
    def _simplify_product(operator):
        masses = {}
        rest = []
        
        for tensor in operator.tensors:
            if isinstance(tensor, Mass):
                masses.setdefault((tensor.field_name, tensor.index), 0)
                masses[(tensor.field_name, tensor.index)] += tensor.exponent
            else:
                rest.append(tensor)

        mass_product = Operator(
            [
                Mass(field_name, index, exponent)
                for (field_name, index), exponent in masses.items()
            ],
            1
        )
        
        return mass_product * Operator(rest, operator.coefficient)

    
class HeavyField(object, metaclass=ABCMeta):
    @abstractmethod
    def quadratic_terms(self):
        pass

    @abstractmethod
    def eom_solutions(self, interaction_lagrangian, inverse_mass_order):
        pass

    
class Scalar(HeavyField):
    def __init__(self, field, flavor_index=None):
        self.field = field
        self.mass = Mass(
            field_name=field.name,
            index=flavor_index,
            exponent=1
        )

    def quadratic_terms(self):
        mu = Index('mu')
        factor = 1/2 if isinstance(self.field, RealField) else 1
        
        kinetic_term = D(mu, self.field.conjugate()) * D(mu, self.field)
        mass_term = -self.mass**2 * self.field.conjugate() * self.field
        
        return factor * (kinetic_term + mass_term)

    def propagator(self, target, max_dimension):
        def minus_D2_over_M2(subtarget):
            subtarget = subtarget.filter_by_max_dimension(max_dimension - 2)
            mu = Index('mu')
            return -self.mass ** (-2) * D(mu, D(mu, subtarget))
        
        return sum(
            iterate(
                minus_D2_over_M2,
                -self.mass ** (-2) * target,
                round(max_dimension / 2)
            )
        )

    def eom_solutions(self, interaction_lagrangian, max_dimension):
        if isinstance(self.field, RealField):
            variation = interaction_lagrangian.variation(self.field)
            
            return [
                EOMSolution(
                    self.field.name,
                    -self.propagator(variation, max_dimension)
                )
            ]
        
        else:
            variation = interaction_lagrangian.variation(field)
            conjugate_variation = interaction_lagrangian.variation(
                self.field.conjugate()
            )
            
            return [
                EOMSolution(
                    self.field.name,
                    -self.propagator(conjugate_variation, max_dimension)),
                EOMSolution(
                    self.field.conjugate().name,
                    -self.propagator(variation, max_dimension))
            ]


def integrate_out(interaction_lagrangian, heavy_fields, max_dimension):
    eom_solutions = EOMSolutionSystem(
        concat(
            heavy_field.eom_solutions(interaction_lagrangian, max_dimension)
            for heavy_field in heavy_fields
        )
    ).solve(max_dimension - 1)
    
    quadratic_lagrangian = sum(
        heavy_field.quadratic_terms()
        for heavy_field in heavy_fields
    )

    full_lagrangian = quadratic_lagrangian + interaction_lagrangian

    return sum(
        map(
            Mass._simplify_product,
            eom_solutions.substitute(full_lagrangian, max_dimension).operators
        )
    )

    
