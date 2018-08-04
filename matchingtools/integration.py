from abc import ABCMeta, abstractmethod
from itertools import combinations

from matchingtools.core import (
    RealConstant, RealField, Statistics, Operator, OperatorSum,
    epsilon_up, epsilon_down, sigma_vector
)
from matchingtools.eomsolutions import EOMSolution, EOMSolutionSystem
from matchingtools.indices import Index
from matchingtools.lsttools import concat, iterate
from matchingtools.shortcuts import D
from matchingtools.utils import merge_dicts


class Mass(RealConstant):
    # Masses are tensors that are associated with some field. They can have
    # zero or one indices, and they can be exponentiated to some integer
    # power. Products of powers of masses may be simplified.

    def __init__(self, field_name, indices=None, exponent=1):
        # Compute the empty or one-element list of indices
        if indices is None:
            indices = []

        # Extra attributes that do not appear in Tensor
        self.name = "M_" + field_name
        self.indices = indices
        self.derivatives_indices = []
        self.statistics = Statistics.BOSON
        self._tensor_dimension = 0
        self.field_name = field_name
        self.exponent = exponent

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
            and self.indices == other.indices
            and self.exponent == other.exponent
        )

    def __hash__(self):
        return hash(self.name)

    def clone(self):
        return Mass(
            field_name=self.field_name,
            indices=self.indices,
            exponent=self.exponent
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
        return [
            Mass._simplify_product(operator)
            for operator in convertible._to_operator_sum().operators
        ]

    def shares_index(self, other):
        return (
            len(self.indices) == 0 and len(other.indices) == 0
            or any(own_index in other.indices for own_index in self.indices)
        )

    def free_index_pair(self, other):
        if len(self.indices) == 0:
            return []

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
            if tensor.name[0] == 'M'
            # if isinstance(tensor, Mass)
        ]
        rest = [
            tensor for tensor in operator.tensors
            if tensor.name[0] != 'M'
            # if not isinstance(tensor, Mass)
        ]

        new_masses = []
        while masses:
            for mass_1, mass_2 in combinations(masses, 2):
                names_match = mass_1.field_name == mass_2.field_name
                index_shared = mass_1.shares_index(mass_2)
                if (names_match and index_shared):
                    new_masses.append(
                        Mass(mass_1.field_name, mass_1.free_index_pair(mass_2))
                    )
                    masses.remove(mass_1)
                    masses.remove(mass_2)
            else:
                new_masses += masses
                break

        return Operator(new_masses + rest, operator.coefficient)


class HeavyField(object, metaclass=ABCMeta):
    @abstractmethod
    def quadratic_terms(self):
        pass

    @abstractmethod
    def eom_solutions(self, interaction_lagrangian, inverse_mass_order):
        pass


class BosonMixin(object):
    def eom_solutions(self, interaction_lagrangian, max_dimension):
        variation = interaction_lagrangian.variation(self.field.conjugate())

        return [
            EOMSolution(
                self.field,
                -self.propagator(variation, max_dimension)
            )
        ]


class Scalar(BosonMixin, HeavyField):
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
            ),
            OperatorSum()
        )


class Vector(BosonMixin, HeavyField):
    def __init__(self, field, vector_index, flavor_index=None):
        self.field = field
        self.vector_index = vector_index
        self.mass = Mass(
            field_name=field.name,
            index=flavor_index,
            exponent=1
        )

    def quadratic_terms(self):
        factor = 1/2 if isinstance(self.field, RealField) else 1

        mu = self.vector_index
        nu = Index(self.vector_index.name)

        field_mu = self.field
        field_nu = self.field._replace_indices({mu: nu})

        kinetic_terms = (
            - D(mu, field_nu.conjugate()) * D(mu, field_nu)
            + D(mu, field_nu.conjugate()) * D(nu, field_mu)
        )
        mass_term = self.mass**2 * field_mu.conjugate() * field_mu

        return factor * (kinetic_terms + mass_term)

    def propagator(self, target, max_dimension):
        nu = self.vector_index

        def diff_op(subtarget):
            mu = Index(nu.name)
            subtarget = subtarget.filter_by_max_dimension(max_dimension - 2)

            return self.mass ** (-2) * (
                D(mu, D(nu, subtarget._replace_indices({nu: mu})))
                + D(mu, D(mu, subtarget))
            )

        return sum(
            iterate(
                diff_op,
                self.mass ** (-2) * target,
                round(max_dimension / 2)
            ),
            OperatorSum()
        )


class DiracFermion(HeavyField):
    def __init__(
            self, name,
            left_field, right_field,
            left_spinor_index, right_spinor_index,
            flavor_index=None):
        self.left_field = left_field
        self.right_field = right_field
        self.left_spinor_index = left_spinor_index
        self.right_spinor_index = right_spinor_index
        self.mass = Mass(
            field_name=name,
            index=flavor_index,
            exponent=1
        )

    def quadratic_terms(self):
        alpha = self.left_spinor_index
        alpha_dot = Index(alpha.name)
        beta_dot = self.right_spinor_index
        beta = Index(beta_dot.name)
        mu = Index('mu')

        fL = self.left_field
        fR = self.right_field
        fLc = self.left_field.conjugate()._replace_indices({alpha: alpha_dot})
        fRc = self.right_field.conjugate()._replace_indices({beta_dot: beta})

        kinetic_terms = (
            1j * fLc * sigma_vector(mu, alpha_dot, alpha) * D(mu, fL)
            + 1j * fRc * sigma_vector.c(mu, beta, beta_dot) * D(mu, fR)
        )

        mass_terms = - self.mass * (
            fLc * epsilon_down(alpha_dot, beta_dot) * fR
            + fRc * epsilon_up(beta, alpha) * fL
        )

        return kinetic_terms + mass_terms

    def eom_solutions(self, interaction_lagrangian, max_dimension):
        alpha = Index(self.left_spinor_index.name)
        alpha_dot = Index(alpha.name)
        beta_dot = Index(self.right_spinor_index.name)
        beta = Index(beta_dot.name)
        mu = Index('mu')

        fL = self.left_field._replace_indices(
            {self.left_spinor_index: alpha}
        )
        fR = self.right_field._replace_indices(
            {self.right_spinor_index: beta_dot}
        )

        variation_left = interaction_lagrangian.variation(fL)
        variation_right = interaction_lagrangian.variation(fR)

        replacement_left = self.mass ** (-1) * epsilon_down(alpha, beta) * (
            -1j * sigma_vector.c(mu, beta, self.right_spinor_index)
            * D(mu, self.right_field)
            + variation_right
        )
        replacement_right = self.mass ** (-1) * epsilon_up(
            beta_dot, alpha_dot
        ) * (
            -1j * sigma_vector.c(mu, alpha_dot, self.left_spinor_index)
            * D(mu, self.left_field)
            + variation_left
        )

        return [
            EOMSolution(fL, replacement_left),
            EOMSolution(fR, replacement_right)
        ]


def integrate_out(interaction_lagrangian, heavy_fields, max_dimension):
    eom_solutions = EOMSolutionSystem(
        concat(
            heavy_field.eom_solutions(interaction_lagrangian, max_dimension)
            for heavy_field in heavy_fields
        )
    ).solve(max_dimension - 1)

    quadratic_lagrangian = sum(
        (heavy_field.quadratic_terms() for heavy_field in heavy_fields),
        OperatorSum()
    )

    full_lagrangian = quadratic_lagrangian + interaction_lagrangian

    return sum(
        map(
            Mass._simplify_product,
            eom_solutions.substitute(full_lagrangian, max_dimension).operators
        ),
        OperatorSum()
    )
