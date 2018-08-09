from abc import ABCMeta, abstractmethod

from matchingtools.core import RealField
from matchingtools.solutions import System
from matchingtools.indices import Index
from matchingtools.shortcuts import D
from matchingtools.invertibles import MassMatrix, MassScalar


class CanonicalField(object, metaclass=ABCMeta):
    @abstractmethod
    def quadratic_terms(self):
        pass


class BosonMixin(object):
    def boson_mass_term(self):
        if self.flavor_index is None:
            mass = MassScalar(self.field.name)
            return -mass**2 * self.field.conjugate() * self.field

        new_flavor_index = Index(self.flavor_index.name)
        mass = MassMatrix(
            self.field.name,
            new_flavor_index,
            self.flavor_index
        )

        return (
            -mass**2
            * self.field.conjugate()._replace_indices(
                {self.flavor_index: new_flavor_index}
            )
            * self.field
        )

    def factor(self):
        return 1/2 if isinstance(self.field, RealField) else 1


class Scalar(BosonMixin, CanonicalField):
    def __init__(self, field, flavor_index=None):
        self.field = field
        self.flavor_index = flavor_index

    def quadratic_terms(self):
        mu = Index('mu')

        kinetic_term = D(mu, self.field.conjugate()) * D(mu, self.field)

        return self.factor() * (kinetic_term + self.boson_mass_term())


class Vector(BosonMixin, CanonicalField):
    def __init__(self, field, vector_index, flavor_index=None):
        self.field = field
        self.vector_index = vector_index
        self.flavor_index = flavor_index

    def quadratic_terms(self):
        new_vector_index = Index(self.vector_index.name)

        kinetic_terms = (
            - D(new_vector_index, self.field.conjugate())
            * D(new_vector_index, self.field)

            + D(new_vector_index, self.field.conjugate())
            * D(
                self.vector_index,
                self.field.replace_indices(
                    {self.vector_index: new_vector_index}
                )
            )
        )

        return self.factor() * (kinetic_terms + self.boson_mass_term())


class UVTheory(object):
    def __init__(self, interaction_lagrangian, heavy_fields):
        self.heavy_fields = heavy_fields
        self.lagrangian = sum(
            (
                heavy_field.quadratic_terms()
                for heavy_field in heavy_fields
            ),
            interaction_lagrangian
        )

    def equations_of_motion_solutions(self, max_dimension):
        return System(
            [
                self.lagrangian.variation(heavy_field.field)
                for heavy_field in self.heavy_fields
            ],
            [heavy_field.field for heavy_field in self.heavy_fields]
        ).solve(max_dimension)

    def integrate_out(self, max_dimension):
        effective_lagrangian = (
            self.equations_of_motion_solutions(max_dimension - 1)
            .substitute(self.lagrangian, max_dimension)
        )
        return MassScalar._simplify(
            MassMatrix._simplify(
                effective_lagrangian
            )
        )

# def quadratic_terms(self):
#     alpha = self.left_spinor_index
#     alpha_dot = Index(alpha.name)
#     beta_dot = self.right_spinor_index
#     beta = Index(beta_dot.name)
#     mu = Index('mu')

#     fL = self.left_field
#     fR = self.right_field
#     fLc = self.left_field.conjugate()._replace_indices({alpha: alpha_dot})
#     fRc = self.right_field.conjugate()._replace_indices({beta_dot: beta})

#     kinetic_terms = (
#         1j * fLc * sigma_vector(mu, alpha_dot, alpha) * D(mu, fL)
#         + 1j * fRc * sigma_vector.c(mu, beta, beta_dot) * D(mu, fR)
#     )

#     mass_terms = - self.mass * (
#         fLc * epsilon_down(alpha_dot, beta_dot) * fR
#         + fRc * epsilon_up(beta, alpha) * fLp
#     )


#     return kinetic_terms + mass_terms
