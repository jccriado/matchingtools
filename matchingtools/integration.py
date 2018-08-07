from matchingtools.core import RealField
from matchingtools.solutions import System
from matchingtools.indices import Index
from matchingtools.shortcuts import D
from matchingtools.invertibles import MassMatrix, MassScalar


def boson_mass_term(field, flavor_index):
    if flavor_index is None:
        mass = MassScalar(field.name)
        return -mass**2 * field.conjugate() * field

    new_flavor_index = Index(flavor_index.name)
    mass = MassMatrix(
        field.name,
        new_flavor_index,
        flavor_index
    )

    return (
        -mass**2
        * field.conjugate()._replace_indices(
            {flavor_index: new_flavor_index}
        )
        * field
    )


def scalar_quadratic_terms(field, flavor_index=None):
    mu = Index('mu')
    factor = 1/2 if isinstance(field, RealField) else 1

    kinetic_term = D(mu, field.conjugate()) * D(mu, field)

    return factor * (kinetic_term + boson_mass_term(field, flavor_index))


def vector_quadratic_terms(field, vector_index, flavor_index=None):
    factor = 1/2 if isinstance(field, RealField) else 1
    new_vector_index = Index(vector_index.name)

    kinetic_terms = (
        - D(new_vector_index, field.conjugate())
        * D(new_vector_index, field)

        + D(new_vector_index, field.conjugate())
        * D(
            vector_index,
            field.replace_indices({vector_index: new_vector_index})
        )
    )

    return factor * (kinetic_terms + boson_mass_term(field, flavor_index))


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
#         + fRc * epsilon_up(beta, alpha) * fL
#     )


#     return kinetic_terms + mass_terms


def integrate_out(lagrangian, heavy_fields, max_dimension):
    eom_solutions = System(
        [
            lagrangian.variation(heavy_field)
            for heavy_field in heavy_fields
        ],
        heavy_fields
    ).solve(max_dimension - 1)

    return MassScalar._simplify(
        MassMatrix._simplify(
            eom_solutions.substitute(lagrangian, max_dimension)
        )
    )
