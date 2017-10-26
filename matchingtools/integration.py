"""
Module for the integration of heavy fields from a high-energy
lagrangian. Contains the function :func:`integrate` and the classes 
for representing the different types of heavy fields.
"""

import sys
from fractions import Fraction

from matchingtools.core import (
    Tensor, Op, OpSum,
    apply_derivatives, concat, i_op, number_op, power_op,
    generic, boson, fermion,
    sigma4, sigma4bar, epsUp, epsUpDot, epsDown, epsDownDot)

class Scalar(object):
    """Provide a propagator for scalars."""
    
    def apply_diff_op(self, operator_sum):
        """Apply the differential operator -D^2/M^2."""
        final_op_sum = OpSum()
        for operator in operator_sum.operators:
            new_ind = operator.max_index + 1
            final_op_sum += apply_derivatives(
                [new_ind, new_ind],
                -self.free_inv_mass_sq * operator)
        return final_op_sum

    def apply_propagator(self, operator_sum, max_order, max_dim):
        """
        Apply the scalar propagator -(D^2 + M^2)^(-1).

        Args:
            operator_sum (OperatorSum): the propagator is to be applied to it
            max_order=2 (int): maximum order in the derivatives to which
                               the propagator is to be expanded 
            max_dim=4 (int): maximum dimension of the resulting operators

        Return:
            OperatorSum of the operators with the propagator applied.
        """
        final_op_sum = operator_sum
        for order in range(0, max_order + 1, 2):
            operator_sum = self.apply_diff_op(operator_sum)
            final_op_sum += operator_sum
        coef_op = -self.free_inv_mass_sq
        return OpSum(*[op * coef_op for op in final_op_sum.operators
                       if op.dimension <= self.max_dim])

    
class Vector(object):
    """Provide a propagator for vectors."""
    
    def apply_diff_op(self, operator_sum):
        """
        Apply the diff. op. (D_mu D_nu - D^2 eta_{munu})/M^2.

        The index mu is to be contracted with the first free index of the
        operator sum J_mu provided as an argument.
        """
        n = self.num_of_inds
        inds_1 = [0] + list(range(-2, -n - 1, -1))
        inds_2 = list(range(-1, -n - 1, -1))
        generic_1 = Op(generic(*inds_1))
        generic_2 = Op(generic(*inds_2))
        generic_der_1 = apply_derivatives([0,-1], generic_1)
        generic_der_2 = apply_derivatives([0, 0], generic_2)
        structure = (OpSum(self.free_inv_mass_sq) * generic_der_1 +
                     OpSum(number_op(-1) * self.free_inv_mass_sq) *
                     generic_der_2)
        return structure.replace_all({"generic": operator_sum}, 6)

    def apply_propagator(self, operator_sum, max_order, max_dim):
        """
        Apply the vector propagator [(M^2 + D^2) eta_{munu} - D_mu D_nu]^(-1).

        Args:
            operator_sum (OperatorSum): the propagator is to be applied to it
            max_order=2 (int): maximum order in the derivatives to which
                               the propagator is to be expanded 
            max_dim=4 (int): maximum dimension of the resulting operators

        Return:
            OperatorSum of the operators with the propagator applied.
        """
        final_op_sum = operator_sum
        for order in range(0, max_order + 1, 2):
            operator_sum = self.apply_diff_op(operator_sum)
            final_op_sum += operator_sum
        coef_op = self.free_inv_mass_sq
        return OpSum(*[op * coef_op for op in final_op_sum.operators
                       if op.dimension <= max_dim])

                
class RealBoson(object):
    """
    Provide the EOMs and attributes for real bosons.

    Attributes:
        name (string): name identifier of the corresponding tensor
        num_of_inds (int): number of indices of the corr. tensor
        free_inv_mass_sq (Operator): operator containing the inverse mass
                                     squared tensor with a free index
        mass_sq (Operator): operator containing the mass squared tensor
                            non-negative index (ready to appear directly)
    """
    def __init__(self, name, num_of_inds, has_flavor=True, order=2, max_dim=4):
        """
        Args:
            name (string): name identifier of the corresponding tensor
            num_of_inds (int): number of indices of the corr. tensor
            has_flavor (bool): specifies if there are several generations
            order (int): maximum order in the derivatives for the propagator
        """
        self.name = name
        self.num_of_inds = num_of_inds
        mass = "M" + self.name
        free_mass_inds = [-num_of_inds] if has_flavor else None
        self.free_inv_mass_sq = power_op(mass, -2, indices=free_mass_inds)
        mass_inds = [num_of_inds-1] if has_flavor else None
        self.mass_sq = power_op(mass, 2, indices=mass_inds)
        self.order = order
        self.max_dim = max_dim
    
    def equations_of_motion(self, interaction_lagrangian):
        """
        Solve the EOMs to given order in 1/M.

        Args:
            interaction_lagrangian (OperatorSum)
            max_order=2 (int): maximum order in 1/M to which the propagator
                               is to be expanded
            max_dim=4 (int): maximum dimension of the resulting operators
        Return:
            A single-element list with a pair whose second
            element is the solution to the EOMs
        """
        variation = interaction_lagrangian.variation(self.name, True)
        return [(self.name,
                 self.apply_propagator(-variation, self.order, self.max_dim))]

class ComplexBoson(object):
    """
    Provide the EOMs and attributes for complex bosons.

    Attributes:
        name (string): name identifier of the corresponding tensor
        c_name (string): name identifier of the conjugate tensor
        num_of_inds (int): number of indices of the corr. tensor
        free_inv_mass_sq (Operator): operator containing the inverse mass
                                     squared tensor with a free index
        mass_sq (Operator): operator containing the mass squared tensor
                            non-negative index (ready to appear directly)
    """
    def __init__(self, name, c_name, num_of_inds, has_flavor=True, order=2,
                 max_dim=4):
        """
        Args:
            name (string): name identifier of the corresponding tensor
            c_name (string): name identifier of the conjugate tensor
            num_of_inds (int): number of indices of the corr. tensor
            has_flavor (bool): specifies if there are several generations
            order (int): maximum order in the derivatives for the propagator
        """
        self.name = name
        self.c_name = c_name
        self.num_of_inds = num_of_inds
        mass = "M" + self.name
        free_mass_inds = [-num_of_inds] if has_flavor else None
        self.free_inv_mass_sq = power_op(mass, -2, indices=free_mass_inds)
        mass_inds = [num_of_inds-1] if has_flavor else None
        self.mass_sq = power_op(mass, 2, indices=mass_inds)
        self.order = order
        self.max_dim = max_dim

    def equations_of_motion(self, interaction_lagrangian):
        """
        Solve the EOMs to given order in 1/M.

        Args:
            interaction_lagrangian (OperatorSum)
            max_order=2 (int): maximum order in 1/M to which the propagator
                               is to be expanded
            max_dim=4 (int): maximum dimension of the resulting operators
        Return:
            A two-element list with a pairs whose second
            element are the solution to the EOMs for the field itself
            and the complex conjugate, respectively
        """
        variation = interaction_lagrangian.variation(self.name, True)
        c_variation = interaction_lagrangian.variation(self.c_name, True)
        
        return [(self.name,
                 self.apply_propagator(
                     -c_variation, self.order, self.max_dim)),
                (self.c_name,
                 self.apply_propagator(
                     -variation, self.order, self.max_dim))]

class RealScalar(RealBoson, Scalar):
    """
    Representation for heavy real scalar bosons.

    The ``has_flavor`` argument to the constructor is a bool that
    specifies whether there are several generations of the heavy field,
    whereas the ``order`` argument gives a maximum order in the 
    derivatives for the propagator.

    Attributes:
        name (string): name identifier of the corresponding tensor
        num_of_inds (int): number of indices of the corresponding tensor
    """
    def quadratic_terms(self):
        """Construct the quadratic terms (1/2)[(DS)^2 + M^2 S^2]."""
        n = self.num_of_inds
        f = Tensor(self.name, list(range(n)), is_field=True,
                   dimension=1, statistics=boson)
        f_op = Op(f)
        kinetic_term = (OpSum(number_op(Fraction(1, 2))) *
                        Op(f).derivative(n + 1) *
                        Op(f).derivative(n + 1))
        mass_term = number_op(-Fraction(1, 2)) * self.mass_sq * f_op * f_op
        return kinetic_term + OpSum(mass_term)

class ComplexScalar(ComplexBoson, Scalar):
    """
    Representation for heavy complex scalar bosons.

    The ``has_flavor`` argument to the constructor is a bool that
    specifies whether there are several generations of the heavy field,
    whereas the ``order`` argument gives a maximum order in the 
    derivatives for the propagator.

    Attributes:
        name (string): name identifier of the corresponding tensor
        c_name (string): name identifier of the conjugate tensor
        num_of_inds (int): number of indices of the corresponding tensor
    """
    def quadratic_terms(self):
        """Construct the quadratic terms DSc DS + M^2 Sc S."""
        n = self.num_of_inds
        f = Tensor(self.name, list(range(n)), is_field=True,
                   dimension=1, statistics=boson)
        c_f = Tensor(self.c_name, list(range(n)), is_field=True,
                     dimension=1, statistics=boson)
        f_op = Op(f)
        c_f_op = Op(c_f)
        kinetic_term = Op(c_f).derivative(n + 1) * Op(f).derivative(n + 1)
        mass_term = number_op(-1) * self.mass_sq * c_f_op * f_op
        return kinetic_term + OpSum(mass_term)

class RealVector(RealBoson, Vector):
    """
    Representation for heavy real vector bosons.

    The ``has_flavor`` argument to the constructor is a bool that
    specifies whether there are several generations of the heavy field,
    whereas the ``order`` argument gives a maximum order in the 
    derivatives for the propagator.

    Attributes:
        name (string): name identifier of the corresponding tensor
        num_of_inds (int): number of indices of the corresponding tensor
    """
    def quadratic_terms(self):
        n = self.num_of_inds
        f1 = Op(Tensor(self.name, list(range(n)), is_field=True,
                       dimension=1, statistics=boson))
        f2 = Op(Tensor(self.name, [n] + list(range(1, n)),
                       is_field=True, dimension=1, statistics=boson))
        half = OpSum(number_op(Fraction(1, 2)))
        kin1 = -half * f1.derivative(n) * f1.derivative(n)
        kin2 = half * f2.derivative(0) * f1.derivative(n)
        mass_term = half * OpSum(self.mass_sq * f1 * f1)
        return kin1 + kin2 + mass_term

class ComplexVector(ComplexBoson, Vector):
    """
    Representation for heavy complex vector bosons.

    The ``has_flavor`` argument to the constructor is a bool that
    specifies whether there are several generations of the heavy field,
    whereas the ``order`` argument gives a maximum order in the 
    derivatives for the propagator.

    Attributes:
        name (string): name identifier of the corresponding tensor
        c_name (string): name identifier of the conjugate tensor
        num_of_inds (int): number of indices of the corresponding tensor
    """
    def quadratic_terms(self):
        n = self.num_of_inds
        f1 = Op(Tensor(self.name, list(range(n)), is_field=True,
                       dimension=1, statistics=boson))
        f1c = Op(Tensor(self.c_name, list(range(n)),
                        is_field=True, dimension=1, statistics=boson))
        f2c = Op(Tensor(self.c_name, [n] + list(range(1, n)),
                        is_field=True, dimension=1, statistics=boson))
        kin1 = -f1c.derivative(n) * f1.derivative(n)
        kin2 = f2c.derivative(0) * f1.derivative(n)
        mass_term = OpSum(self.mass_sq * f1c * f1)
        return kin1 + kin2 + mass_term
    
class VectorLikeFermion(object):
    """
    Representation for heavy vector-like fermions

    The ``has_flavor`` argument to the constructor is a bool that
    specifies whether there are several generations of the heavy field.

    Attributes:
        name (string): name of the field
        L_name (string): name of the left-handed part
        R_name (string): name of the right-handed part
        Lc_name (string): name of the conjugate of the left-handed part
        Rc_name (string): name of the conjugate of the right-handed part
        num_of_inds (int): number of indices of the corresponding tensor
    """
    def __init__(self, name, L_name, R_name, Lc_name, Rc_name, num_of_inds,
                 has_flavor=True):
        """
        Args:
            name (string): name of the field
            L_name (string): name of the left-handed part
            R_name (string): name of the right-handed part
            Lc_name (string): name of the conjugate of the left-handed part
            Rc_name (string): name of the conjugate of the right-handed part
            num_of_inds (int): number of indices of the corresponding tensor
            has_flavor (bool): specifies if there are several generations
        """
        self.name = name
        self.L_name = L_name
        self.R_name = R_name
        self.Lc_name = Lc_name
        self.Rc_name = Rc_name
        self.num_of_inds = num_of_inds
        mass = "M" + self.name
        free_mass_inds = [-num_of_inds] if has_flavor else None
        self.free_inv_mass = power_op(mass, -1, indices=free_mass_inds)
        mass_inds = [num_of_inds - 1] if has_flavor else None
        self.mass = power_op(mass, 1, indices=mass_inds)
        
    def L_der(self):
        n = self.num_of_inds
        factor = OpSum(i_op * Op(sigma4bar(n, -1, 0)))
        f = Op(Tensor(self.L_name, [0] + list(range(-2, -n - 1, -1)),
                      is_field=True, dimension=1.5, statistics=fermion))
        return factor * f.derivative(n) # + 1)

    def R_der(self):
        n = self.num_of_inds
        factor = OpSum(i_op * Op(sigma4(n, -1, 0)))
        f = Op(Tensor(self.R_name, [0] + list(range(-2, -n - 1, -1)),
                      is_field=True, dimension=1.5, statistics=fermion))
        return factor * f.derivative(n)

    def Lc_der(self):
        n = self.num_of_inds
        factor = OpSum(- i_op * Op(sigma4bar(n, 0, -1)))
        f = Op(Tensor(self.Lc_name, [0] + list(range(-2, -n - 1, -1)),
                      is_field=True, dimension=1.5, statistics=fermion))
        return factor * f.derivative(n)

    def Rc_der(self):
        n = self.num_of_inds
        factor = OpSum(- i_op * Op(sigma4(n, 0, -1)))
        f = Op(Tensor(self.Rc_name, [0] + list(range(-2, -n - 1, -1)),
                      is_field=True, dimension=1.5, statistics=fermion))
        return factor * f.derivative(n)

    def equations_of_motion(self, interaction_lagrangian):
        L_variation = -interaction_lagrangian.variation(self.L_name, fermion)
        R_variation = -interaction_lagrangian.variation(self.R_name, fermion)
        Lc_variation = interaction_lagrangian.variation(self.Lc_name, fermion)
        Rc_variation = interaction_lagrangian.variation(self.Rc_name, fermion)
        op_sum_inv_mass = OpSum(self.free_inv_mass)
        return [(self.L_name, op_sum_inv_mass * (self.R_der() + Rc_variation)),
                (self.R_name, op_sum_inv_mass * (self.L_der() + Lc_variation)),
                (self.Lc_name, op_sum_inv_mass * (self.Rc_der() + R_variation)),
                (self.Rc_name, op_sum_inv_mass * (self.Lc_der() + L_variation))]

    def _create_op_field(self, name):
        return Op(Tensor(name, list(range(self.num_of_inds)), is_field=True,
                         dimension=1.5, statistics=fermion))

    def quadratic_terms(self):
        """
        Construct the terms (1/2) [i FLc (D FL) - i (D FLc) FL + i FRc D FR
        - i (D FRc) FR] - M (FLc FR + FRc FL)
        """
        n = self.num_of_inds
        fL, fR, fLc, fRc = map(self._create_op_field,
                               [self.L_name, self.R_name,
                                self.Lc_name, self.Rc_name])
        half = OpSum(number_op(Fraction(1, 2)))
        kinL = (fLc * fL).replace_first(self.L_name, self.L_der())
        kinR = (fRc * fR).replace_first(self.R_name, self.R_der())
        kinLc = (fLc * fL).replace_first(self.Lc_name, self.Lc_der())
        kinRc = (fRc * fR).replace_first(self.Rc_name, self.Rc_der())
        mass1 = self.mass * fLc * fR
        mass2 = self.mass * fRc * fL
        return half * (kinL + kinR + kinLc + kinRc) + OpSum(-mass1, -mass2)

class MajoranaFermion(object):
    """
    Representation for heavy Majorana fermions.

    The ``has_flavor`` argument to the constructor is a bool that
    specifies whether there are several generations of the heavy field.

    Attributes:
        name (string): name identifier of the corresponding tensor
        c_name (string): name identifier of the conjugate tensor
        num_of_inds (int): number of indices of the corresponding tensor
    """
    def __init__(self, name, c_name, num_of_inds, has_flavor=True):
        """
        Args:
            name (string): name identifier of the corresponding tensor
            c_name (string): name identifier of the conjugate tensor
            num_of_inds (int): number of indices of the corresponding tensor
            has_flavor (bool): specifies if there are several generations
        """
        self.name = name
        self.c_name = c_name
        self.num_of_inds = num_of_inds
        mass = "M" + self.name
        free_mass_inds = [-num_of_inds] if has_flavor else None
        self.free_inv_mass = power_op(mass, -1, indices=free_mass_inds)
        mass_inds = [num_of_inds - 1] if has_flavor else None
        self.mass = power_op(mass, 1, indices=mass_inds)

    def der(self):
        n = self.num_of_inds
        factor = OpSum(i_op * Op(sigma4bar(n + 1, -1, 0)))
        f = Op(Tensor(self.name, [0] + list(range(-2, -n - 1, -1)),
                      is_field=True, dimension=1.5, statistics=fermion))
        return factor * f.derivative(n + 1)

    def c_der(self):
        n = self.num_of_inds
        factor = OpSum(i_op * Op(sigma4bar(n + 1, 0, -1)))
        f = Op(Tensor(self.c_name, [0] + list(range(-2, -n - 1, -1)),
                      is_field=True, dimension=1.5, statistics=fermion))
        return factor * f.derivative(n + 1)

    def pre_eps(self, op_sum):
        n = self.num_of_inds
        F = generic(*([0] + list(range(-2, -n - 1, -1))))
        return Op(epsDownDot(0, -1), F).replace_first("generic", op_sum)

    def app_eps(self, op_sum):
        n = self.num_of_inds
        F = generic(*([0] + list(range(-2, -n - 1, -1))))
        return Op(F, epsDown(0, -1)).replace_first("generic", op_sum)

    
    def equations_of_motion(self, interaction_lagrangian):
        variation = interaction_lagrangian.variation(self.name, fermion)
        c_variation = interaction_lagrangian.variation(self.c_name, fermion)
        inv_mass = OpSum(self.free_inv_mass)
        return [(self.c_name, -inv_mass * self.pre_eps(self.der() + c_variation)),
                (self.name, inv_mass * self.app_eps(self.c_der() + variation))]

    def quadratic_terms(self):
        """
        Construct the terms (i Fc (D F) - i (D Fc) F - (FF + Fc Fc))/2
        """
        n = self.num_of_inds
        f = Op(Tensor(self.name, list(range(n)), is_field=True,
                         dimension=1.5, statistics=fermion))
        f1 = Op(Tensor(self.name, [n] + list(range(1, n)), is_field=True,
                      dimension=1.5, statistics=fermion))
        c_f = Op(Tensor(self.c_name, list(range(n)), is_field=True,
                         dimension=1.5, statistics=fermion))
        c_f1 = Op(Tensor(self.c_name, [n] + list(range(1, n)), is_field=True,
                      dimension=1.5, statistics=fermion))
        half = OpSum(number_op(Fraction(1, 2)))
        kin = (c_f * f).replace_first(self.name, self.der())
        kinc = -(c_f * f).replace_first(self.c_name, self.c_der())
        mass1 = self.mass * Op(epsUp(0, n)) * f * f1
        mass2 = self.mass * c_f * c_f1 * Op(epsUpDot(0, n))
        return half * (kin + kinc + OpSum(mass1, -mass2))

def integrate(heavy_fields, interaction_lagrangian, max_dim=6, verbose=True):
    """
    Integrate out heavy fields.

    Heavy field classes: ``RealScalar``, ``ComplexScalar``, ``RealVector``, 
    ``ComplexVector``, ``VectorLikeFermion`` or ``MajoranaFermion``.

    Args:
        heavy_fields (list of heavy fields): to be integrated out
        interaction_lagrangian (``matchingtools.operators.OperatorSum``):
            from which to integrate out the heavy fields
        max_dim (int): maximum dimension of the operators in the effective
            lagrangian
        verbose (bool): specifies whether to print messages signaling
            the start and end of the integration process.
    """
    if verbose:
        sys.stdout.write("Integrating... ")
        sys.stdout.flush()

    eoms = dict(concat([field.equations_of_motion(interaction_lagrangian)
                        for field in heavy_fields]))
    replaced_eoms = {field_name:
                     replacement.replace_all(eoms, max_dim - 2)
                     for field_name, replacement in eoms.items()}
    quadratic_lagrangian = sum([field.quadratic_terms()
                                for field in heavy_fields],
                               OpSum())
    total_lagrangian = quadratic_lagrangian + interaction_lagrangian
    result = total_lagrangian.replace_all(replaced_eoms, max_dim)

    if verbose:
        sys.stdout.write("done.\n")

    return result












