from operators import (
    Tensor, Operator, OperatorSum, Op, OpSum,
    TensorBuilder, FieldBuilder, D_op,
    apply_derivatives, concat, number_op, symbol_op,
    generic, boson, fermion,
    sigma4, sigma4bar, epsUp, epsUpDot, epsDown, epsDownDot)

class Scalar(object):
    def apply_diff_op(self, operator_sum):
        final_op_sum = OperatorSum()
        for operator in operator_sum.operators:
            new_ind = operator.max_index + 1
            final_op_sum += apply_derivatives(
                [new_ind, new_ind],
                -self.free_inv_mass_sq * operator)
        return final_op_sum

    def apply_propagator(self, operator_sum, max_order=2, max_dim=4):
        final_op_sum = operator_sum
        for order in range(0, max_order + 1, 2):
            operator_sum = self.apply_diff_op(operator_sum)
            final_op_sum += operator_sum
        coef_op = -self.free_inv_mass_sq
        return OperatorSum([op * coef_op for op in final_op_sum.operators
                            if op.dimension <= max_dim])

    
class Vector(object):
    def apply_diff_op(self, operator_sum):
        n = self.num_of_inds
        inds_1 = [0] + list(range(-2, -n - 1, -1))
        inds_2 = list(range(-1, -n - 1, -1))
        generic_1 = Operator([generic(*inds_1)])
        generic_2 = Operator([generic(*inds_2)])
        generic_der_1 = apply_derivatives([0,-1], generic_1)
        generic_der_2 = apply_derivatives([0, 0], generic_2)
        structure = (OperatorSum([self.free_inv_mass_sq]) * generic_der_1 +
                     OperatorSum([number_op(-1) * self.free_inv_mass_sq]) *
                     generic_der_2)
        return structure.replace_all({"generic": operator_sum}, 6)

    def apply_propagator(self, operator_sum, max_order=2, max_dim=4):
        final_op_sum = operator_sum
        for order in range(0, max_order + 1, 2):
            operator_sum = self.apply_diff_op(operator_sum)
            final_op_sum += operator_sum
        coef_op = self.free_inv_mass_sq
        return OperatorSum([op * coef_op for op in final_op_sum.operators
                            if op.dimension <= max_dim])

                
class RealBoson(object):
    def __init__(self, name, num_of_inds, has_flavor=True):
        self.name = name
        self.num_of_inds = num_of_inds
        mass = "M" + self.name
        free_mass_inds = [-num_of_inds] if has_flavor else None
        self.free_inv_mass_sq = symbol_op(mass, -2, indices=free_mass_inds)
        mass_inds = [num_of_inds-1] if has_flavor else None
        self.mass_sq = symbol_op(mass, 2, indices=mass_inds)
    
    def equations_of_motion(self, interaction_lagrangian):
        variation = interaction_lagrangian.variation(self.name, True)
        return [(self.name, self.apply_propagator(-variation))]

class ComplexBoson(object):
    def __init__(self, name, c_name, num_of_inds, has_flavor=True):
        self.name = name
        self.c_name = c_name
        self.num_of_inds = num_of_inds
        mass = "M" + self.name
        free_mass_inds = [-num_of_inds] if has_flavor else None
        self.free_inv_mass_sq = symbol_op(mass, -2, indices=free_mass_inds)
        mass_inds = [num_of_inds-1] if has_flavor else None
        self.mass_sq = symbol_op(mass, 2, indices=mass_inds)

    def equations_of_motion(self, interaction_lagrangian):
        variation = interaction_lagrangian.variation(self.name, True)
        c_variation = interaction_lagrangian.variation(self.c_name, True)
        return [(self.name, self.apply_propagator(-c_variation)),
                (self.c_name, self.apply_propagator(-variation))]

class RealScalar(RealBoson, Scalar):
    def quadratic_terms(self):
        n = self.num_of_inds
        f = Tensor(self.name, list(range(n)), is_field=True,
                   dimension=1, statistics=boson)
        f_op = Operator([f])
        kinetic_term = OperatorSum([number_op(0.5)]) * D_op(n + 1, f) * D_op(n + 1, f)
        mass_term = number_op(-0.5) * self.mass_sq * f_op * f_op
        return kinetic_term + OperatorSum([mass_term])

class ComplexScalar(ComplexBoson, Scalar):
    def quadratic_terms(self):
        n = self.num_of_inds
        f = Tensor(self.name, list(range(n)), is_field=True,
                   dimension=1, statistics=boson)
        c_f = Tensor(self.c_name, list(range(n)), is_field=True,
                     dimension=1, statistics=boson)
        f_op = Operator([f])
        c_f_op = Operator([c_f])
        kinetic_term = D_op(n + 1, c_f) * D_op(n + 1, f)
        mass_term = number_op(-1) * self.mass_sq * c_f_op * f_op
        return kinetic_term + OperatorSum([mass_term])

class RealVector(RealBoson, Vector):
    def quadratic_terms(self):
        n = self.num_of_inds
        f1 = Op(Tensor(self.name, list(range(n)), is_field=True,
                       dimension=1, statistics=boson))
        f2 = Op(Tensor(self.name, [n] + list(range(1, n)),
                       is_field=True, dimension=1, statistics=boson))
        half = OpSum(number_op(0.5))
        kin1 = -half * f1.derivative(n) * f1.derivative(n)
        kin2 = half * f2.derivative(0) * f1.derivative(n)
        mass_term = half * OpSum(self.mass_sq * f1 * f1)
        return kin1 + kin2 + mass_term

class ComplexVector(ComplexBoson, Vector):
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
    def __init__(self, name, L_name, R_name, Lc_name, Rc_name, num_of_inds,
                 has_flavor=True):
        self.name = name
        self.L_name = L_name
        self.R_name = R_name
        self.Lc_name = Lc_name
        self.Rc_name = Rc_name
        self.num_of_inds = num_of_inds
        mass = "M" + self.name
        free_mass_inds = [-num_of_inds] if has_flavor else None
        self.free_inv_mass = symbol_op(mass, -1, indices=free_mass_inds)
        mass_inds = [num_of_inds - 1] if has_flavor else None
        self.mass = symbol_op(mass, 1, indices=mass_inds)
        
    def L_der(self):
        n = self.num_of_inds
        factor = OpSum(number_op(1j) * Op(sigma4bar(n + 1, -1, 0)))
        f = Op(Tensor(self.L_name, [0] + list(range(-2, -n - 1, -1)), is_field=True,
                      dimension=1.5, statistics=fermion))
        return factor * f.derivative(n + 1)

    def R_der(self):
        n = self.num_of_inds
        factor = OpSum(number_op(1j) * Op(sigma4(n + 1, -1, 0)))
        f = Op(Tensor(self.R_name, [0] + list(range(-2, -n - 1, -1)), is_field=True,
                      dimension=1.5, statistics=fermion))
        return factor * f.derivative(n + 1)

    def Lc_der(self):
        n = self.num_of_inds
        factor = OpSum(number_op(-1j) * Op(sigma4bar(n + 1, 0, -1)))
        f = Op(Tensor(self.Lc_name, [0] + list(range(-2, -n - 1, -1)), is_field=True,
                      dimension=1.5, statistics=fermion))
        return factor * f.derivative(n + 1)

    def Rc_der(self):
        n = self.num_of_inds
        factor = OpSum(number_op(-1j) * Op(sigma4(n + 1, 0, -1)))
        f = Op(Tensor(self.Rc_name, [0] + list(range(-2, -n - 1, -1)), is_field=True,
                      dimension=1.5, statistics=fermion))
        return factor * f.derivative(n + 1)

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
        n = self.num_of_inds
        fL, fR, fLc, fRc = map(self._create_op_field, [self.L_name, self.R_name,
                                                       self.Lc_name, self.Rc_name])
        half = OpSum(number_op(0.5))
        kinL = (fLc * fL).replace_first(self.L_name, self.L_der())
        kinR = (fRc * fR).replace_first(self.R_name, self.R_der())
        kinLc = (fLc * fL).replace_first(self.Lc_name, self.Lc_der())
        kinRc = (fRc * fR).replace_first(self.Rc_name, self.Rc_der())
        mass1 = self.mass * fLc * fR
        mass2 = self.mass * fRc * fL
        return half * (kinL + kinR + kinLc + kinRc) + OpSum(-mass1, -mass2)

class MajoranaFermion(object):
    def __init__(self, name, c_name, num_of_inds, has_flavor=True):
        self.name = name
        self.c_name = c_name
        self.num_of_inds = num_of_inds
        mass = "M" + self.name
        free_mass_inds = [-num_of_inds] if has_flavor else None
        self.free_inv_mass = symbol_op(mass, -1, indices=free_mass_inds)
        mass_inds = [num_of_inds - 1] if has_flavor else None
        self.mass = symbol_op(mass, 1, indices=mass_inds)

    def der(self):
        n = self.num_of_inds
        factor = OpSum(number_op(1j) * Op(sigma4bar(n + 1, -1, 0)))
        f = Op(Tensor(self.name, [0] + list(range(-2, -n - 1, -1)), is_field=True,
                      dimension=1.5, statistics=fermion))
        return factor * f.derivative(n + 1)

    def c_der(self):
        n = self.num_of_inds
        factor = OpSum(number_op(-1j) * Op(sigma4bar(n + 1, 0, -1)))
        f = Op(Tensor(self.c_name, [0] + list(range(-2, -n - 1, -1)), is_field=True,
                      dimension=1.5, statistics=fermion))
        return factor * f.derivative(n + 1)

    def pre_eps(self, op_sum):
        n = self.num_of_inds
        F = generic(*([0] + list(range(-2, -n - 1, -1))))
        return (Op(epsDownDot(0, -1)) * F).replace_all("generic", op_sum)

    def app_eps(self, op_sum):
        n = self.num_of_inds
        F = generic(*([0] + list(range(-2, -n - 1, -1))))
        return (F * Op(epsDown(0, -1))).replace_all("generic", op_sum)

    
    def equations_of_motion(self, interaction_lagrangian):
        variation = interaction_lagrangian.variation(self.name, fermion)
        c_variation = interaction_lagrangian.variation(self.c_name, fermion)
        inv_mass = OpSum(self.free_inv_mass) 
        return [(self.L_name, inv_mass * self.pre_eps(self.R_der() + Rc_variation)),
                (self.R_name, inv_mass * self.app_eps(self.L_der() + Lc_variation))]

    def _create_op_field(self, name):
        return Op(Tensor(name, list(range(self.num_of_inds)), is_field=True,
                         dimension=1.5, statistics=fermion))

    def quadratic_terms(self):
        n = self.num_of_inds
        f, fc = map(self._create_op_field, [self.name, self.c_name])
        half = OpSum(number_op(0.5))
        kin = (fc * f).replace_first(self.name, self.der())
        kinc = (fc * f).replace_first(self.c_name, self.c_der())
        mass1 = self.mass * f * f
        mass2 = self.mass * fc * fc
        return half * (kin + kinc + OpSum(-mass1, -mass2))

    
def integrate(heavy_fields, interaction_lagrangian, max_dim=6):
    eoms = dict(concat([field.equations_of_motion(interaction_lagrangian)
                        for field in heavy_fields]))
    quadratic_lagrangian = sum([field.quadratic_terms() for field in heavy_fields],
                               OperatorSum())
    total_lagrangian = quadratic_lagrangian + interaction_lagrangian
    return total_lagrangian.replace_all(eoms, max_dim)
