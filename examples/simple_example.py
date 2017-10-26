"""
Simple example to illustrate some of the features of `matchingtools`.
The model has a :math:`SU(2)\times U(1)` simmetry and contains
a complex scalar doublet :math:`\phi` (the Higgs) with hypercharge
:math:`1/2` and a real scalar triplet :math:`\Xi` with zero
hypercharge that couple as:

.. math::
    \mathcal{L}_{int} = - \kappa\Xi^a\phi^\dagger\sigma^a\phi
    - \lambda \Xi^a \Xi^a \phi^\dagger\phi,

where :math:`\kappa` and :math:`\lambda` are a coupling constants
and :math:`\sigma^a` are the Pauli matrices. We will then integrate
out the heavy scalar :math:`\Xi` to obtain an effective Lgrangian
which we will finally write in terms of the operators.

.. math::
    \mathcal{O}_{\phi 6}=(\phi^\dagger\phi)^3, \\
    \mathcal{O}_{\phi 4}=(\phi^\dagger\phi)^2, \\
    \mathcal{O}^{(1)}_{\phi}= \phi^\dagger\phi
                              (D_\mu \phi)^\dagger D^\mu \phi, \\
    \mathcal{O}^{(3)}_{\phi}= (\phi^\dagger D_\mu \phi)
                              (D^\mu \phi)^\dagger \phi, \\
    \mathcal{O}_{D \phi} = \phi^\dagger(D_\mu \phi) 
                           \phi^\dagger D^\mu\phi, \\
    \mathcal{O}^*_{D \phi} = (D_\mu\phi)^\dagger\phi 
                             (D^\mu\phi)^\dagger\phi

"""

from matchingtools.core import (
    TensorBuilder, FieldBuilder, Op, OpSum, D,
    number_op, tensor_op, boson, fermion, kdelta)

from matchingtools.integration import RealScalar, integrate

from matchingtools.transformations import apply_rules

from matchingtools.output import Writer

# Creation of the model

sigma = TensorBuilder("sigma")
kappa = TensorBuilder("kappa")
lamb = TensorBuilder("lamb")

phi = FieldBuilder("phi", 1, boson)
phic = FieldBuilder("phic", 1, boson)
Xi = FieldBuilder("Xi", 1, boson)

interaction_lagrangian = -OpSum(
    Op(kappa(), Xi(0), phic(1), sigma(0, 1, 2), phi(2)),
    Op(lamb(), Xi(0), Xi(0), phic(1), phi(1)))

# Integration

heavy_Xi = RealScalar("Xi", 1, has_flavor=False)

heavy_fields = [heavy_Xi]
max_dim = 6
effective_lagrangian = integrate(
    heavy_fields, interaction_lagrangian, max_dim)

# Transformations of the effective Lgrangian

fierz_rule = (
    Op(sigma(0, -1, -2), sigma(0, -3, -4)),
    OpSum(number_op(2) * Op(kdelta(-1, -4), kdelta(-3, -2)),
          -Op(kdelta(-1, -2), kdelta(-3, -4))))
	      
Ophi6 = tensor_op("Ophi6")
Ophi4 = tensor_op("Ophi4")
O1phi = tensor_op("O1phi")
O3phi = tensor_op("O3phi")
ODphi = tensor_op("ODphi")
ODphic = tensor_op("ODphic")

definition_rules = [
    (Op(phic(0), phi(0), phic(1), phi(1), phic(2), phi(2)),
     OpSum(Ophi6)),
    (Op(phic(0), phi(0), phic(1), phi(1)),
     OpSum(Ophi4)),
    (Op(D(2, phic(0)), D(2, phi(0)), phic(1), phi(1)),
     OpSum(O1phi)),
    (Op(phic(0), D(2, phi(0)), D(2, phic(1)), phi(1)),
     OpSum(O3phi)),
    (Op(phic(0), D(2, phi(0)), phic(1), D(2, phi(1))),
     OpSum(ODphi)),
    (Op(D(2, phic(0)), phi(0), D(2, phic(1)), phi(1)),
     OpSum(ODphic))]

rules = [fierz_rule] + definition_rules
max_iterations = 2
transf_eff_lag = apply_rules(effective_lagrangian, rules, max_iterations)

# Output

if __name__ == "__main__":
    final_op_names = [
        "Ophi6", "Ophi4", "O1phi", "O3phi", "ODphi", "ODphic"]
    eff_lag_writer = Writer(transf_eff_lag, final_op_names)
    eff_lag_writer.write_text_file("simple_example_results.txt")


# -- LaTeX output (uncomment to produce it) --------------------------
# latex_tensor_reps = {"kappa": r"\kappa",
#                      "lamb": r"\lambda",
#                      "MXi": r"M_{{\Xi}}",
#                      "phi": r"\phi_{}",
#                      "phic": r"\phi^*_{}"}
# latex_coef_reps = {
#     "Ophi6": r"\frac{{\alpha_{{\phi 6}}}}{{\Lambda^2}}",
#     "Ophi4": r"\alpha_{{\phi 4}}",
#     "O1phi": r"\frac{{\alpha^{{(1)}}_{{\phi}}}}{{\Lambda^2}}",
#     "O3phi": r"\frac{{\alpha^{{(3)}}_{{\phi}}}}{{\Lambda^2}}",
#     "ODphi": r"\frac{{\alpha_{{D\phi}}}}{{\Lambda^2}}",
#     "ODphic": r"\frac{{\alpha^*_{{D\phi}}}}{{\Lambda^2}}"}
# latex_indices = ["i", "j", "k", "l"]
# eff_lag_writer.write_latex(
#     "simple_example", latex_tensor_reps, 
#     latex_coef_reps, latex_indices)
