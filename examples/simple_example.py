"""
Simple example to illustrate some of the features of `effective`
The model has a :math:`SU(2)\times U(1)` simmetry and contains
a complex scalar doublet :math:`\phi` (the Higgs) with hypercharge
:math:`1/2` and a real scalar triplet :math:`\Xi` with zero
 hypercharge that couple as:

.. math::
   \mathcal{L}_{int} = - \kappa\Xi^a\phi^\dagger\sigma^a\phi
   - \lamb \Xi^a \Xi^a \phi^\dagger\phi,

where :math:`\kappa` and :math:`\lamb` are a coupling constants
and :math:`\sigma^a` are the Pauli matrices. We will then integrate
out the heavy scalar :math:`\Xi` to obtain an effective Lagrangian
which we will finally write in terms of the operators.

.. math::
   \mathcal{O}_\phi=(\phi^\dagger\phi)^3,\;
   \mathcal{O}_{\phi 4}=(\phi^\dagger\phi)^2
"""

from effective.operators import (
    TensorBuilder, FieldBuilder, Op, OpSum,
    number_op, tensor_op, boson, fermion, kdelta)

from effective.integration import RealScalar, integrate

from effective.transformations import apply_rules

from effective.output import Writer

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

heavy_Xi = RealScalar("Xi", 1)

heavy_fields = [heavy_Xi]
max_dim = 6
effective_lagrangian = integrate(
    heavy_fields, interaction_lagrangian, max_dim)

# Transformations of the effective Lagrangian

fierz_rule = (
    Op(sigma(0, -1, -2), sigma(0, -3, -4)),
    OpSum(number_op(2) * Op(kdelta(-1, -4), kdelta(-3, -2)),
          -Op(kdelta(-1, -2), kdelta(-3, -4))))
	      
Ophi = tensor_op("Ophi")
Ophi4 = tensor_op("Ophi4")

definition_rules = [
    (Op(phic(0), phi(0), phic(1), phi(1), phic(2), phi(2)),
     OpSum(Ophi)),
    (Op(phic(0), phi(0), phic(1), phi(1)),
     OpSum(Ophi4))]

rules = [fierz_rule] + definition_rules
max_iterations = 2
transf_eff_lag = apply_rules(
    effective_lagrangian, rules, max_iterations)

# Output

final_op_names = ["Ophi", "Ophi4"]
eff_lag_writer = Writer(transf_eff_lag, final_op_names)
eff_lag_writer.write_text_file("simple_example")


latex_tensor_reps = {"kappa": r"\kappa",
                     "lamb": r"\lambda",
                     "MXi": r"M_{{\Xi}}",
                     "phi": r"\phi_{}",
                     "phic": r"\phi^*_{}"}

latex_op_reps = {"Ophi": r"\mathcal{{O}}_{{\phi}}",
	         "Ophi4": r"\mathcal{{O}}_{{\phi 4}}"}
		   
latex_indices = ["i", "j", "k", "l"]
eff_lag_writer.write_pdf(
    "simple_example", latex_tensor_reps, 
    latex_op_reps, latex_indices)
