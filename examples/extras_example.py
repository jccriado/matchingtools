"""
Simple example to illustrate some of the features of `matchingtools`.
We consider the Standard Model with two families of heavy quarks
added: SU(2) singlets :math:`D_i` with hypercharge :math:`-1/3`
and doublets :math:`(DY)_j` with hypercharge :math:`-5/6`
with interaction lagrangian:

.. math::
    \mathcal{L}_\int =
    - \lambda^{Dq}_{ij} \bar{D}_{Ri} \phi^\dagger q_{Lj}
    - \lambda^{(DY)u}_{ij} (\bar{D} \bar{Y})_{Li} \tilde{\phi} d_{Rj}
    - \lambda^{(DY)D}_{ij} (\bar{D} \bar{Y})_{Li} \tilde{\phi} D_{Rj}
    + h.c.

We will integrate out both heavy quarks and write the final result
in the operator basis:

.. math::
    \mathcal{O}_{u \phi} = (\phi^\dagger \phi) (\bar{q}_L \tilde{\phi} u_R)
    \mathcal{O}_{d \phi} = (\phi^\dagger \phi) (\bar{q}_L \\phi d_R)
    \mathcal{O}^{(1)}_{\phi u} = 
        (\phi^\dagger D_\mu \phi) (\bar{u}_R \gamma^\mu u_R)
    \mathcal{O}^{(1)}_{\phi d} = 
        (\phi^\dagger D_\mu \phi) (\bar{d}_R \gamma^\mu d_R)
    \mathcal{O}^{(1)}_{\phi q} = 
        (\phi^\dagger D_\mu \phi) (\bar{q}_R \gamma^\mu q_R)
    \mathcal{O}^{(3)}_{\phi q} = 
        (\phi^\dagger \sigma^a D_\mu \phi) (\bar{q}_R \gamma^\mu \sigma^a q_R)

(plus their complex conjugates)

"""

import context

from matchingtools.core import (
    TensorBuilder, FieldBuilder, D, Op, OpSum,
    number_op, flavor_tensor_op, boson, fermion, kdelta,
    sigma4, sigma4bar)

from matchingtools.integration import (VectorLikeFermion, integrate)

from matchingtools.transformations import apply_rules

from matchingtools.output import Writer

from matchingtools.extras.SM import (
    phi, phic, qL, qLc, uR, uRc, dR, dRc, eoms_SM)

from matchingtools.extras.SU2 import (epsSU2, rules_SU2)

import matchingtools.extras.SM_dim_6_basis as basis

# Model

lambdaDq = TensorBuilder("lambdaDq")
lambdaDqc = TensorBuilder("lambdaDqc")
lambdaDYd = TensorBuilder("lambdaDYd")
lambdaDYdc = TensorBuilder("lambdaDYdc")

lambdaDYD = TensorBuilder("lambdaDYD")
lambdaDYDc = TensorBuilder("lambdaDYDc")

DL = FieldBuilder("DL", 1.5, fermion)
DR = FieldBuilder("DR", 1.5, fermion)
DLc = FieldBuilder("DLc", 1.5, fermion)
DRc = FieldBuilder("DRc", 1.5, fermion)
DYL = FieldBuilder("DYL", 1.5, fermion)
DYR = FieldBuilder("DYR", 1.5, fermion)
DYLc = FieldBuilder("DYLc", 1.5, fermion)
DYRc = FieldBuilder("DYRc", 1.5, fermion)

interaction_lagrangian = -OpSum(
    # Linear part
    Op(lambdaDq(0, 1), DRc(2, 3, 0), phic(4), qL(2, 3, 4, 1)),
    Op(lambdaDqc(0, 1), qLc(2, 3, 4, 1), phi(4), DR(2, 3, 0)),
    Op(lambdaDYd(0, 1), DYLc(2, 3, 4, 0),
       epsSU2(4, 5), phic(5), dR(2, 3, 1)),
    Op(lambdaDYdc(0, 1), dRc(2, 3, 1),
       epsSU2(4, 5), phi(5), DYL(2, 3, 4, 0)),

    # D - DY interaction
    Op(lambdaDYD(0, 1), DYRc(2, 3, 4, 0), epsSU2(4, 5), phic(5), DL(2, 3, 1)),
    Op(lambdaDYDc(0, 1), DLc(2, 3, 1), epsSU2(4, 5), phi(5), DYR(2, 3, 4, 0)))

# Integration

heavy_D = VectorLikeFermion("D", "DL", "DR", "DLc", "DRc", 3)
heavy_DY = VectorLikeFermion("DY", "DYL", "DYR", "DYLc", "DYRc", 4)

heavy_fields = [heavy_D, heavy_DY]
effective_lagrangian = integrate(
    heavy_fields, interaction_lagrangian, 6)

# Transformations of the effective Lgrangian

extra_SU2_identities = [
    (Op(phi(0), epsSU2(0, 1), phi(1)),
     OpSum()),

    (Op(phic(0), epsSU2(0, 1), phic(1)),
     OpSum()),

    (Op(qLc(0, 5, 1, -1), sigma4bar(2, 0, 3), qL(3, 5, 4, -2),
        phic(4), D(2, phi(1))),
     OpSum(number_op(-0.5j) * basis.O3phiq(-1, -2),
           number_op(-0.5j) * basis.O1phiq(-1, -2))),

    (Op(qLc(0, 5, 1, -2), sigma4bar(2, 0, 3), qL(3, 5, 4, -1),
        D(2, phic(4)), phi(1)),
     OpSum(number_op(0.5j) * basis.O3phiqc(-1, -2),
           number_op(0.5j) * basis.O1phiqc(-1, -2)))]

rules = (rules_SU2 + eoms_SM + extra_SU2_identities +
         basis.rules_basis_definitions)

transf_eff_lag = apply_rules(effective_lagrangian, rules, 2)

# Output

eff_lag_writer = Writer(transf_eff_lag, basis.latex_basis_coefs.keys())
eff_lag_writer.write_text_file("extras_example_results.txt")

