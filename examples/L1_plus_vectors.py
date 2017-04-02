"""
This script defines the hypercharge 1/2 doublet vector L1
that couples linearly through renormalizable interactions to the 
Standard Model. Its interactions with the Standard Model and other
vectors are specified. Their are integrated out and the contributions
that don't contain L1 are removed. The Lagrangian is then expressed in 
terms of the basis defined in `efttools.extras.SM_dim_6_basis`.
"""

import context
import sys


# -- Core tools --------------------------------------------------------------

from efttools.operators import (
    TensorBuilder, FieldBuilder, D, Op, OpSum,
    number_op, symbol_op, tensor_op, flavor_tensor_op,
    boson, fermion, kdelta)

from efttools.integration import integrate, ComplexVector

from efttools.transformations import apply_rules, group_op_sum

from efttools.output import Writer


# -- Predefined tensors and rules --------------------------------------------

from efttools.extras.SM import (
    mu2phi, lambdaphi, ye, yec, yd, ydc, yu, yuc, V, Vc, gb, gw,
    phi, phic, lL, lLc, qL, qLc, eR, eRc, dR, dRc, uR, uRc,
    bFS, wFS, gFS, eoms_SM, latex_SM)

from efttools.extras.Lorentz import (
    sigma4, sigma4bar, epsUp, epsUpDot, epsDown, epsDownDot, eps4,
    rules_Lorentz, latex_Lorentz)

from efttools.extras.SU2 import epsSU2, sigmaSU2, rules_SU2, latex_SU2

from efttools.extras.SU3 import TSU3, epsSU3, rules_SU3, latex_SU3

from efttools.extras.SM_dim_6_basis import (
    O1phil, O1philc, O3phil, O3philc, O1phiq, O1phiqc, O3phiq, O3phiqc,
    OphiB, OphiW, OWB, OphiBTilde, OphiWTilde, OWBTilde,
    rules_basis_definitions, latex_basis_coefs)


# -- Import from the example vectors.py --------------------------------------

from vectors import (
    B, W, B1, B1c, W1, W1c,
    L_vectors, heavy_vectors, latex_tensors_vectors)


# -- Tensors -----------------------------------------------------------------

gamma = TensorBuilder("gamma")
gammac = TensorBuilder("gammac")
zetaB = TensorBuilder("zetaB")
zetaBc = TensorBuilder("zetaBc")
zetaW = TensorBuilder("zetaW")
zetaWc = TensorBuilder("zetaWc")
zetaB1 = TensorBuilder("zetaB1")
zetaB1c = TensorBuilder("zetaB1c")
zetaW1 = TensorBuilder("zetaW1")
zetaW1c = TensorBuilder("zetaW1c")
gB = TensorBuilder("gB")
gW = TensorBuilder("gW")
gTildeB = TensorBuilder("gTildeB")
gTildeW = TensorBuilder("gTildeW")
h1 = TensorBuilder("h1")
h2 = TensorBuilder("h2")
h3 = TensorBuilder("h3")
h3c = TensorBuilder("h3c")


# -- Fields ------------------------------------------------------------------

L1 = FieldBuilder("L1", 1, boson)
L1c = FieldBuilder("L1c", 1, boson)

# -- Lagrangian --------------------------------------------------------------

L_L1_plus_vectors =  -OpSum(
    Op(gamma(0), L1c(1, 2, 0), D(1, phi(2))),
    Op(gammac(0), L1(1, 2, 0), D(1, phic(2))),

    Op(zetaB(0, 1), L1c(2, 3, 0), phi(3), B(2, 1)),
    Op(zetaBc(0, 1), L1(2, 3, 0), phic(3), B(2, 1)),
    Op(zetaW(0, 1), L1c(2, 3, 0), sigmaSU2(4, 3, 5), phi(5), W(2, 4, 1)),
    Op(zetaWc(0, 1), phic(2), sigmaSU2(3, 2, 4), L1(5, 4, 0), W(5, 3, 1)),
    Op(zetaB1(0, 1), epsSU2(2, 3), L1(4, 3, 0), phi(2), B1c(4, 1)),
    Op(zetaB1c(0, 1), phic(2), epsSU2(2, 3), L1c(4, 3, 0), B1(4, 1)),
    Op(zetaW1(0, 1), epsSU2(2, 3), L1(4, 3, 0), sigmaSU2(5, 2, 6),
       phi(6), W1c(4, 5, 1)),
    Op(zetaW1c(0, 1), phic(2), sigmaSU2(3, 2, 4),
       epsSU2(4, 5), L1c(6, 5, 0), W1(6, 3, 1)),

    number_op(1j) * Op(gB(0, 1), L1c(2, 3, 0), L1(4, 3, 1), bFS(2, 4)),
    number_op(1j) * Op(gW(0, 1), L1c(2, 3, 0), sigmaSU2(4, 3, 5),
                       L1(6, 5, 1), wFS(2, 6, 4)),
    number_op(1j) * Op(gTildeB(0, 1), L1c(2, 3, 0), L1(4, 3, 1),
                       eps4(2, 4, 5, 6), bFS(5, 6)),
    number_op(1j) * Op(gTildeW(0, 1), L1c(2, 3, 0), sigmaSU2(4, 3, 5),
                       L1(6, 5, 1), eps4(2, 6, 7, 8), wFS(7, 8, 4)),

    Op(h1(0, 1), L1c(2, 3, 0), L1(2, 3, 1), phic(4), phi(4)),
    Op(h2(0, 1), L1c(2, 3, 0), phi(3), phic(4), L1(2, 4, 1)),
    Op(h3(0, 1), L1c(2, 3, 0), phi(3), L1c(2, 4, 1), phi(4)),
    Op(h3c(0, 1), phic(2), L1(3, 2, 0), phic(4), L1(3, 4, 1)))

# -- Heavy fields ------------------------------------------------------------

heavy_L1 = ComplexVector("L1", "L1c", 3)

heavy_vectors_plus_L1 = heavy_vectors + [heavy_L1]


# -- LaTeX representation ----------------------------------------------------

latex_tensors_L1 = {
    "gamma": "\\gamma_{{{}}}",
    "gammac": "\\gamma^*_{{{}}}",
    "zetaB": "\\zeta^{{{}{}}}_{{\\mathcal{{B}}}}",
    "zetaBc": "\\zeta^{{{}{}*}}_{{\\mathcal{{B}}}}",
    "zetaW": "\\zeta^{{{}{}}}_{{\mathcal{{W}}}}",
    "zetaWc": "\\zeta^{{{}{}*}}_{{\mathcal{{W}}}}",
    "zetaB1": "\\zeta^{{{}{}}}_{{\mathcal{{B}}_1}}",
    "zetaB1c": "\\zeta^{{{}{}*}}_{{\mathcal{{B}}_1}}",
    "zetaW1": "\\zeta^{{{}{}}}_{{\mathcal{{W}}_1}}",
    "zetaW1c": "\\zeta^{{{}{}*}}_{{\mathcal{{W}}_1}}",
    "gB": "g^{{{}{}}}_B",
    "gW": "g^{{{}{}}}_W",
    "gTildeB": "\\tilde{{g}}^{{{}{}}}_B",
    "gTildeW": "\\tilde{{g}}^{{{}{}}}_W",
    "h1": "h^{{(1)}}_{{{}{}}}",
    "h2": "h^{{(2)}}_{{{}{}}}",
    "h3": "h^{{(3)}}_{{{}{}}}",
    "h3c": "h^{{(3)*}}_{{{}{}}}",

    "ML1": "M_{{\mathcal{{L}}^1_{}}}"}

if __name__ == "__main__":
    
    # -- Integration ---------------------------------------------------------
    
    eff_lag = integrate(heavy_vectors_plus_L1,
                        L_vectors + L_L1_plus_vectors, 6)


    # -- Transformations -----------------------------------------------------
    
    # Remove operators without ML1
    
    L1_eff_lag = group_op_sum(OpSum(*[op for op in eff_lag.operators
                                  if op.contains_symbol("ML1")]))

    # Define specific rules for this case

    specific_rules = [

        # Higgs and derivatives
    
        (Op(D(0, phic(1)), phi(1), D(0, phic(2)), phi(2)),
         -OpSum(Op(D(0, phic(1)), phi(1), phic(2), D(0, phi(2))),
                Op(D(0, phic(1)), D(0, phi(1)), phic(2), phi(2)),
                Op(D(1, D(1, phic(0))), phi(0), phic(2), phi(2)))),

        (Op(phic(1), D(0, phi(1)), phic(2), D(0, phi(2))),
         -OpSum(Op(D(0, phic(1)), phi(1), phic(2), D(0, phi(2))),
                Op(D(0, phic(1)), D(0, phi(1)), phic(2), phi(2)),
                Op(phic(1), D(0, D(0, phi(1))), phic(2), phi(2)))),

        
        # Higgs and fermion doublets

        (Op(lLc(0, 1, -1), sigma4bar(2, 0, 3), lL(3, 4, -2),
            phic(4), D(2, phi(1))),
         OpSum(number_op(-0.5j) * O3phil(-1, -2),
               number_op(-0.5j) * O1phil(-1, -2))),

        (Op(lLc(0, 1, -2), sigma4bar(2, 0, 3), lL(3, 4, -1),
            D(2, phic(4)), phi(1)),
         OpSum(number_op(0.5j) * O3philc(-1, -2),
           number_op(0.5j) * O1philc(-1, -2))),

        (Op(qLc(0, 5, 1, -1), sigma4bar(2, 0, 3), qL(3, 5, 4, -2),
            phic(4), D(2, phi(1))),
         OpSum(number_op(-0.5j) * O3phiq(-1, -2),
               number_op(-0.5j) * O1phiq(-1, -2))),

        (Op(qLc(0, 5, 1, -2), sigma4bar(2, 0, 3), qL(3, 5, 4, -1),
            D(2, phic(4)), phi(1)),
         OpSum(number_op(0.5j) * O3phiqc(-1, -2),
               number_op(0.5j) * O1phiqc(-1, -2))),

        
        # Involving field strengths and Higgs doublets

        (Op(D(0, phic(1)), D(2, phi(1)), bFS(0, 2)),
         OpSum(number_op(-0.5) * Op(phic(1), D(0, D(2, phi(1))), bFS(0, 2)),
               number_op(0.5) * Op(D(0, D(2, phic(1))), phi(1), bFS(0, 2)),
               number_op(-0.5) * Op(phic(1), D(2, phi(1)), D(0, bFS(0, 2))),
               number_op(0.5) * Op(D(2, phic(1)), phi(1), D(0, bFS(0, 2))))),

        (Op(D(0, phic(1)), sigmaSU2(3, 1, 4), D(2, phi(4)), wFS(0, 2, 3)),
         OpSum(number_op(-0.5) * Op(phic(1), sigmaSU2(3, 1, 4),
                                    D(0, D(2, phi(4))), wFS(0, 2, 3)),
               number_op(0.5) * Op(D(0, D(2, phic(1))), sigmaSU2(3, 1, 4),
                                   phi(4), wFS(0, 2, 3)),
               number_op(-0.5) * Op(phic(1), sigmaSU2(3, 1, 4),
                                    D(2, phi(4)), D(0, wFS(0, 2, 3))),
               number_op(0.5) * Op(D(2, phic(1)), sigmaSU2(3, 1, 4),
                                   phi(4), D(0, wFS(0, 2, 3))))),

        (Op(D(0, phic(1)), D(2, phi(1)), eps4(0, 2, 3, 4), bFS(3, 4)),
         OpSum(number_op(-0.5) * Op(phic(1), D(0, D(2, phi(1))),
                                    eps4(0, 2, 3, 4), bFS(3, 4)),
               number_op(0.5) * Op(D(0, D(2, phic(1))), phi(1),
                                   eps4(0, 2, 3, 4), bFS(3, 4)))),

        (Op(D(0, phic(1)), sigmaSU2(3, 1, 4), D(2, phi(4)),
            eps4(0, 2, 5, 6), wFS(5, 6, 3)),
         OpSum(number_op(-0.5) * Op(phic(1), sigmaSU2(3, 1, 4),
                                    D(0, D(2, phi(4))),
                                    eps4(0, 2, 5, 6), wFS(5, 6, 3)),
               number_op(0.5) * Op(D(0, D(2, phic(1))), sigmaSU2(3, 1, 4),
                                   phi(4), eps4(0, 2, 5, 6), wFS(5, 6, 3)))),

        (Op(phic(2), D(0, D(1, phi(2))), bFS(0, 1)),
         OpSum(number_op(-0.25j) * Op(gb()) * OphiB,
               number_op(-0.25j) * Op(gw()) * OWB)),
        (Op(D(0, D(1, phic(2))), phi(2), bFS(0, 1)),
         OpSum(number_op(0.25j) * Op(gb()) * OphiB,
               number_op(0.25j) * Op(gw()) * OWB)),

        (Op(phic(2), sigmaSU2(3, 2, 4), D(0, D(1, phi(4))), wFS(0, 1, 3)),
         OpSum(number_op(-0.25j) * Op(gb()) * OWB,
               number_op(-0.25j) * Op(gw()) * OphiW)),
        (Op(D(0, D(1, phic(2))), sigmaSU2(3, 2, 4), phi(4), wFS(0, 1, 3)),
         OpSum(number_op(0.25j) * Op(gb()) * OWB,
               number_op(0.25j) * Op(gw()) * OphiW)),

        (Op(phic(2), D(0, D(1, phi(2))), eps4(0, 1, 3, 4), bFS(3, 4)),
         OpSum(number_op(-0.25j) * Op(gb()) * OphiBTilde,
               number_op(-0.25j) * Op(gw()) * OWBTilde)),
        (Op(D(0, D(1, phic(2))), phi(2), eps4(0, 1, 3, 4), bFS(3, 4)),
         OpSum(number_op(0.25j) * Op(gb()) * OphiBTilde,
               number_op(0.25j) * Op(gw()) * OWBTilde)),

        (Op(phic(2), sigmaSU2(3, 2, 4), D(0, D(1, phi(4))),
            eps4(0, 1, 5, 6), wFS(5, 6, 3)),
         OpSum(number_op(-0.25j) * Op(gb()) * OWBTilde,
               number_op(-0.25j) * Op(gw()) * OphiWTilde)),
        (Op(D(0, D(1, phic(2))), sigmaSU2(3, 2, 4), phi(4),
            eps4(0, 1, 5, 6), wFS(5, 6, 3)),
         OpSum(number_op(0.25j) * Op(gb()) * OWBTilde,
               number_op(0.25j) * Op(gw()) * OphiWTilde)),

        
        # With four covariant derivatives and two Higgs doublets
        
        (Op(D(0, D(1, phic(2))), D(0, D(1, phi(2)))),
         OpSum(-number_op(0.5) * Op(D(0, D(0, D(1, phic(2)))), D(1, phi(2))),
               -number_op(0.5) * Op(D(1, phic(2)),
                                    D(0, D(0, D(1, phi(2))))))),

        (Op(D(0, D(1, phic(2))), D(1, D(0, phi(2)))),
         OpSum(-number_op(0.5) * Op(D(1, D(0, D(1, phic(2)))), D(0, phi(2))),
               -number_op(0.5) * Op(D(1, phic(2)),
                                    D(0, D(1, D(0, phi(2))))))),

        (Op(D(0, phic(1)), D(2, D(0, D(2, phi(1))))),
         OpSum(-Op(D(0, D(0, phic(1))), D(2, D(2, phi(1)))),
               number_op(0.5j) * Op(gb(), bFS(0, 1), D(0, phic(2)),
                                    D(1, phi(2))),
               number_op(0.5j) * Op(gw(), wFS(0, 1, 2), D(0, phic(3)),
                                    sigmaSU2(2, 3, 4), D(1, phi(4))))),

        (Op(D(0, D(2, D(0, phic(1)))), D(2, phi(1))),
         OpSum(-Op(D(0, D(0, phic(1))), D(2, D(2, phi(1)))),
               number_op(0.5j) * Op(gb(), bFS(0, 1), D(0, phic(2)),
                                    D(1, phi(2))),
               number_op(0.5j) * Op(gw(), wFS(0, 1, 2), D(0, phic(3)),
                                    sigmaSU2(2, 3, 4), D(1, phi(4))))),

        (Op(D(0, phic(1)), D(2, D(2, D(0, phi(1))))),
         OpSum(-Op(D(0, D(0, phic(1))), D(2, D(2, phi(1)))),
               number_op(1j) * Op(gb(), D(0, phic(1)), D(2, phi(1)),
                                  bFS(0, 2)),
               number_op(1j) * Op(gw(), D(0, phic(1)), sigmaSU2(2, 1, 3),
                              D(4, phi(3)), wFS(0, 4, 2)),
               number_op(-0.5j) * Op(gb(), D(0, phic(1)), phi(1),
                                     D(2, bFS(2, 0))),
               number_op(-0.5j) * Op(gw(), D(0, phic(1)), sigmaSU2(2, 1, 3),
                                     phi(3), D(4, wFS(4, 0, 2))))),

        (Op(D(0, D(0, D(1, phic(2)))), D(1, phi(2))),
         OpSum(-Op(D(0, D(0, phic(1))), D(2, D(2, phi(1)))),
               number_op(1j) * Op(gb(), D(0, phic(1)), D(2, phi(1)),
                                  bFS(0, 2)),
               number_op(1j) * Op(gw(), D(0, phic(1)), sigmaSU2(2, 1, 3),
                                  D(4, phi(3)), wFS(0, 4, 2)),
               number_op(0.5j) * Op(gb(), phic(1), D(0, phi(1)),
                                    D(2, bFS(2, 0))),
               number_op(0.5j) * Op(gw(), phic(1), sigmaSU2(2, 1, 3),
                                    D(0, phi(3)),D(4, wFS(4, 0, 2)))))]
    
    transpose_epsSU2 = [(Op(epsSU2(-1, -2)), OpSum(Op(epsSU2(-2, -1))))]


    all_rules = (specific_rules + rules_SU2 + rules_SU3 + rules_Lorentz +
                 eoms_SM + rules_basis_definitions + transpose_epsSU2)

    transf_eff_lag = apply_rules(L1_eff_lag, all_rules, 4)


    # -- Output --------------------------------------------------------------
    
    transf_eff_lag_writer = Writer(transf_eff_lag,
                                   latex_basis_coefs.keys())

    sys.stdout.write(str(transf_eff_lag_writer) + "\n")

    latex_tensors = {}
    latex_tensors.update(latex_tensors_L1)
    latex_tensors.update(latex_tensors_vectors)
    latex_tensors.update(latex_SM)
    latex_tensors.update(latex_SU2)
    latex_tensors.update(latex_SU3)
    latex_tensors.update(latex_Lorentz)

    transf_eff_lag_writer.show_pdf(
        "L1_plus_vectors", "open", latex_tensors, latex_basis_coefs,
        list(map(chr, range(ord('a'), ord('z')))))
