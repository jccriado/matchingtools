"""
This script defines all the heavy vectors
that couple linearly through renormalizable interactions to the 
Standard Model (except for the hypercharge 1/2 doublet). 
It specifies their interaction lagrangian and integrates them out.
"""

import context
import sys


# -- Core tools --------------------------------------------------------------

from effective.operators import (
    TensorBuilder, FieldBuilder, D, Op, OpSum,
    number_op, symbol_op, tensor_op, flavor_tensor_op,
    boson, fermion, kdelta)

from effective.integration import integrate, RealVector, ComplexVector

from effective.output import Writer


# -- Predefined tensors and rules --------------------------------------------

from effective.extras.SM import (
    mu2phi, lambdaphi, ye, yec, yd, ydc, yu, yuc, V, Vc,
    phi, phic, lL, lLc, qL, qLc, eR, eRc, dR, dRc, uR, uRc,
    bFS, wFS, gFS, latex_SM)

from effective.extras.Lorentz import (
    sigma4, sigma4bar, epsUp, epsUpDot, epsDown, epsDownDot, latex_Lorentz)

from effective.extras.SU2 import epsSU2, sigmaSU2, latex_SU2

from effective.extras.SU3 import TSU3, epsSU3, latex_SU3


# -- Tensors -----------------------------------------------------------------

glB = TensorBuilder("glB")
gqB = TensorBuilder("gqB")
geB = TensorBuilder("geB")
gdB = TensorBuilder("gdB")
guB = TensorBuilder("guB")
gphiB = TensorBuilder("gphiB")
gphiBc = TensorBuilder("gphiBc")
glW = TensorBuilder("glW")
glWc = TensorBuilder("glWc")
gqW = TensorBuilder("gqW")
gphiW = TensorBuilder("gphiW")
gphiWc = TensorBuilder("gphiWc")
gqG = TensorBuilder("gqG")
guG = TensorBuilder("guG")
gdG = TensorBuilder("gdG")
gqH = TensorBuilder("gqH")
gduB1 = TensorBuilder("gduB1")
gduB1c = TensorBuilder("gduB1c")
gphiB1 = TensorBuilder("gphiB1")
gphiB1c = TensorBuilder("gphiB1c")
gphiW1 = TensorBuilder("gphiW1")
gphiW1c = TensorBuilder("gphiW1c")
gduG1 = TensorBuilder("gduG1")
gduG1c = TensorBuilder("gduG1c")
gelL3 = TensorBuilder("gelL3")
gelL3c = TensorBuilder("gelL3c")
gedU2 = TensorBuilder("gedU2")
gedU2c = TensorBuilder("gedU2c")
glqU2 = TensorBuilder("glqU2")
glqU2c = TensorBuilder("glqU2c")
geuU5 = TensorBuilder("geuU5")
geuU5c = TensorBuilder("geuU5c")
gulQ1 = TensorBuilder("gulQ1")
gulQ1c = TensorBuilder("gulQ1c")
gdqQ1 = TensorBuilder("gdqQ1")
gdqQ1c = TensorBuilder("gdqQ1c")
gdlQ5 = TensorBuilder("gdlQ5")
gdlQ5c = TensorBuilder("gdlQ5c")
geqQ5 = TensorBuilder("geqQ5")
geqQ5c = TensorBuilder("geqQ5c")
guqQ5 = TensorBuilder("guqQ5")
guqQ5c = TensorBuilder("guqQ5c")
glqX = TensorBuilder("glqX")
glqXc = TensorBuilder("glqXc")
gdqY1 = TensorBuilder("gdqY1")
gdqY1c = TensorBuilder("gdqY1c")
guqY5 = TensorBuilder("guqY5")
guqY5c = TensorBuilder("guqY5c")


# -- Fields ------------------------------------------------------------------

# Neutral singlet
B = FieldBuilder("B", 1, boson)

# Neutral SU(2) triplet
W = FieldBuilder("W", 1, boson)

# Neutral SU(3) octet
G = FieldBuilder("G", 1, boson)

# Neutral SU(3) octet, SU(2) triplet
H = FieldBuilder("H", 1, boson)

# Hypercharge 1 singlet
B1 = FieldBuilder("B1", 1, boson)
B1c = FieldBuilder("B1c", 1, boson)

# Hypercharge 1 SU(2) triplet
W1 = FieldBuilder("W1", 1, boson)
W1c = FieldBuilder("W1c", 1, boson)

# Hypercharge 1 SU(3) octet
G1 = FieldBuilder("G1", 1, boson)
G1c = FieldBuilder("G1c", 1, boson)

# Hypercharge -3/2 SU(2) doublet
L3 = FieldBuilder("L3", 1, boson)
L3c = FieldBuilder("L3c", 1, boson)

# Hypercharge 2/3 SU(3) triplet
U2 = FieldBuilder("U2", 1, boson)
U2c = FieldBuilder("U2c", 1, boson)

# Hypercharge 5/3 SU(3) triplet
U5 = FieldBuilder("U5", 1, boson)
U5c = FieldBuilder("U5c", 1, boson)

# Hypercharge 1/6 SU(3) triplet, SU(2) doublet
Q1 = FieldBuilder("Q1", 1, boson)
Q1c = FieldBuilder("Q1c", 1, boson)

# Hypercharge -5/6 SU(3) triplet, SU(2) doublet
Q5 = FieldBuilder("Q5", 1, boson)
Q5c = FieldBuilder("Q5c", 1, boson)

# Hypercharge 2/3 SU(3) triplet, SU(2) triplet
X = FieldBuilder("X", 1, boson)
Xc = FieldBuilder("Xc", 1, boson)

# Hypercharge 1/6 SU(3) antisextet, SU(2) doublet
# (the antisextet index is represented by two SU(3) triplet indices
# for which the field is antisymmetric)
Y1 = FieldBuilder("Y1", 1, boson)
Y1c = FieldBuilder("Y1c", 1, boson)

# Hypercharge -5/6 SU(3) antisextet, SU(2) doublet
# (the antisextet index is represented by two SU(3) triplet indices
# for which the field is antisymmetric)
Y5 = FieldBuilder("Y5", 1, boson)
Y5c = FieldBuilder("Y5c", 1, boson)

# -- Lagrangian --------------------------------------------------------------

L_vectors = -OpSum(
    # B
    Op(glB(0, 1, 2), B(3, 0), lLc(4, 5, 1), sigma4bar(3, 4, 6), lL(6, 5, 2)),
    Op(gqB(0, 1, 2), B(3, 0), qLc(4, 5, 6, 1), sigma4bar(3, 4, 7),
       qL(7, 5, 6, 2)),
    Op(geB(0, 1, 2), B(3, 0), eRc(4, 1), sigma4(3, 4, 5), eR(5, 2)),
    Op(gdB(0, 1, 2), B(3, 0), dRc(4, 5, 1), sigma4(3, 4, 6), dR(6, 5, 2)),
    Op(guB(0, 1, 2), B(3, 0), uRc(4, 5, 1), sigma4(3, 4, 6), uR(6, 5, 2)),
    number_op(1j) * Op(gphiB(0), B(1, 0), phic(2), D(1, phi(2))),
    number_op(-1j) * Op(gphiBc(0), B(1, 0), phi(2), D(1, phic(2))),

    # W
    number_op(0.5) * Op(glW(0, 1, 2), W(3, 4, 0), lLc(5, 6, 1),
                        sigma4bar(3, 5, 7), sigmaSU2(4, 6, 8), lL(7, 8, 2)),
    number_op(0.5) * Op(gqW(0, 1, 2), W(3, 4, 0), qLc(5, 6, 7, 1),
                        sigma4bar(3, 5, 8), sigmaSU2(4, 7, 9),
                        qL(8, 6, 9, 2)),
    number_op(0.5j) * Op(gphiW(0), W(1, 2, 0), phic(3), sigmaSU2(2, 3, 4),
                         D(1, phi(4))),
    number_op(-0.5j) * Op(gphiWc(0), W(1, 2, 0), D(1, phic(3)),
                          sigmaSU2(2, 3, 4), phi(4)),

    # G
    Op(gqG(0, 1, 2), G(3, 4, 0), qLc(5, 6, 7, 1), sigma4bar(3, 5, 8),
       TSU3(4, 6, 9), qL(8, 9, 7, 2)),
    Op(guG(0, 1, 2), G(3, 4, 0), uRc(5, 6, 1), sigma4(3, 5, 7),
       TSU3(4, 6, 8), uR(7, 8, 2)),
    Op(gdG(0, 1, 2), G(3, 4, 0), dRc(5, 6, 1), sigma4(3, 5, 7),
       TSU3(4, 6, 8), dR(7, 8, 2)),

    # H
    number_op(0.5) * Op(gqH(0, 1, 2), H(3, 4, 5, 0), qLc(6, 7, 8, 1),
                        sigma4bar(3, 6, 9), sigmaSU2(5, 8, 10),
                        TSU3(4, 7, 11), qL(9, 11, 10, 2)),
    
    # B1
    Op(gduB1(0, 1, 2), B1c(3, 0), dRc(4, 5, 1), sigma4(3, 4, 6), uR(6, 5, 2)),
    Op(gduB1c(0, 1, 2), B1(3, 0), uRc(4, 5, 2), sigma4(3, 4, 6), dR(6, 5, 1)),
    number_op(1j) * Op(gphiB1(0), B1c(1, 0), D(1, phi(2)), epsSU2(2, 3), phi(3)),
    number_op(-1j) * Op(gphiB1c(0), B1(1, 0), D(1, phic(2)), epsSU2(2, 3), phic(3)),

    # W1
    number_op(0.5j) * Op(gphiW1(0), W1c(1, 2, 0), D(1, phi(3)),
                         epsSU2(3, 4), sigmaSU2(2, 4, 5), phi(5)),
    number_op(-0.5j) * Op(gphiW1c(0), W1(1, 2, 0), D(1, phic(3)),
                          epsSU2(3, 4), sigmaSU2(2, 5, 4), phic(5)),
    
    # G1
    Op(gduG1(0, 1, 2), G1c(3, 4, 0), dRc(5, 6, 1), TSU3(4, 6, 7),
       sigma4(3, 5, 8), uR(8, 7, 2)),
    Op(gduG1c(0, 1, 2), G1(3, 4, 0), uRc(8, 7, 2), TSU3(4, 7, 6),
       sigma4(3, 8, 5), dR(5, 6, 1)),

    # L3
    Op(gelL3(0, 1, 2), L3c(3, 4, 0), eR(5, 1),
       epsDownDot(6, 5), sigma4(3, 6, 7), lL(7, 4, 2)),
    Op(gelL3c(0, 1, 2), L3(3, 4, 0), lLc(7, 4, 2),
       sigma4(3, 7, 6), epsDown(6, 5), eRc(5, 1)),

    # U2
    Op(gedU2(0, 1, 2), U2c(3, 4, 0), eRc(5, 1), sigma4(3, 5, 6), dR(6, 4, 2)),
    Op(gedU2c(0, 1, 2), U2(3, 4, 0), dRc(6, 4, 2), sigma4(3, 6, 5), eR(5, 1)),
    Op(glqU2(0, 1, 2), U2c(3, 4, 0), lLc(5, 6, 1),
       sigma4bar(3, 5, 7), qL(7, 4, 6, 2)),
    Op(glqU2c(0, 1, 2), U2(3, 4, 0), qLc(7, 4, 6, 2),
       sigma4bar(3, 7, 5), lL(5, 6, 1)),

    # U5
    Op(geuU5(0, 1, 2), U5c(3, 4, 0), eRc(5, 1), sigma4(3, 5, 6), uR(6, 4, 2)),
    Op(geuU5c(0, 1, 2), U5(3, 4, 0), uRc(6, 4, 2), sigma4(3, 6, 5), eR(5, 1)),

    # Q1
    Op(gulQ1(0, 1, 2), Q1c(3, 4, 5, 0), uR(6, 4, 1), epsDownDot(7, 6),
       sigma4bar(3, 7, 8), lL(8, 5, 2)),
    Op(gulQ1c(0, 1, 2), Q1(3, 4, 5, 0), lLc(8, 5, 2),
       sigma4bar(3, 8, 7), epsDown(7, 6), uRc(6, 4, 1)),
    Op(gdqQ1(0, 1, 2), Q1c(3, 4, 5, 0), epsSU3(4, 6, 7),
       dRc(8, 6, 1), sigma4(3, 8, 9),
       epsSU2(5, 10), epsUpDot(9, 11), qLc(11, 7, 10, 2)),
    Op(gdqQ1c(0, 1, 2), Q1(3, 4, 5, 0), epsSU3(4, 6, 7),
       qL(11, 7, 10, 2), epsUp(9, 11), epsSU2(5, 10),
       sigma4(3, 9, 8), dR(8, 6, 1)),

    # Q5
    Op(gdlQ5(0, 1, 2), Q5c(3, 4, 5, 0), dR(6, 4, 1), epsDownDot(7, 6),
       sigma4bar(3, 7, 8), lL(8, 5, 2)),
    Op(gdlQ5c(0, 1, 2), Q5(3, 4, 5, 0), lLc(8, 5, 2),
       sigma4bar(3, 8, 7), epsDown(7, 6), dRc(6, 4, 1)),
    Op(geqQ5(0, 1, 2), Q5c(3, 4, 5, 0), eR(6, 1), epsDownDot(7, 6),
       sigma4bar(3, 7, 8), qL(8, 4, 5, 2)),
    Op(geqQ5c(0, 1, 2), Q5(3, 4, 5, 0), qLc(8, 4, 5, 2),
       sigma4bar(3, 8, 7), epsDown(7, 6), eRc(6, 1)),
    Op(guqQ5(0, 1, 2), Q5c(3, 4, 5, 0), epsSU3(4, 6, 7),
       uRc(8, 6, 1), sigma4(3, 8, 9),
       epsSU2(5, 10), epsUpDot(9, 11), qLc(11, 7, 10, 2)),
    Op(guqQ5c(0, 1, 2), Q5(3, 4, 5, 0), epsSU3(4, 6, 7),
       qL(11, 7, 10, 2), epsUp(9, 11), epsSU2(5, 10),
       sigma4(3, 9, 8), uR(8, 6, 1)),

    # X
    number_op(0.5) * Op(glqX(0, 1, 2), Xc(3, 4, 5, 0), lLc(6, 7, 1),
                        sigma4bar(3, 6, 8), sigmaSU2(5, 7, 9),
                        qL(8, 4, 9, 2)),
    number_op(0.5) * Op(glqXc(0, 1, 2), X(3, 4, 5, 0), qL(8, 4, 9, 2),
                        sigma4bar(3, 8, 6), sigmaSU2(5, 9, 7), lL(6, 7, 1)),

    # Y1
    Op(gdqY1(0, 1, 2), Y1c(3, 4, 5, 6, 0), dRc(7, 4, 1), sigma4(3, 7, 8),
       epsSU2(6, 9), epsUpDot(8, 10), qLc(10, 5, 9, 2)),
    Op(gdqY1c(0, 1, 2), Y1(3, 4, 5, 6, 0), epsSU2(6, 9), qL(10, 5, 9, 2),
       epsUp(8, 10), sigma4(3, 8, 7), dR(7, 4, 1)),

    # Y5
    Op(guqY5(0, 1, 2), Y1c(3, 4, 5, 6, 0), uRc(7, 4, 1), sigma4(3, 7, 8),
       epsSU2(6, 9), epsUpDot(8, 10), qLc(10, 5, 9, 2)),
    Op(guqY5c(0, 1, 2), Y1(3, 4, 5, 6, 0), epsSU2(6, 9),
       epsUp(0, 10), sigma4(3, 8, 7), uR(7, 4, 1)),
)

# -- Heavy fields ------------------------------------------------------------

heavy_B = RealVector("B", 2)
heavy_W = RealVector("W", 3)
heavy_G = RealVector("G", 3)
heavy_H = RealVector("H", 4)
heavy_B1 = ComplexVector("B1", "B1c", 2)
heavy_W1 = ComplexVector("W1", "W1c", 3)
heavy_G1 = ComplexVector("G1", "G1c", 3)
heavy_L3 = ComplexVector("L3", "L3c", 3)
heavy_U2 = ComplexVector("U2", "U2c", 3)
heavy_U5 = ComplexVector("U5", "U5c", 3)
heavy_Q1 = ComplexVector("Q1", "Q1c", 4)
heavy_Q5 = ComplexVector("Q5", "Q5c", 4)
heavy_X = ComplexVector("X", "Xc", 4)
heavy_Y1 = ComplexVector("Y1", "Y1c", 5)
heavy_Y5 = ComplexVector("Y5", "Y5c", 5)

heavy_vectors = [heavy_B, heavy_W, heavy_G, heavy_H, heavy_B1, heavy_W1,
                heavy_G1, heavy_L3, heavy_U2, heavy_U5, heavy_Q1, heavy_Q5,
                heavy_X, heavy_Y1, heavy_Y5]
"""
All the heavy vector fields that couple linearly through renormalizable
interactions to the Standard Model.
"""


# -- LaTeX representation ----------------------------------------------------

latex_tensors_vectors = {
    "glB": r"(g^l_{{\mathcal{{B}}_{}}})_{{{}{}}}",
    "gqB": r"(g^q_{{\mathcal{{B}}_{}}})_{{{}{}}}",
    "geB": r"(g^e_{{\mathcal{{B}}_{}}})_{{{}{}}}",
    "gdB": r"(g^d_{{\mathcal{{B}}_{}}})_{{{}{}}}",
    "guB": r"(g^u_{{\mathcal{{B}}_{}}})_{{{}{}}}",
    "gphiB": r"(g^{{\phi}}_{{\mathcal{{B}}_{}}})",
    "gphiBc": r"(g^{{\phi}}_{{\mathcal{{B}}_{}}})^*",
    
    "glW": r"(g^l_{{\mathcal{{W}}_{}}})_{{{}{}}}",
    "glWc": r"(g^l_{{\mathcal{{W}}_{}}})_{{{}{}}}",
    "gqW": r"(g^q_{{\mathcal{{W}}_{}}})_{{{}{}}}",
    "gphiW": r"(g^{{\phi}}_{{\mathcal{{W}}_{}}})",
    "gphiWc": r"(g^{{\phi}}_{{\mathcal{{W}}_{}}})^*",
    
    "gqG": r"(g^q_{{\mathcal{{G}}_{}}})_{{{}{}}}",
    "guG": r"(g^u_{{\mathcal{{G}}_{}}})_{{{}{}}}",
    "gdG": r"(g^d_{{\mathcal{{W}}_{}}})_{{{}{}}}",
    
    "gqH": r"(g^q_{{\mathcal{{H}}_{}}})_{{{}{}}}",

    "gduB1": r"(g^q_{{\mathcal{{B}}^1_{}}})_{{{}{}}}",
    "gduB1c": r"(g^q_{{\mathcal{{B}}^1_{}}})^*_{{{}{}}}",
    "gphiB1": r"(g^{{\phi}}_{{\mathcal{{B}}^1_{}}})",
    "gphiB1c": r"(g^{{\phi}}_{{\mathcal{{B}}^1_{}}})^*",
    
    "gphiW1": r"(g^{{\phi}}_{{\mathcal{{W}}^1_{{1{}}}}})",
    "gphiW1c": r"(g^{{\phi}}_{{\mathcal{{W}}^1_{{1{}}}}})^*",
    
    "gduG1": r"(g^{{du}}_{{\mathcal{{G}}^1_{}}})_{{{}{}}}",
    "gduG1c": r"(g^{{du}}_{{\mathcal{{G}}^1_{}}})^*_{{{}{}}}",
    
    "gelL3": r"(g^{{el}}_{{\mathcal{{L}}^3_{}}})_{{{}{}}}",
    "gelL3c": r"(g^{{el}}_{{\mathcal{{L}}^3_{}}})^*_{{{}{}}}",
    
    "gedU2": r"(g^{{ed}}_{{\mathcal{{U}}^2_{}}})_{{{}{}}}",
    "gedU2c": r"(g^{{ed}}_{{\mathcal{{U}}^2_{}}})^*_{{{}{}}}",
    "glqU2": r"(g^{{lq}}_{{\mathcal{{U}}^2_{}}})_{{{}{}}}",
    "glqU2c": r"(g^{{lq}}_{{\mathcal{{U}}^2_{}}})^*_{{{}{}}}",
    
    "geuU5": r"(g^{{eu}}_{{\mathcal{{U}}^5_{}}})_{{{}{}}}",
    "geuU5c": r"(g^{{eu}}_{{\mathcal{{U}}^5_{}}})^*_{{{}{}}}",
    
    "gulQ1": r"(g^{{ul}}_{{\mathcal{{Q}}^1_{}}})_{{{}{}}}",
    "gulQ1c": r"(g^{{ul}}_{{\mathcal{{Q}}^1_{}}})^*_{{{}{}}}",
    "gdqQ1": r"(g^{{dq}}_{{\mathcal{{Q}}^1_{}}})_{{{}{}}}",
    "gdqQ1c": r"(g^{{dq}}_{{\mathcal{{Q}}^1_{}}})^*_{{{}{}}}",
    
    "gdlQ5": r"(g^{{dl}}_{{\mathcal{{Q}}^5_{}}})_{{{}{}}}",
    "gdlQ5c": r"(g^{{dl}}_{{\mathcal{{Q}}^5_{}}})^*_{{{}{}}}",
    "geqQ5": r"(g^{{eq}}_{{\mathcal{{Q}}^5_{}}})_{{{}{}}}",
    "geqQ5c": r"(g^{{eq}}_{{\mathcal{{Q}}^5_{}}})^*_{{{}{}}}",
    "guqQ5": r"(g^{{uq}}_{{\mathcal{{Q}}^5_{}}})_{{{}{}}}",
    "guqQ5c": r"(g^{{uq}}_{{\mathcal{{Q}}^5_{}}})^*_{{{}{}}}",

    "glqX": r"(g^{{lq}}_{{\mathcal{{X}}_{}}})_{{{}{}}}",
    "glqXc": r"(g^{{lq}}_{{\mathcal{{X}}_{}}})^*_{{{}{}}}",
    
    "gdqY1": r"(g^{{dq}}_{{\mathcal{{pY}}^1_{}}})_{{{}{}}}",
    "gdqY1c": r"(g^{{dq}}_{{\mathcal{{Y}}^1_{}}})^*_{{{}{}}}",
    
    "guqY5": r"(g^{{uq}}_{{\mathcal{{Y}}^5_{}}})_{{{}{}}}",
    "guqY5c": r"(g^{{uq}}_{{\mathcal{{Y}}^5_{}}})_{{{}{}}}",

    # Masses
    "MB": "M_{{\mathcal{{B}}_{}}}",
    "MW": "M_{{\mathcal{{W}}_{}}}",
    "MG": "M_{{\mathcal{{G}}_{}}}",
    "MH": "M_{{\mathcal{{H}}_{}}}",
    "MB1": "M_{{\mathcal{{B}}^1_{}}}",
    "MW1": "M_{{\mathcal{{W}}^1_{}}}",
    "MG1": "M_{{\mathcal{{G}}^1_{}}}",
    "ML3": "M_{{\mathcal{{L}}^3_{}}}",
    "MU2": "M_{{\mathcal{{U}}^2_{}}}",
    "MU5": "M_{{\mathcal{{U}}^5_{}}}",
    "MQ1": "M_{{\mathcal{{Q}}^1_{}}}",
    "MQ5": "M_{{\mathcal{{Q}}^5_{}}}",
    "MX": "M_{{\mathcal{{X}}_{}}}",
    "MY1": "M_{{\mathcal{{Y}}^1_{}}}",
    "MY5": "M_{{\mathcal{{Y}}^5_{}}}"}

"""
LaTeX representation for the tensors and field defined for heavy vectors.
"""

if __name__ == "__main__":
    
    # -- Integration ---------------------------------------------------------
    
    eff_lag = integrate(heavy_vectors, L_vectors, 6)

    
    # -- Transformations -----------------------------------------------------
    #
    # Here's where the rules for the transformations to a basis of effective
    # operators should be given, together with the definition of the basis.
    # Then, the function effective.transformations.apply_rules can be used
    # to apply them to the effective lagrangian.
    
    
    # -- Output --------------------------------------------------------------
    
    eff_lag_writer = Writer(eff_lag, {})

    sys.stdout.write(str(eff_lag_writer) + "\n")

    latex_tensors = {}
    latex_tensors.update(latex_tensors_vectors)
    latex_tensors.update(latex_SM)
    latex_tensors.update(latex_SU2)
    latex_tensors.update(latex_SU3)
    latex_tensors.update(latex_Lorentz)

    eff_lag_writer.show_pdf(
        "vectors", "open", latex_tensors, {},
        list(map(chr, range(ord('a'), ord('z')))))
                          
