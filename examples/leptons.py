"""
This script defines all the heavy leptons (color-singlet fermions)
that couple linearly through renormalizable interactions to the 
Standard Model. It specifies their interaction lagrangian and 
integrates them out.
"""

from __future__ import print_function
import context

# -- Core tools --------------------------------------------------------------

from effective.core import (
    TensorBuilder, FieldBuilder, D, Op, OpSum,
    number_op, symbol_op, tensor_op, flavor_tensor_op,
    boson, fermion, kdelta)

from effective.integration import (
    integrate, VectorLikeFermion, MajoranaFermion)

from effective.output import Writer


# -- Predefined tensors and rules --------------------------------------------

from effective.extras.SM import (
    mu2phi, lambdaphi, ye, yec, yd, ydc, yu, yuc, V, Vc,
    phi, phic, lL, lLc, qL, qLc, eR, eRc, dR, dRc, uR, uRc,
    bFS, wFS, gFS, latex_SM)

from effective.extras.Lorentz import (
    sigma4, sigma4bar, epsUp, epsUpDot, epsDown, epsDownDot, latex_Lorentz)

from effective.extras.SU2 import epsSU2, sigmaSU2, latex_SU2


# -- Tensors -----------------------------------------------------------------

# Heavy - light

lambdaDelta1e = TensorBuilder("lambdaDelta1e")
lambdaDelta1ec = TensorBuilder("lambdaDelta1ec")

lambdaDelta3e = TensorBuilder("lambdaDelta3e")
lambdaDelta3ec = TensorBuilder("lambdaDelta3ec")

lambdaNRl = TensorBuilder("lambdaNRl")
lambdaNRlc = TensorBuilder("lambdaNRlc")
lambdaNLl = TensorBuilder("lambdaNLl")
lambdaNLlc = TensorBuilder("lambdaNLlc")
lambdaNmajl = TensorBuilder("lambdaNmajl")
lambdaNmajlc = TensorBuilder("lambdaNmajlc")

lambdaEl = TensorBuilder("lambdaEl")
lambdaElc = TensorBuilder("lambdaElc")

lambdaSigma0Rl = TensorBuilder("lambdaSigma0Rl")
lambdaSigma0Rlc = TensorBuilder("lambdaSigma0Rlc")
lambdaSigma0Ll = TensorBuilder("lambdaSigma0Ll")
lambdaSigma0Llc = TensorBuilder("lambdaSigma0Llc")
lambdaSigma0majl = TensorBuilder("lambdaSigma0majl")
lambdaSigma0majlc = TensorBuilder("lambdaSigma0majlc")

lambdaSigma1l = TensorBuilder("lambdaSigma1l")
lambdaSigma1lc = TensorBuilder("lambdaSigma1lc")

# Heavy - heavy

lambdaDelta1NL = TensorBuilder("lambdaDelta1NL")
lambdaDelta1NLc = TensorBuilder("lambdaDelta1NLc")
lambdaDelta1NR = TensorBuilder("lambdaDelta1NR")
lambdaDelta1NRc = TensorBuilder("lambdaDelta1NRc")
lambdaDelta1Nmaj = TensorBuilder("lambdaDelta1Nmaj")
lambdaDelta1Nmajc = TensorBuilder("lambdaDelta1Nmajc")

lambdaDelta1E = TensorBuilder("lambdaDelta1E")
lambdaDelta1Ec = TensorBuilder("lambdaDelta1Ec")

lambdaDelta3E = TensorBuilder("lambdaDelta3E")
lambdaDelta3Ec = TensorBuilder("lambdaDelta3Ec")

lambdaDelta1Sigma0L = TensorBuilder("lambdaDelta1Sigma0L")
lambdaDelta1Sigma0Lc = TensorBuilder("lambdaDelta1Sigma0Lc")
lambdaDelta1Sigma0R = TensorBuilder("lambdaDelta1Sigma0R")
lambdaDelta1Sigma0Rc = TensorBuilder("lambdaDelta1Sigma0Rc")
lambdaDelta1Sigma0maj = TensorBuilder("lambdaDelta1Sigma0maj")
lambdaDelta1Sigma0majc = TensorBuilder("lambdaDelta1Sigma0majc")

lambdaDelta1Sigma1 = TensorBuilder("lambdaDelta1Sigma1")
lambdaDelta1Sigma1c = TensorBuilder("lambdaDelta1Sigma1c")

lambdaDelta3Sigma1 = TensorBuilder("lambdaDelta3Sigma1")
lambdaDelta3Sigma1c = TensorBuilder("lambdaDelta3Sigma1c")

# -- Fields (SU(3) singlets) -------------------------------------------------

# Neutral vector-like SU(2) singlet
NL = FieldBuilder("NL", 1.5, fermion)
NR = FieldBuilder("NR", 1.5, fermion)
NLc = FieldBuilder("NLc", 1.5, fermion)
NRc = FieldBuilder("NRc", 1.5, fermion)

# Neutral Majorana SU(2) singlet
Nmaj = FieldBuilder("Nmaj", 1.5, fermion)
Nmajc = FieldBuilder("Nmajc", 1.5, fermion)

# Hypercharge -1 vector-like SU(2) singlet
EL = FieldBuilder("EL", 1.5, fermion)
ER = FieldBuilder("ER", 1.5, fermion)
ELc = FieldBuilder("ELc", 1.5, fermion)
ERc = FieldBuilder("ERc", 1.5, fermion)

# Hypercharge -1/2 vector-like SU(2) doublet
Delta1L = FieldBuilder("Delta1L", 1.5, fermion)
Delta1R = FieldBuilder("Delta1R", 1.5, fermion)
Delta1Lc = FieldBuilder("Delta1Lc", 1.5, fermion)
Delta1Rc = FieldBuilder("Delta1Rc", 1.5, fermion)

# Hypercharge -3/2 vector-like SU(2) doublet
Delta3L = FieldBuilder("Delta3L", 1.5, fermion)
Delta3R = FieldBuilder("Delta3R", 1.5, fermion)
Delta3Lc = FieldBuilder("Delta3Lc", 1.5, fermion)
Delta3Rc = FieldBuilder("Delta3Rc", 1.5, fermion)

# Hypercharge 0 vector-like SU(2) triplet
Sigma0L = FieldBuilder("Sigma0L", 1.5, fermion)
Sigma0R = FieldBuilder("Sigma0R", 1.5, fermion)
Sigma0Lc = FieldBuilder("Sigma0Lc", 1.5, fermion)
Sigma0Rc = FieldBuilder("Sigma0Rc", 1.5, fermion)

# Hypercharge 0 Majorana SU(2) triplet
Sigma0maj = FieldBuilder("Sigma0maj", 1.5, fermion)
Sigma0majc = FieldBuilder("Sigma0majc", 1.5, fermion)

# Hypercharge -1 vector-like SU(2) triplet
Sigma1L = FieldBuilder("Sigma1L", 1.5, fermion)
Sigma1R = FieldBuilder("Sigma1R", 1.5, fermion)
Sigma1Lc = FieldBuilder("Sigma1Lc", 1.5, fermion)
Sigma1Rc = FieldBuilder("Sigma1Rc", 1.5, fermion)


# -- Lagrangian --------------------------------------------------------------

L_leptons = -OpSum(
    # Delta1
    Op(lambdaDelta1e(0, 1), Delta1Lc(2, 3, 0), phi(3), eR(2, 1)),
    Op(lambdaDelta1ec(0, 1), eRc(2, 1), phic(3), Delta1L(2, 3, 0)),

    # Delta3
    Op(lambdaDelta3e(0, 1), Delta3Lc(2, 3, 0),
       epsSU2(3, 4), phic(4), eR(2, 1)),
    Op(lambdaDelta3ec(0, 1), eRc(2, 1),
       epsSU2(3, 4), phi(4), Delta3L(2, 3, 0)),

    # N
    Op(lambdaNLl(0, 1), NL(2, 0), epsSU2(3, 4), phi(4),
       epsUp(5, 2), lL(5, 3, 1)),
    Op(lambdaNLlc(0, 1), lLc(2, 3, 1), epsSU2(3, 4), phic(4),
       epsUpDot(2, 5), NLc(5, 0)),
    Op(lambdaNRl(0, 1), NRc(2, 0), epsSU2(3, 4), phi(4), lL(2, 3, 1)),
    Op(lambdaNRlc(0, 1), lLc(2, 3, 1), epsSU2(3, 4), phic(4), NR(2, 0)),


    #Nmaj
    Op(lambdaNmajl(0, 1), Nmaj(2, 0), epsSU2(3, 4), phi(4),
       epsUp(5, 2), lL(5, 3, 1)),
    Op(lambdaNmajlc(0, 1), lLc(2, 3, 1), epsSU2(3, 4), phic(4),
       epsUpDot(2, 5), Nmajc(5, 0)),

    # E
    Op(lambdaEl(0, 1), ERc(2, 0), phic(3), lL(2, 3, 1)),
    Op(lambdaElc(0, 1), lLc(2, 3, 1), phi(3), ER(2, 0)),

    # Sigma0
    number_op(0.5) * Op(lambdaSigma0Rl(0, 1), Sigma0Rc(2, 3, 0),
                        epsSU2(4, 5), phi(5), sigmaSU2(3, 4, 6),
                        lL(2, 6, 1)),
    number_op(0.5) * Op(lambdaSigma0Rlc(0, 1), lLc(2, 6, 1),
                        sigmaSU2(3, 6, 4), epsSU2(4, 5), phic(5),
                        Sigma0R(2, 3, 0)),
    number_op(0.5) * Op(lambdaSigma0Ll(0, 1), Sigma0L(2, 3, 0),
                        epsSU2(4, 5), phi(5), sigmaSU2(3, 4, 6),
                        epsUp(7, 2), lL(7, 6, 1)),
    number_op(0.5) * Op(lambdaSigma0Llc(0, 1), lLc(7, 6, 1),
                        sigmaSU2(3, 6, 4), epsSU2(4, 5), phic(5),
                        epsUpDot(7, 2), Sigma0Lc(2, 3, 0)),

    # Sigma0maj
    number_op(0.5) * Op(lambdaSigma0majl(0, 1), Sigma0maj(2, 3, 0),
                        epsSU2(4, 5), phi(5), sigmaSU2(3, 4, 6),
                        epsUp(7, 2), lL(7, 6, 1)),
    number_op(0.5) * Op(lambdaSigma0majlc(0, 1), lLc(7, 6, 1),
                        sigmaSU2(3, 6, 4), epsSU2(4, 5), phic(5),
                        epsUpDot(7, 2), Sigma0majc(2, 3, 0)),

    # Sigma1
    number_op(0.5) * Op(lambdaSigma1l(0, 1), Sigma1Rc(2, 3, 0),
                        phic(4), sigmaSU2(3, 4, 5), lL(2, 5, 1)),
    number_op(0.5) * Op(lambdaSigma1lc(0, 1), lLc(2, 5, 1),
                        sigmaSU2(3, 5, 4), phi(4), Sigma1R(2, 3, 0)),

    # Delta1 and N
    Op(lambdaDelta1NL(0, 1), Delta1Rc(2, 3, 0),
       epsSU2(3, 4), phic(4), NL(2, 1)),
    Op(lambdaDelta1NLc(0, 1), NLc(2, 1),
       epsSU2(3, 4), phi(4), Delta1R(2, 3, 0)),
    Op(lambdaDelta1NR(0, 1), Delta1Rc(2, 3, 0),
       epsSU2(3, 4), phic(4), epsDown(2, 5), NRc(5, 1)),
    Op(lambdaDelta1NRc(0, 1), NR(2, 1), epsSU2(3, 4), phi(4),
       epsDownDot(5, 2), Delta1R(5, 3, 0)),

    # Delta1 and Nmaj
    Op(lambdaDelta1Nmaj(0, 1), Delta1Rc(2, 3, 0),
       epsSU2(3, 4), phic(4), Nmaj(2, 1)),
    Op(lambdaDelta1Nmajc(0, 1), Nmajc(2, 1),
       epsSU2(3, 4), phi(4), Delta1R(2, 3, 0)),

    # Delta1 and E
    Op(lambdaDelta1E(0, 1), Delta1Rc(2, 3, 0), phi(3), EL(2, 1)),
    Op(lambdaDelta1Ec(0, 1), ELc(2, 1), phic(3), Delta1R(2, 3, 0)),

    # Delta3 and E
    Op(lambdaDelta3E(0, 1), Delta3Rc(2, 3, 0),
       epsSU2(3, 4), phic(4), EL(2, 1)),
    Op(lambdaDelta3Ec(0, 1), ELc(2, 1),
       epsSU2(3, 4), phi(4), Delta3R(2, 3, 0)),

    # Delta1 and Sigma0
    Op(lambdaDelta1Sigma0L(0, 1), Delta1Rc(2, 3, 0), sigmaSU2(4, 3, 5),
       epsSU2(5, 6), phic(6), Sigma0L(2, 4, 1)),
    Op(lambdaDelta1Sigma0Lc(0, 1), Sigma0Lc(2, 4, 1), epsSU2(5, 6), phi(6),
       sigmaSU2(4, 5, 3), Delta1R(2, 3, 0)),
    Op(lambdaDelta1Sigma0R(0, 1), Delta1Rc(2, 3, 0), sigmaSU2(4, 3, 5),
       epsSU2(5, 6), phic(6), epsDown(2, 7), Sigma0Rc(7, 4, 1)),
    Op(lambdaDelta1Sigma0Rc(0, 1), Sigma0R(2, 4, 1), epsSU2(5, 6), phi(6),
       sigmaSU2(4, 5, 3), epsDownDot(7, 2), Delta1R(7, 3, 0)),

    # Delta1 and Sigma0maj
    Op(lambdaDelta1Sigma0maj(0, 1), Delta1Rc(2, 3, 0), sigmaSU2(4, 3, 5),
       epsSU2(5, 6), phic(6), Sigma0maj(2, 4, 1)),
    Op(lambdaDelta1Sigma0Lc(0, 1), Sigma0majc(2, 4, 1), epsSU2(5, 6), phi(6),
       sigmaSU2(4, 5, 3), Delta1R(2, 3, 0)),

    # Delta1 and Sigma1
    Op(lambdaDelta1Sigma1(0, 1), Delta1Rc(2, 3, 0), sigmaSU2(4, 3, 5),
       phi(5), Sigma1L(2, 4, 1)),
    Op(lambdaDelta1Sigma1c(0, 1), Sigma1Lc(2, 4, 1), phic(5),
       sigmaSU2(4, 5, 3), Delta1R(2, 3, 0)),

    # Delta3 and Sigma1
    Op(lambdaDelta3Sigma1(0, 1), Delta3Rc(2, 3, 0), sigmaSU2(4, 3, 5),
       epsSU2(5, 6), phic(6), Sigma1L(2, 4, 1)),
    Op(lambdaDelta3Sigma1c(0, 1), Sigma1Lc(2, 4, 1), epsSU2(5, 6), phi(6),
       sigmaSU2(4, 5, 3), Delta3R(2, 3, 0)))


# -- Heavy fields ------------------------------------------------------------

heavy_N = VectorLikeFermion("N", "NL", "NR", "NLc", "NRc", 2)
heavy_Nmaj = MajoranaFermion("Nmaj", "Nmajc", 2)
heavy_E = VectorLikeFermion("E", "EL", "ER", "ELc", "ERc", 2)
heavy_Delta1 = VectorLikeFermion("Delta1", "Delta1L", "Delta1R",
                                 "Delta1Lc", "Delta1Rc", 3)
heavy_Delta3 = VectorLikeFermion("Delta3", "Delta3L", "Delta3R",
                                 "Delta3Lc", "Delta3Rc", 3)
heavy_Sigma0 = VectorLikeFermion("Sigma0", "Sigma0L", "Sigma0R",
                                 "Sigma0Lc", "Sigma0Rc", 3)
heavy_Sigma0maj = MajoranaFermion("Sigma0maj", "Sigma0majc", 3)
heavy_Sigma1 = VectorLikeFermion("Sigma1", "Sigma1L", "Sigma1R",
                                 "Sigma1Lc", "Sigma1Rc", 3)

heavy_leptons = [heavy_N, heavy_Nmaj, heavy_E, heavy_Delta1, heavy_Delta3,
                heavy_Sigma0, heavy_Sigma0maj, heavy_Sigma1]
"""
All the heavy lepton (SU(3) singlet fermion) fields that couple linearly 
through renormalizable interactions to the Standard Model.
"""


# -- LaTeX representation ----------------------------------------------------

latex_tensors_leptons = {
    
    # Heavy - light
    
    "lambdaDelta1e": r"(\lambda^{{\Delta_1}}_e)_{{{}{}}}",
    "lambdaDelta1ec":  r"(\lambda^{{\Delta_1}}_e)^*_{{{}{}}}",

    "lambdaDelta3e":  r"(\lambda^{{\Delta_3}}_e)_{{{}{}}}",
    "lambdaDelta3ec":  r"(\lambda^{{\Delta_3}}_e)_{{{}{}}}",

    "lambdaNRl":  r"(\lambda^{{N_R}}_l)_{{{}{}}}",
    "lambdaNRlc": r"(\lambda^{{N_R}}_l)^*_{{{}{}}}",
    "lambdaNLl": r"(\lambda^{{N_L}}_l)_{{{}{}}}",
    "lambdaNLlc": r"(\lambda^{{N_L}}_l)^*_{{{}{}}}",
    "lambdaNmajl": r"(\lambda^{{N^{{(maj)}}}}_l)_{{{}{}}}",
    "lambdaNmajlc": r"(\lambda^{{N^{{(maj)}}}}_l)^*_{{{}{}}}",

    "lambdaEl": r"(\lambda^{{E}}_l)_{{{}{}}}",
    "lambdaElc": r"(\lambda^{{E}}_l)^*_{{{}{}}}",

    "lambdaSigma0Rl": r"(\lambda^{{\Sigma_{{0R}}}}_l)_{{{}{}}}",
    "lambdaSigma0Rlc":  r"(\lambda^{{\Sigma_{{0R}}}}_l)^*_{{{}{}}}",
    "lambdaSigma0Ll": r"(\lambda^{{\Sigma_{{0L}}}}_l)_{{{}{}}}",
    "lambdaSigma0Llc":  r"(\lambda^{{\Sigma_{{0L}}}}_l)^*_{{{}{}}}",
    "lambdaSigma0majl":  r"(\lambda^{{\Sigma^{{(maj)}}_{{0}}}}_l)_{{{}{}}}",
    "lambdaSigma0majlc":
    r"(\lambda^{{\Sigma^{{(maj)}}_{{0}}}}_l)^*_{{{}{}}}",

    "lambdaSigma1l":  r"(\lambda^{{\Sigma_{{1}}}}_l)_{{{}{}}}",
    "lambdaSigma1lc":  r"(\lambda^{{\Sigma_{{1}}}}_l)^*_{{{}{}}}",

    # Heavy - heavy

    "lambdaDelta1NL": r"(\lambda^{{\Delta_{{1}}}}_{{N_L}})_{{{}{}}}",
    "lambdaDelta1NLc": r"(\lambda^{{\Delta_{{1}}}}_{{N_L}})^*_{{{}{}}}",
    "lambdaDelta1NR":  r"(\lambda^{{\Delta_{{1}}}}_{{N_R}})_{{{}{}}}",
    "lambdaDelta1NRc":  r"(\lambda^{{\Delta_{{1}}}}_{{N_R}})^*_{{{}{}}}",
    "lambdaDelta1Nmaj":
    r"(\lambda^{{\Delta_{{1}}}}_{{N^{{(maj)}}}})_{{{}{}}}",
    "lambdaDelta1Nmajc":
    r"(\lambda^{{\Delta_{{1}}}}_{{N^{{(maj)}}}})_{{{}{}}}",

    "lambdaDelta1E":  r"(\lambda^{{\Delta_{{1}}}}_{{E}})_{{{}{}}}",
    "lambdaDelta1Ec":  r"(\lambda^{{\Delta_{{1}}}}_{{E}})^*_{{{}{}}}",

    "lambdaDelta3E":  r"(\lambda^{{\Delta_{{3}}}}_{{E}})_{{{}{}}}",
    "lambdaDelta3Ec":  r"(\lambda^{{\Delta_{{3}}}}_{{E}})^*_{{{}{}}}",

    "lambdaDelta1Sigma0L":
    r"(\lambda_{{\Sigma_{{0L}}}}^{{Delta_1}})_{{{}{}}}",
    "lambdaDelta1Sigma0Lc":
    r"(\lambda_{{\Sigma_{{0L}}}}^{{Delta_1}})^*_{{{}{}}}",
    "lambdaDelta1Sigma0R":
    r"(\lambda_{{\Sigma_{{0R}}}}^{{Delta_1}})_{{{}{}}}",
    "lambdaDelta1Sigma0Rc":
    r"(\lambda_{{\Sigma_{{0R}}}}^{{Delta_1}})_{{{}{}}}",
    "lambdaDelta1Sigma0maj":
    r"(\lambda_{{\Sigma^{{(maj)}}}}^{{Delta_1}})_{{{}{}}}",
    "lambdaDelta1Sigma0majc":
    r"(\lambda_{{\Sigma^{{(maj)}}}}^{{Delta_1}})^*_{{{}{}}}",

    "lambdaDelta1Sigma1":
    r"(\lambda_{{\Sigma_{{1}}}}^{{Delta_1}})_{{{}{}}}",
    "lambdaDelta1Sigma1c":
    r"(\lambda_{{\Sigma_{{1}}}}^{{Delta_1}})^*_{{{}{}}}",

    "lambdaDelta3Sigma1":
    r"(\lambda_{{\Sigma_{{1}}}}^{{Delta_3}})_{{{}{}}}",
    "lambdaDelta3Sigma1c":
    r"(\lambda_{{\Sigma_{{1}}}}^{{Delta_3}})^*_{{{}{}}}",

    # Masses
    "MN": r"M_{{N_{}}}",
    "MNmaj": r"M_{{N^{{(maj)}}_{}}}",
    "ME": r"M_{{E_{}}}",
    "MDelta1": r"M_{{\Delta_{{1{}}}}}",
    "MDelta3": r"M_{{\Delta_{{3{}}}}}",
    "MSigma0": r"M_{{\Sigma_{{0{}}}}}",
    "MSigma0maj": r"M_{{\Sigma^{{(maj)}}_{{0{}}}}}",
    "MSigma1": r"M_{{\Sigma_{{1{}}}}}"}


if __name__ == "__main__":
    
    # -- Integration ---------------------------------------------------------
    
    eff_lag = integrate(heavy_leptons, L_leptons, 6)

    
    # -- Transformations -----------------------------------------------------
    #
    # Here's where the rules for the transformations to a basis of effective
    # operators should be given, together with the definition of the basis.
    # Then, the function effective.transformations.apply_rules can be used
    # to apply them to the effective lagrangian.

    # -- Output --------------------------------------------------------------
    
    eff_lag_writer = Writer(eff_lag, {})
    
    print(eff_lag_writer)

    latex_tensors = {}
    latex_tensors.update(latex_tensors_leptons)
    latex_tensors.update(latex_SM)
    latex_tensors.update(latex_SU2)
    latex_tensors.update(latex_Lorentz)

    eff_lag_writer.write_latex(
        "leptons", latex_tensors, {},
        list(map(chr, range(ord('a'), ord('z')))))
                          
