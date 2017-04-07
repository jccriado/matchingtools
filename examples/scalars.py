"""
This script defines all the heavy leptons (color-singlet fermions)
that couple linearly through renormalizable interactions to the 
Standard Model. It specifies their interaction lagrangian and 
integrates them out.
"""

import context
import sys


# -- Core tools --------------------------------------------------------------

from effective.operators import (
    TensorBuilder, FieldBuilder, D, Op, OpSum,
    number_op, symbol_op, tensor_op, flavor_tensor_op,
    boson, fermion, kdelta)

from effective.integration import integrate, RealScalar, ComplexScalar

from effective.output import Writer


# -- Predefined tensors and rules --------------------------------------------

from effective.extras.SM import (
    mu2phi, lambdaphi, ye, yec, yd, ydc, yu, yuc, V, Vc,
    phi, phic, lL, lLc, qL, qLc, eR, eRc, dR, dRc, uR, uRc,
    bFS, wFS, gFS, latex_SM)

from effective.extras.Lorentz import (
    sigma4, sigma4bar, epsUp, epsUpDot, epsDown, epsDownDot, latex_Lorentz)

from effective.extras.SU3 import TSU3, epsSU3, latex_SU3

from effective.extras.SU2 import (
    epsSU2, sigmaSU2, CSU2, CSU2c, epsSU2quadruplets, fSU2, latex_SU2)


# -- Tensors -----------------------------------------------------------------

kappaS = TensorBuilder("kappaS")
lambdaS = TensorBuilder("lambdaS")
kappaS3 = TensorBuilder("kappaS3")

ylS1  = TensorBuilder("ylS1")
ylS1c = TensorBuilder("ylS1c")

yeS2 = TensorBuilder("yeS2")
yeS2c = TensorBuilder("yeS2c")

yevarphi = TensorBuilder("yevarphi")
yevarphic = TensorBuilder("yevarphic")
ydvarphi = TensorBuilder("ydvarphi")
ydvarphic = TensorBuilder("ydvarphic")
yuvarphi = TensorBuilder("yuvarphi")
yuvarphic = TensorBuilder("yuvarphic")
lambdavarphi = TensorBuilder("lambdavarphi")
lambdavarphic = TensorBuilder("lambdavarphic")

kappaXi0 = TensorBuilder("kappaXi0")
lambdaXi0 = TensorBuilder("lambdaXi0")

ylXi1 = TensorBuilder("ylXi1")
ylXi1c = TensorBuilder("ylXi1c")
kappaXi1 = TensorBuilder("kappaXi1")
kappaXi1c = TensorBuilder("kappaXi1c")
lambdaXi1 = TensorBuilder("lambdaXi1")
lambdaTildeXi1 = TensorBuilder("lambdaTildeXi1")

lambdaTheta1 = TensorBuilder("lambdaTheta1")
lambdaTheta1c = TensorBuilder("lambdaTheta1c")

lambdaTheta3 = TensorBuilder("lambdaTheta3")
lambdaTheta3c = TensorBuilder("lambdaTheta3c")

yqlomega1 = TensorBuilder("yqlomega1")
yqlomega1c = TensorBuilder("yqlomega1c")
yqqomega1 = TensorBuilder("yqqomega1")
yqqomega1c = TensorBuilder("yqqomega1c")
yeuomega1 = TensorBuilder("yeuomega1")
yeuomega1c = TensorBuilder("yeuomega1c")
yduomega1 = TensorBuilder("yduomega1")
yduomega1c = TensorBuilder("yduomega1c")

ydomega2 = TensorBuilder("ydomega2")
ydomega2c = TensorBuilder("ydomega2c")

yedomega4 = TensorBuilder("yedomega4")
yedomega4c = TensorBuilder("yedomega4c")
yuuomega4 = TensorBuilder("yuuomega4")
yuuomega4c = TensorBuilder("yuuomega4c")

yldPi1 = TensorBuilder("yldPi1")
yldPi1c = TensorBuilder("yldPi1c")

yluPi7 = TensorBuilder("yluPi7")
yluPi7c = TensorBuilder("yluPi7c")
yeqPi7 = TensorBuilder("yeqPi7")
yeqPi7c = TensorBuilder("yeqPi7c")

yqlzeta = TensorBuilder("yqlzeta")
yqlzetac = TensorBuilder("yqlzetac")
yqqzeta = TensorBuilder("yqqzeta")
yqqzetac = TensorBuilder("yqqzetac")

yudOmega1 = TensorBuilder("yudOmega1")
yudOmega1c = TensorBuilder("yudOmega1c")
yqqOmega1 = TensorBuilder("yqqOmega1")
yqqOmega1c = TensorBuilder("yqqOmega1c")

ydOmega2 = TensorBuilder("ydOmega2")
ydOmega2c = TensorBuilder("ydOmega2c")

yuOmega4 = TensorBuilder("yuOmega4")
yuOmega4c = TensorBuilder("yuOmega4c")

yqUpsilon = TensorBuilder("yqUpsilon")
yqUpsilonc = TensorBuilder("yqUpsilonc")

yquPhi = TensorBuilder("yquPhi")
yquPhic = TensorBuilder("yquPhic")
ydqPhi = TensorBuilder("ydqPhi")
ydqPhic = TensorBuilder("ydqPhic")

kappaSvarphi = TensorBuilder("kappaSvarphi")
kappaSvarphic = TensorBuilder("kappaSvarphic")
kappaSXi0 = TensorBuilder("kappaSXi0")
kappaSXi1 = TensorBuilder("kappaSXi1")
kappaXi03 = TensorBuilder("kappaXi03")
kappaXi0Xi1 = TensorBuilder("kappaXi0Xi1")
kappaXi0varphi = TensorBuilder("kappaXi0varphi")
kappaXi0varphic = TensorBuilder("kappaXi0varphic")
kappaXi1varphi = TensorBuilder("kappaXi1varphi")
kappaXi1varphic = TensorBuilder("kappaXi1varphic")
kappaXi0Theta1 = TensorBuilder("kappaXi0Theta1")
kappaXi0Theta1c = TensorBuilder("kappaXi0Theta1c")
kappaXi1Theta1 = TensorBuilder("kappaXi1Theta1")
kappaXi1Theta1c = TensorBuilder("kappaXi1Theta1c")
kappaXi1Theta3 = TensorBuilder("kappaXi1Theta3")
kappaXi1Theta3c = TensorBuilder("kappaXi1Theta3c")
lambdaSXi0 = TensorBuilder("lambdaSXi0")
lambdaSXi1 = TensorBuilder("lambdaSXi1")
lambdaSXi1c = TensorBuilder("lambdaSXi1c")
lambdaXi1Xi0 = TensorBuilder("lambdaXi1Xi0")
lambdaXi1Xi0c = TensorBuilder("lambdaXi1Xi0c")


# -- Fields ------------------------------------------------------------------

# Neutral singlet
S = FieldBuilder("S", 1, boson)

# Hypercharge 1 singlet
S1 = FieldBuilder("S1", 1, boson)
S1c = FieldBuilder("S1c", 1, boson)

# Hypercharge 2 singlet
S2 = FieldBuilder("S2", 1, boson)
S2c = FieldBuilder("S2c", 1, boson)

# Hypercharge 1/2 SU(2) doublet
varphi = FieldBuilder("varphi", 1, boson)
varphic = FieldBuilder("varphic", 1, boson)

# Neutral SU(2) triplet
Xi0 = FieldBuilder("Xi0", 1, boson)

# Hypercharge 1 SU(2) triplet
Xi1 = FieldBuilder("Xi1", 1, boson)
Xi1c = FieldBuilder("Xi1c", 1, boson)

# Hypercharge 1/2 SU(2) quadruplet
Theta1 = FieldBuilder("Theta1", 1, boson)
Theta1c = FieldBuilder("Theta1c", 1, boson)

# Hypercharge 3/2 SU(2) quadruplet
Theta3 = FieldBuilder("Theta3", 1, boson)
Theta3c = FieldBuilder("Theta3c", 1, boson)

# Hypercharge -1/3 SU(3) triplet
omega1 = FieldBuilder("omega1", 1, boson)
omega1c = FieldBuilder("omega1c", 1, boson)

# Hypercharge 2/3 SU(3) triplet
omega2 = FieldBuilder("omega2", 1, boson)
omega2c = FieldBuilder("omega2c", 1, boson)

# Hypercharge -4/3 SU(3) triplet
omega4 = FieldBuilder("omega4", 1, boson)
omega4c = FieldBuilder("omega4c", 1, boson)

# Hypercharge 1/6 SU(3) triplet, SU(2) doublet
Pi1 = FieldBuilder("Pi1", 1, boson)
Pi1c = FieldBuilder("Pi1c", 1, boson)

# Hypercharge 7/6 SU(3) triplet, SU(2) doublet
Pi7 = FieldBuilder("Pi7", 1, boson)
Pi7c = FieldBuilder("Pi7c", 1, boson)

# Hypercharge -1/3 SU(3) triplet, SU(2) triplet
zeta = FieldBuilder("zeta", 1, boson)
zetac = FieldBuilder("zetac", 1, boson)

# Hypercharge 1/3 SU(3) sextet
# (the sextet index is represented by two SU(3) antitriplet indices
# for which the field is antisymmetric)
Omega1 = FieldBuilder("Omega1", 1, boson)
Omega1c = FieldBuilder("Omega1c", 1, boson)

# Hypercharge -2/3 SU(3) sextet
# (the sextet index is represented by two SU(3) antitriplet indices
# for which the field is antisymmetric)
Omega2 = FieldBuilder("Omega2", 1, boson)
Omega2c = FieldBuilder("Omega2c", 1, boson)

# Hypercharge 4/3 SU(3) sextet
# (the sextet index is represented by two SU(3) antitriplet indices
# for which the field is antisymmetric)
Omega4 = FieldBuilder("Omega4", 1, boson)
Omega4c = FieldBuilder("Omega4c", 1, boson)

# Hypercharge 1/3 SU(3) sextet, SU(2) triplet
# (the sextet index is represented by two SU(3) antitriplet indices
# for which the field is antisymmetric)
Upsilon = FieldBuilder("Upsilon", 1, boson)
Upsilonc = FieldBuilder("Upsilonc", 1, boson)

# Hypercharge 1/2 SU(3) octet, SU(2) doublet
Phi = FieldBuilder("Phi", 1, boson)
Phic = FieldBuilder("Phic", 1, boson)


# -- Lagrangian --------------------------------------------------------------

L_scalars = -OpSum(
    # S
    Op(kappaS(0), phic(1), phi(1), S(0)),
    Op(lambdaS(0, 1), S(0), S(1), phic(2), phi(2)),
    Op(kappaS3(0, 1, 2), S(0), S(1), S(2)),
    
    # S1
    Op(ylS1(0, 1, 2), S1c(0), lLc(3, 4, 1), epsUpDot(3, 5),
              epsSU2(4, 6), lLc(5, 6, 2)),
    Op(ylS1c(0, 1, 2), S1(0), lL(3, 4, 1), epsUp(3, 5),
              epsSU2(4, 6), lL(5, 6, 2)),

    # S2
    Op(yeS2(0, 1, 2), S2c(0), epsDown(3, 4), eRc(3, 1), eRc(4, 2)),
    Op(yeS2c(0, 1, 2), S2(0), epsDownDot(3, 4), eR(3, 1), eR(4, 2)),

    # varphi
    Op(yevarphi(0, 1, 2), varphic(3, 0), eRc(4, 1), lL(4, 3, 2)),
    Op(yevarphic(0, 1, 2), varphi(3, 0), lLc(4, 3, 2), eR(4, 1)),
    Op(ydvarphi(0, 1, 2), varphic(3, 0), dRc(4, 5, 1), qL(4, 5, 3, 2)),
    Op(ydvarphic(0, 1, 2), varphi(3, 0), qLc(4, 5, 3, 2), dR(4, 5, 1)),
    Op(yuvarphi(0, 1, 2), varphic(3, 0), epsSU2(3, 5),
       qLc(4, 6, 5, 1), uR(4, 6, 2)),
    Op(yuvarphic(0, 1, 2), varphi(3, 0), epsSU2(3, 5),
       uRc(4, 6, 2), qL(4, 6, 5, 1)),
    Op(lambdavarphi(0), varphic(1, 0), phi(1), phic(2), phi(2)),
    Op(lambdavarphic(0), phic(1), varphi(1, 0), phic(2), phi(2)),

    # Xi0
    Op(kappaXi0(0), Xi0(1, 0), phic(2), sigmaSU2(1, 2, 3), phi(3)),
    Op(lambdaXi0(0, 1), Xi0(2, 0), Xi0(2, 1), phic(3), phi(3)),
    Op(kappaXi03(0, 1, 2), fSU2(3, 4, 5), Xi0(3, 0), Xi0(4, 1), Xi0(5, 2)),

    # Xi1
    Op(ylXi1(0, 1, 2), Xi1c(3, 0), lLc(4, 5, 1), lLc(6, 7, 2),
       epsUpDot(4, 6), sigmaSU2(3, 5, 8), epsSU2(8, 7)),
    Op(ylXi1c(0, 1, 2), Xi1(3, 0), lL(4, 5, 1), lL(6, 7, 2),
       epsUp(4, 6), sigmaSU2(3, 8, 5), epsSU2(8, 7)),
    Op(kappaXi1(0), Xi1c(1, 0), epsSU2(2, 3), phi(3),
       sigmaSU2(1, 2, 4), phi(4)),
    Op(kappaXi1c(0), Xi1(1, 0), phic(2), sigmaSU2(1, 2, 3),
       epsSU2(3, 4), phic(4)),
    Op(lambdaXi1(0, 1), Xi1c(2, 0), Xi1(2, 1), phic(3), phi(3)),
    Op(lambdaTildeXi1(0, 1), fSU2(2, 3, 4), Xi1c(2, 0), Xi1(3, 1),
              phic(5), sigmaSU2(4, 5, 6), phi(6)),

    # Theta1
    Op(lambdaTheta1(0), phic(1), sigmaSU2(2, 1, 3), phi(3),
       CSU2(4, 2, 5), epsSU2(5, 6), phic(6),
       epsSU2quadruplets(4, 7), Theta1(7, 0)),
    Op(lambdaTheta1(0), phic(1), sigmaSU2(2, 1, 3), phi(3),
       CSU2c(4, 2, 5), epsSU2(5, 6), phi(6),
       epsSU2quadruplets(4, 7), Theta1c(7, 0)),

    # Theta3
    Op(lambdaTheta3(0), phic(1), sigmaSU2(2, 1, 3), epsSU2(3, 4), phic(4),
       CSU2(5, 2, 6), epsSU2(6, 7), phic(7), epsSU2quadruplets(5, 8),
       Theta3(8, 0)),
    Op(lambdaTheta3c(0), epsSU2(1, 3), phi(3), sigmaSU2(2, 1, 4), phi(4),
       CSU2c(5, 2, 6), epsSU2(6, 7), phi(7), epsSU2quadruplets(5, 8),
       Theta3c(8, 0)),

    # omega1
    Op(yqlomega1(0, 1, 2), omega1c(3, 0), qL(4, 3, 5, 1),
       epsSU2(5, 6), epsUp(4, 7), lL(7, 6, 2)),
    Op(yqlomega1c(0, 1, 2), omega1(3, 0), lLc(7, 6, 2),
       epsSU2(5, 6), epsUpDot(7, 4), qLc(4, 3, 5, 1)),
    Op(yqqomega1(0, 1, 2), omega1c(3, 0), epsSU3(3, 4, 5),
       qLc(6, 4, 7, 1), epsSU2(7, 8), epsUpDot(6, 9), qLc(9, 5, 8, 2)),
    Op(yqqomega1c(0, 1, 2), omega1(3, 0), epsSU3(3, 4, 5),
       qL(9, 5, 8, 2), epsSU2(7, 8), epsUp(9, 6), qL(6, 4, 7, 1)),
    Op(yeuomega1(0, 1, 2), omega1c(3, 0), eR(4, 1),
       epsDownDot(4, 5), uR(5, 3, 2)),
    Op(yeuomega1c(0, 1, 2), omega1(3, 0), uRc(5, 3, 2),
       epsDown(5, 4), eRc(4, 1)),
    Op(yduomega1(0, 1, 2), omega1c(3, 0), epsSU3(3, 4, 5),
       dRc(6, 4, 1), epsDown(6, 7), uRc(7, 5, 2)),
    Op(yduomega1c(0, 1, 2), omega1(3, 0), epsSU3(3, 4, 5),
       uRc(7, 5, 2), epsDownDot(7, 6), dRc(6, 4, 1)),

    # omega2
    Op(ydomega2(0, 1, 2), omega2c(3, 0), epsSU3(3, 4, 5),
       dRc(6, 4, 1), epsDown(6, 7), dRc(7, 5, 2)),
    Op(ydomega2c(0, 1, 2), omega2(3, 0), epsSU3(3, 4, 5),
       dRc(7, 5, 2), epsDownDot(7, 6), dRc(6, 4, 1)),

    # omega4
    Op(yedomega4(0, 1, 2), omega4c(3, 0), eR(4, 1),
       epsDownDot(4, 5), dR(5, 3, 2)),
    Op(yedomega4c(0, 1, 2), omega4(3, 0), dRc(5, 3, 2),
       epsDown(5, 4), eR(4, 1)),
    Op(yuuomega4(0, 1, 2), omega4c(3, 0), epsSU3(3, 4, 5),
       uRc(6, 4, 1), epsDown(6, 7), uRc(7, 5, 2)),
    Op(yuuomega4c(0, 1, 2), omega4(3, 0), epsSU3(3, 4, 5),
       uRc(7, 5, 2), epsDownDot(7, 6), uRc(7, 4, 1)),

    # Pi1
    Op(yldPi1(0, 1, 2), Pi1c(3, 4, 0), epsSU2(4, 5),
       lLc(6, 5, 1), dR(6, 3, 2)),
    Op(yldPi1c(0, 1, 2), Pi1(3, 4, 0), epsSU2(4, 5),
       dRc(6, 3, 2), lL(6, 5, 1)),

    # Pi7
    Op(yluPi7(0, 1, 2), Pi7c(3, 4, 0), epsSU2(4, 5),
       lLc(6, 5, 1), uR(6, 3, 2)),
    Op(yluPi7c(0, 1, 2), Pi7(3, 4, 0), epsSU2(4, 5),
       uRc(6, 3, 2), lL(6, 5, 1)),
    Op(yeqPi7(0, 1, 2), Pi7c(3, 4, 0), eRc(5, 1), qL(5, 3, 4, 2)),
    Op(yeqPi7c(0, 1, 2), Pi7(3, 4, 0), qLc(5, 3, 4, 2), eR(5, 1)),

    # zeta
    Op(yqlzeta(0, 1, 2), zetac(3, 4, 0), qL(5, 3, 6, 1), epsSU2(6, 7),
       sigmaSU2(4, 7, 8), epsUp(5, 9), lL(9, 8, 2)),
    Op(yqlzetac(0, 1, 2), zeta(3, 4, 0), lLc(9, 8, 2), epsUpDot(9, 5),
       sigmaSU2(4, 8, 7), epsSU2(6, 7), qLc(5, 3, 6, 1)),
    Op(yqqzeta(0, 1, 2), zetac(3, 4, 0), epsSU3(3, 5, 6),
       qLc(7, 5, 8, 1), sigmaSU2(4, 8, 9), epsSU2(9, 10),
       epsUpDot(7, 11), qLc(11, 6, 10, 2)),
    Op(yqqzetac(0, 1, 2), zeta(3, 4, 0), epsSU3(3, 5, 6),
       qL(11, 6, 10, 2), epsUp(11, 7), epsSU2(9, 10),
       sigmaSU2(4, 9, 8), qL(7, 5, 8, 1)),

    # Omega1
    Op(yudOmega1(0, 1, 2), Omega1c(3, 4, 0), uR(5, 3, 1),
       epsDownDot(5, 6), dR(6, 4, 2)),
    Op(yudOmega1c(0, 1, 2), Omega1(3, 4, 0), dRc(6, 4, 2),
       epsDown(6, 5), uR(5, 3, 1)),
    Op(yqqOmega1(0, 1, 2), Omega1c(3, 4, 0), qL(5, 3, 6, 1),
       epsSU2(6, 7), epsUp(5, 8), qL(8, 4, 7, 2)),
    Op(yqqOmega1c(0, 1, 2), Omega1(3, 4, 0), qLc(8, 4, 7, 2),
       epsSU2(6, 7), epsUpDot(8, 5), qLc(5, 3, 6, 1)),

    # Omega2
    Op(ydOmega2(0, 1, 2), Omega2c(3, 4, 0), dR(5, 3, 1),
       epsDownDot(5, 6), dR(6, 4, 2)),
    Op(ydOmega2c(0, 1, 2), Omega2(3, 4, 0), dRc(6, 4, 2),
       epsDown(6, 5), dR(5, 3, 1)),

    # Omega4
    Op(yuOmega4(0, 1, 2), Omega4c(3, 4, 0), uR(5, 3, 1),
       epsDownDot(5, 6), uR(6, 4, 2)),
    Op(yuOmega4c(0, 1, 2), Omega4(3, 4, 0), uRc(6, 4, 2),
       epsDown(6, 5), uR(5, 3, 1)),

    # Upsilon
    Op(yqUpsilon(0, 1, 2), Upsilonc(3, 4, 5, 0),
       qL(6, 3, 7, 1), epsSU2(7, 8), sigmaSU2(5, 8, 9),
       epsUp(6, 10), qL(10, 4, 9, 2)),
    Op(yqUpsilonc(0, 1, 2), Upsilon(3, 4, 5, 0),
       qLc(10, 4, 9, 2), sigmaSU2(5, 9, 8), epsSU2(7, 8),
       epsUpDot(10, 6), qLc(6, 3, 7, 1)),

    # Phi
    Op(yquPhi(0, 1, 2), Phic(3, 4, 0), qLc(5, 6, 4, 1),
       TSU3(3, 6, 7), uR(5, 7, 2)),
    Op(yquPhic(0, 1, 2), Phi(3, 4, 0), uRc(5, 7, 2),
       TSU3(3, 7, 6), qL(5, 6, 4, 1)),

    # S and varphi
    Op(kappaSvarphi(0, 1), S(0), varphic(2, 1), phi(2)),
    Op(kappaSvarphic(0, 1), S(0), phic(2), varphi(2, 1)),

    # S and Xi0
    Op(kappaSXi0(0, 1, 2), S(0), Xi0(3, 1), Xi0(3, 2)),
    Op(lambdaSXi0(0, 1), S(0), Xi0(2, 1), phic(3), sigmaSU2(2, 3, 4), phi(4)),
    
    # S and Xi1
    Op(kappaSXi1(0, 1, 2), S(0), Xi1c(3, 1), Xi1(3, 2)),
    Op(lambdaSXi1(0, 1), S(0), Xi1c(2, 1),
       epsSU2(3, 4), phi(4), sigmaSU2(2, 4, 5), phi(5)),
    Op(lambdaSXi1c(0, 1), S(0), Xi1(2, 1),
       phic(3), sigmaSU2(2, 3, 4), epsSU2(4, 5), phic(5)),

    # Xi0 and Xi1
    Op(kappaXi0Xi1(0, 1, 2), fSU2(3, 4, 5), Xi0(3, 0), Xi1c(4, 1), Xi1(5, 2)),
    Op(lambdaXi1Xi0(0, 1), fSU2(2, 3, 4), Xi1c(2, 0), Xi0(3, 1),
       epsSU2(5, 6), phi(6), sigmaSU2(4, 5, 7), phi(7)),
    Op(lambdaXi1Xi0c(0, 1), fSU2(2, 3, 4), Xi1(2, 0), Xi0(3, 1),
       phic(5), sigmaSU2(4, 5, 6), epsSU2(6, 7), phic(7)),

    # Xi0 and varphi
    Op(kappaXi0varphi(0, 1), Xi0(2, 0), varphic(3, 1),
       sigmaSU2(2, 3, 4), phi(4)),
    Op(kappaXi0varphic(0, 1), Xi0(2, 0), phic(4),
       sigmaSU2(2, 4, 3), varphi(3, 1)),

    # Xi1 and varphi
    Op(kappaXi1varphi(0, 1), Xi1c(2, 0), epsSU2(3, 4), varphi(4, 1),
       sigmaSU2(2, 3, 5), phi(5)),
    Op(kappaXi1varphic(0, 1), Xi1(2, 0), phic(5),
       sigmaSU2(2, 5, 3), epsSU2(3, 4), varphic(4, 1)),

    # Xi0 and Theta1
    Op(kappaXi0Theta1(0, 1), Xi0(3, 0), CSU2(4, 3, 5),
       epsSU2(5, 6), phic(6), epsSU2quadruplets(4, 7), Theta1(7, 1)),
    Op(kappaXi0Theta1c(0, 1), Xi0(3, 0), CSU2c(4, 3, 5),
       epsSU2(5, 6), phi(6), epsSU2quadruplets(4, 7), Theta1c(7, 1)),

    # Xi1 and Theta1
    Op(kappaXi1Theta1(0, 1), Xi1c(3, 0), CSU2(4, 3, 5),
       phi(5), epsSU2quadruplets(4, 7), Theta1(7, 1)),
    Op(kappaXi1Theta1c(0, 1), Xi1(3, 0), CSU2c(4, 3, 5),
       phic(5), epsSU2quadruplets(4, 7), Theta1c(7, 1)),

    # Xi1 and Theta3
    Op(kappaXi1Theta3(0, 1), Xi1c(3, 0), CSU2(4, 3, 5),
       epsSU2(5, 6), phi(6), epsSU2quadruplets(4, 7), Theta3(7, 1)),
    Op(kappaXi1Theta3c(0, 1), Xi1(3, 0), CSU2c(4, 3, 5),
       epsSU2(5, 6), phic(6), epsSU2quadruplets(4, 7), Theta3c(7, 1)))


# -- Heavy fields ------------------------------------------------------------

heavy_S = RealScalar("S", 1)
heavy_S1 = ComplexScalar("S1", "S1c", 1)
heavy_S2 = ComplexScalar("S2", "S2c", 1)
heavy_varphi = ComplexScalar("varphi", "varphic", 2)
heavy_Xi0 = RealScalar("Xi0", 2)
heavy_Xi1 = ComplexScalar("Xi1", "Xi1c", 2)
heavy_Theta1 = ComplexScalar("Theta1", "Theta1c", 2)
heavy_Theta3 = ComplexScalar("Theta3", "Theta3c", 2)
heavy_omega1 = ComplexScalar("omega1", "omega1c", 2)
heavy_omega2 = ComplexScalar("omega2", "omega2c", 2)
heavy_omega4 = ComplexScalar("omega4", "omega4c", 2)
heavy_Pi1 = ComplexScalar("Pi1", "Pi1c", 3)
heavy_Pi7 = ComplexScalar("Pi7", "Pi7c", 3)
heavy_zeta = ComplexScalar("zeta", "zetac", 3)
heavy_Omega1 = ComplexScalar("Omega1", "Omega1c", 3)
heavy_Omega2 = ComplexScalar("Omega2", "Omega2c", 3)
heavy_Omega4 = ComplexScalar("Omega4", "Omega4c", 3)
heavy_Upsilon = ComplexScalar("Upsilon", "Upsilonc", 4)
heavy_Phi = ComplexScalar("Phi", "Phic", 3)

heavy_scalars = [
    heavy_S, heavy_S1, heavy_S2, heavy_varphi, heavy_Xi0, heavy_Xi1,
    heavy_Theta1, heavy_Theta3, heavy_omega1, heavy_omega2, heavy_omega4,
    heavy_Pi1, heavy_Pi7, heavy_zeta, heavy_Omega1, heavy_Omega2, heavy_Omega4,
    heavy_Upsilon, heavy_Phi]
"""
All the heavy scalar fields that couple linearly through renormalizable
interactions to the Standard Model.
"""

# -- LaTeX representation ----------------------------------------------------

latex_tensors_scalars = {
    "kappaS": r"\kappa_{{\mathcal{{S}}_{}}}",
    "lambdaS": r"\lambda^{{{}{}}}_{{\mathcal{{S}}}}",
    "kappaS3": r"\kappa^{{{}{}{}}}_{{\mathcal{{S}}^3}}",
    
    "kappaXi0": r"\kappa_{{\Xi_{{0{}}}}}",
    "lambdaXi0": r"\lambda^{{{}{}}}_{{\Xi_0}}",
    "kappaXi03": r"\kappa^{{{}{}{}}}_{{\Xi^3_0}}",
    
    "ylS1": r"\left(y^l_{{\mathcal{{S}}_{{1{}}}}}\right)_{{{}{}}}",
    "ylS1c": r"\left(y^l_{{\mathcal{{S}}_{{1{}}}}}\right)^*_{{{}{}}}",
    
    "yeS2": r"\left(y^e_{{\mathcal{{S}}_{{2{}}}}}\right)_{{{}{}}}",
    "yeS2c": r"\left(y^e_{{\mathcal{{S}}_{{2{}}}}}\right)^*_{{{}{}}}",
    
    "yevarphi": r"\left(y^e_{{\varphi_{{{}}}}}\right)_{{{}{}}}",
    "yevarphic": r"\left(y^e_{{\varphi_{{{}}}}}\right)^*_{{{}{}}}",
    "ydvarphi": r"\left(y^d_{{\varphi_{{{}}}}}\right)_{{{}{}}}",
    "ydvarphic": r"\left(y^d_{{\varphi_{{{}}}}}\right)^*_{{{}{}}}",
    "yuvarphi": r"\left(y^u_{{\varphi_{{{}}}}}\right)_{{{}{}}}",
    "yuvarphic": r"\left(y^u_{{\varphi_{{{}}}}}\right)^*_{{{}{}}}",
    "lambdavarphi": r"\lambda_{{\varphi_{}}}",
    "lambdavarphic": r"\lambda^*_{{\varphi_{}}}",
    
    "ylXi1": r"\left(y^l_{{\Xi_{{1{}}}}}\right)_{{{}{}}}",
    "ylXi1c": r"\left(y^l_{{\Xi_{{1{}}}}}\right)^*_{{{}{}}}",
    "kappaXi1": r"\kappa_{{\Xi_{{1{}}}}}",
    "kappaXi1c": r"\kappa^*_{{\Xi_{{1{}}}}}",
    "lambdaXi1": r"\lambda^{{{}{}}}_{{\Xi_1}}",
    "lambdaXi1c": r"\left(\lambda^{{{}{}}}_{{\Xi_1}}\right)^*",
    "lambdaTildeXi1": r"\tilde{{\lambda}}^{{{}{}}}_{{\Xi_1}}",
    "lambdaTildeXi1c": r"\left(\tilde{{\lambda}}^{{{}{}}}_{{\Xi_1}}\right)^*",
    
    "lambdaTheta1": r"\lambda^{}_{{\Theta_1}}",
    "lambdaTheta1c": r"\lambda^{{*{}}}_{{\Theta_1}}",
    
    "lambdaTheta3": r"\lambda^{}_{{\Theta_3}}",
    "lambdaTheta3c": r"\lambda^{{*{}}}_{{\Theta_3}}",
    
    "yqlomega1": r"(y^{{ql}}_{{\omega_{{1{}}}}})_{{{}{}}}",
    "yqlomega1c": r"(y^{{ql}}_{{\omega_{{1{}}}}})^*_{{{}{}}}",
    "yqqomega1": r"(y^{{qq}}_{{\omega_{{1{}}}}})_{{{}{}}}",
    "yqqomega1c": r"(y^{{qq}}_{{\omega_{{1{}}}}})^*_{{{}{}}}",
    "yeuomega1": r"(y^{{eu}}_{{\omega_{{1{}}}}})_{{{}{}}}",
    "yeuomega1c": r"(y^{{eu}}_{{\omega_{{1{}}}}})^*_{{{}{}}}",
    "yduomega1": r"(y^{{du}}_{{\omega_{{1{}}}}})_{{{}{}}}",
    "yduomega1c": r"(y^{{du}}_{{\omega_{{1{}}}}})^*_{{{}{}}}",
    
    "ydomega2": r"(y^{{d}}_{{\omega_{{2{}}}}})_{{{}{}}}",
    "ydomega2c": r"(y^{{d}}_{{\omega_{{2{}}}}})^*_{{{}{}}}",
    
    "yedomega4": r"(y^{{ed}}_{{\omega_{{4{}}}}})_{{{}{}}}",
    "yedomega4c": r"(y^{{ed}}_{{\omega_{{4{}}}}})^*_{{{}{}}}",
    
    "yuuomega4": r"(y^{{uu}}_{{\omega_{{4{}}}}})_{{{}{}}}",
    "yuuomega4c": r"(y^{{uu}}_{{\omega_{{4{}}}}})^*_{{{}{}}}",
    
    "yldPi1": r"(y^{{ld}}_{{\Pi_{{1{}}}}})_{{{}{}}}",
    "yldPi1c": r"(y^{{ld}}_{{\Pi_{{1{}}}}})^*_{{{}{}}}",
    
    "yluPi7": r"(y^{{lu}}_{{\Pi_{{7{}}}}})_{{{}{}}}",
    "yluPi7c": r"(y^{{lu}}_{{\Pi_{{7{}}}}})^*_{{{}{}}}",
    "yeqPi7": r"(y^{{eq}}_{{\Pi_{{7{}}}}})_{{{}{}}}",
    "yeqPi7c": r"(y^{{eq}}_{{\Pi_{{7{}}}}})^*_{{{}{}}}",
    
    "yqlzeta": r"(y^{{ql}}_{{\zeta}})_{{{}{}}}",
    "yqlzetac": r"(y^{{ql}}_{{\zeta}})^*_{{{}{}}}",
    "yqqzeta": r"(y^{{qq}}_{{\zeta}})_{{{}{}}}",
    "yqqzetac": r"(y^{{qq}}_{{\zeta}})^*_{{{}{}}}",
    
    "yudOmega1": r"(y^{{ud}}_{{\Omega_{{1{}}}}})_{{{}{}}}",
    "yudOmega1c": r"(y^{{ud}}_{{\Omega_{{1{}}}}})^*_{{{}{}}}",
    "yqqOmega1": r"(y^{{qq}}_{{\Omega_{{1{}}}}})_{{{}{}}}",
    "yqqOmega1c": r"(y^{{qq}}_{{\Omega_{{1{}}}}})^*_{{{}{}}}",

    "ydOmega2": r"(y^{{d}}_{{\Omega_{{2{}}}}})_{{{}{}}}",
    "ydOmega2c": r"(y^{{d}}_{{\Omega_{{2{}}}}})^*_{{{}{}}}",
    
    "yuOmega4": r"(y^{{u}}_{{\Omega_{{4{}}}}})_{{{}{}}}",
    "yuOmega4c": r"(y^{{u}}_{{\Omega_{{4{}}}}})^*_{{{}{}}}",
    
    "yqUpsilon": r"(y^{{q}}_{{\Upsilon}})_{{{}{}}}",
    "yqUpsilonc": r"(y^{{q}}_{{\Upsilon}})^*_{{{}{}}}",
    
    "yquPhi": r"(y^{{qu}}_{{\Phi}})_{{{}{}}}",
    "yquPhic": r"(y^{{qu}}_{{\Phi}})^*_{{{}{}}}",
    "ydqPhi": r"(y^{{dq}}_{{\Phi}})_{{{}{}}}",
    "ydqPhic": r"(y^{{dq}}_{{\Phi}})^*_{{{}{}}}",
    
    "kappaSvarphi": r"\kappa^{{{}{}}}_{{\mathcal{{S}}\varphi}}",
    "kappaSvarphic":
    r"\left(\kappa^{{{}{}}}_{{\mathcal{{S}}\varphi}}\right)^*",
    "kappaSXi0": r"\kappa^{{{}{}{}}}_{{\mathcal{{S}}\Xi_0}}",
    "kappaSXi1": r"\kappa^{{{}{}{}}}_{{\mathcal{{S}}\Xi_1}}",
    "kappaXi0Xi1": r"\kappa^{{{}{}{}}}_{{\Xi_0\Xi_1}}",
    "kappaXi0varphi": r"\kappa^{{{}{}}}_{{\Xi_0\varphi}}",
    "kappaXi0varphic": r"\left(\kappa^{{{}{}}}_{{\Xi_0\varphi}}\right)^*",
    "kappaXi1varphi": r"\kappa^{{{}{}}}_{{\Xi_1\varphi}}",
    "kappaXi1varphic": r"\left(\kappa^{{{}{}}}_{{\Xi_1\varphi}}\right)^*",
    "kappaXi0Theta1": r"\kappa^{{{}{}}}_{{\Xi_0\Theta_1}}",
    "kappaXi0Theta1c": r"(\kappa^{{{}{}}}_{{\Xi_0\Theta_1}})^*",
    "kappaXi1Theta1": r"\kappa^{{{}{}}}_{{\Xi_1\Theta_1}}",
    "kappaXi1Theta1c": r"(\kappa^{{{}{}}}_{{\Xi_1\Theta_1}})^*",
    "kappaXi1Theta3": r"\kappa^{{{}{}}}_{{\Xi_1\Theta_3}}",
    "kappaXi1Theta3c": r"(\kappa^{{{}{}}}_{{\Xi_1\Theta_3}})^*",
    "lambdaSXi0": r"\lambda^{{{}{}}}_{{\mathcal{{S}}\Xi_0}}",
    "lambdaSXi1": r"\lambda^{{{}{}}}_{{\mathcal{{S}}\Xi_1}}",
    "lambdaSXi1c": r"\left(\lambda^{{{}{}}}_{{\mathcal{{S}}\Xi_1}}\right)^*",
    "lambdaXi1Xi0": r"\lambda^{{{}{}}}_{{\Xi_1\Xi_0}}",
    "lambdaXi1Xi0c": r"\left(\lambda^{{{}{}}}_{{\Xi_1\Xi_0}}\right)^*",

    "MS": r"M_{{\mathcal{{S}}_{}}}",
    "MS1": r"M_{{\mathcal{{S}}_{{1{}}}}}",
    "MS2": r"M_{{\mathcal{{S}}_{{2{}}}}}",
    "MXi0": r"M_{{\Xi_{{0{}}}}}",
    "MXi1": r"M_{{\Xi_{{1{}}}}}",
    "Mvarphi": r"M_{{\varphi_{}}}",
    "MTheta1": r"M_{{\Theta_{{1{}}}}}",
    "MTheta3": r"M_{{\Theta_{{3{}}}}}",
    "Momega1": r"M_{{\omega_{{1{}}}}}",
    "Momega2": r"M_{{\omega_{{2{}}}}}",
    "Momega4": r"M_{{\omega_{{4{}}}}}",
    "MPi1": r"M_{{\Pi_{{1{}}}}}",
    "MPi7": r"M_{{\Pi_{{7{}}}}}",
    "Mzeta": r"M_{{\zeta_{{{}}}}}",
    "MOmega1": r"M_{{\Omega_{{1{}}}}}",
    "MOmega2": r"M_{{\Omega_{{2{}}}}}",
    "MOmega4": r"M_{{\Omega_{{4{}}}}}",
    "MUpsilon": r"M_{{\Upsilon_{{{}}}}}",
    "MPhi": r"M_{{\Phi_{{{}}}}}",

    "sqrt(2)": r"\sqrt{{2}}"}
"""
LaTeX representation for the tensors and field defined for heavy scalars.
"""

if __name__ == "__main__":
    
    # -- Integration ---------------------------------------------------------
    
    eff_lag = integrate(heavy_scalars, L_scalars, 6)

    
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
    latex_tensors.update(latex_tensors_scalars)
    latex_tensors.update(latex_SM)
    latex_tensors.update(latex_SU2)
    latex_tensors.update(latex_SU3)
    latex_tensors.update(latex_Lorentz)

    eff_lag_writer.show_pdf(
        "scalars", "open", latex_tensors, {},
        list(map(chr, range(ord('a'), ord('z')))))
                          
