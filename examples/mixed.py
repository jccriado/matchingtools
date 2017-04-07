"""
This script specifies the interacions between all the fields
that couple linearly to the Standard Model through renormalizable
interactions. It integrates them out and removes all contributions
that are not mixed effects of new fields. The Lagrangian is then
expressed in terms of the basis defined in 
`effective.extras.SM_dim_6_basis`.
"""

import sys

import context


# -- Core tools --------------------------------------------------------------

from effective.operators import (
    Op, OpSum, TensorBuilder, FieldBuilder, D,
    number_op, symbol_op, tensor_op, flavor_tensor_op,
    boson, fermion, kdelta)

from effective.transformations import (
    apply_rules, group_op_sum)

from effective.integration import (
    integrate, RealScalar, ComplexScalar, RealVector,
    ComplexVector, VectorLikeFermion, MajoranaFermion)

from effective.output import Writer


# -- Predefined tensors and rules --------------------------------------------

from effective.extras.SM import (
    mu2phi, lambdaphi, ye, yec, yd, ydc, yu, yuc, V, Vc,
    phi, phic, lL, lLc, qL, qLc, eR, eRc, dR, dRc, uR, uRc,
    bFS, wFS, gFS, eoms_SM, latex_SM)

from effective.extras.Lorentz import (
    sigma4, sigma4bar, epsUp, epsUpDot, epsDown, epsDownDot,
    latex_Lorentz, rules_Lorentz)

from effective.extras.SU2 import (
    epsSU2, sigmaSU2, rules_SU2, latex_SU2)

from effective.extras.SU3 import latex_SU3

from effective.extras.SM_dim_6_basis import (
    O1phil, O1philc, O3phil, O3philc, O1phiq, O1phiqc, O3phiq, O3phiqc,
    Ole, O1qu, O8qu, O1qd, O8qd,
    rules_basis_definitions, rules_four_fermions, latex_basis_coefs)


# -- Import all the heavy particles and their lagrangians --------------------

from scalars import (
    S, Xi0, Xi1, Xi1c, varphi, varphic,
    heavy_scalars, latex_tensors_scalars, L_scalars)

from leptons import (
    Nmaj, Nmajc, NL, NR, NLc, NRc, EL, ER, ELc, ERc,
    Delta1L, Delta1R, Delta1Lc, Delta1Rc,
    Delta3L, Delta3R, Delta3Lc, Delta3Rc,
    Sigma0maj, Sigma0majc, Sigma0L, Sigma0R, Sigma0Lc, Sigma0Rc,
    Sigma1L, Sigma1R, Sigma1Lc, Sigma1Rc,
    heavy_leptons, latex_tensors_leptons, L_leptons)

from quarks import (
    UL, UR, ULc, URc, DL, DR, DLc, DRc,
    XUL, XUR, XULc, XURc, UDL, UDR, UDLc, UDRc, DYL, DYR, DYLc, DYRc,
    XUDL, XUDR, XUDLc, XUDRc, UDYL, UDYR, UDYLc, UDYRc,
    heavy_quarks, latex_tensors_quarks, L_quarks)

from vectors import (
    B, W, W1, W1c, heavy_vectors, latex_tensors_vectors, L_vectors)

from L1_plus_vectors import (
    L1, L1c, heavy_L1, latex_tensors_L1, L_L1_plus_vectors)


# -- Tensors -----------------------------------------------------------------

# VDS
deltaB = TensorBuilder("deltaB")
deltaW = TensorBuilder("deltaW")
deltaL1 = TensorBuilder("deltaL1")
deltaL1c = TensorBuilder("deltaL1c")
deltaW1 = TensorBuilder("deltaW1")
deltaW1c = TensorBuilder("deltaW1c")

# VVS
epsilonS = TensorBuilder("epsilonS")
epsilonXi0 = TensorBuilder("epsilonXi0")
epsilonXi1 = TensorBuilder("epsilonXi1")
epsilonXi1c = TensorBuilder("epsilonXi1c")

# VS(SM)
g1Xi1L1 = TensorBuilder("g1Xi1L1")
g1Xi1L1c = TensorBuilder("g1Xi1L1c")
g1Xi0L1 = TensorBuilder("g1Xi0L1")
g1Xi0L1c = TensorBuilder("g1Xi0L1c")
g1SL1 = TensorBuilder("g1SL1")
g1SL1c = TensorBuilder("g1SL1c")
g2Xi1L1 = TensorBuilder("g2Xi1L1")
g2Xi1L1c = TensorBuilder("g2Xi1L1c")
g2Xi0L1 = TensorBuilder("g2Xi0L1")
g2Xi0L1c = TensorBuilder("g2Xi0L1c")
g2SL1 = TensorBuilder("g2SL1")
g2SL1c = TensorBuilder("g2SL1c")

# VF(SM)
zSigma0L = TensorBuilder("zSigma0L")
zSigma0Lc = TensorBuilder("zSigma0Lc")
zSigma0R = TensorBuilder("zSigma0R")
zSigma0Rc = TensorBuilder("zSigma0Rc")
zSigma0maj = TensorBuilder("zSigma0maj")
zSigma0majc = TensorBuilder("zSigma0majc")
zSigma1 = TensorBuilder("zSigma1")
zSigma1c = TensorBuilder("zSigma1c")
zDelta1 = TensorBuilder("zDelta1")
zDelta1c = TensorBuilder("zDelta1c")
zDelta3 = TensorBuilder("zDelta3")
zDelta3c = TensorBuilder("zDelta3c")
zNL = TensorBuilder("zNL")
zNLc = TensorBuilder("zNLc")
zNR = TensorBuilder("zNR")
zNRc = TensorBuilder("zNRc")
zNmaj = TensorBuilder("zNmaj")
zNmajc = TensorBuilder("zNmajc")
zE = TensorBuilder("zE")
zEc = TensorBuilder("zEc")
zXUD = TensorBuilder("zXUD")
zXUDc = TensorBuilder("zXUDc")
zUDY = TensorBuilder("zUDY")
zUDYc = TensorBuilder("zUDYc")
zUDu = TensorBuilder("zUDu")
zUDuc = TensorBuilder("zUDuc")
zUDd = TensorBuilder("zUDd")
zUDdc = TensorBuilder("zUDdc")
zXU = TensorBuilder("zXU")
zXUc = TensorBuilder("zXUc")
zDY = TensorBuilder("zDY")
zDYc = TensorBuilder("zDYc")
zU = TensorBuilder("zU")
zUc = TensorBuilder("zUc")
zD = TensorBuilder("zD")
zDc = TensorBuilder("zDc")

# SF(SM)
wuS = TensorBuilder("wuS")
wuSc = TensorBuilder("wuSc")
wdS = TensorBuilder("wdS")
wdSc = TensorBuilder("wdSc")
weS = TensorBuilder("weS")
weSc = TensorBuilder("weSc")
wqS = TensorBuilder("wqS")
wqSc = TensorBuilder("wqSc")
wlS = TensorBuilder("wlS")
wlSc = TensorBuilder("wlSc")
wuXi0 = TensorBuilder("wuXi0")
wuXi0c = TensorBuilder("wuXi0c")
wdXi0 = TensorBuilder("wdXi0")
wdXi0c = TensorBuilder("wdXi0c")
weXi0 = TensorBuilder("weXi0")
weXi0c = TensorBuilder("weXi0c")
wqXi0 = TensorBuilder("wqXi0")
wqXi0c = TensorBuilder("wqXi0c")
wlXi0 = TensorBuilder("wlXi0")
wlXi0c = TensorBuilder("wlXi0c")
wuXi1 = TensorBuilder("wuXi1")
wuXi1c = TensorBuilder("wuXi1c")
wdXi1 = TensorBuilder("wdXi1")
wdXi1c = TensorBuilder("wdXi1c")
weSigma0LXi1 = TensorBuilder("weSigma0LXi1")
weSigma0LXi1c = TensorBuilder("weSigma0LXi1c")
weSigma0RXi1 = TensorBuilder("weSigma0RXi1")
weSigma0RXi1c = TensorBuilder("weSigma0RXi1c")
weSigma0majXi1 = TensorBuilder("weSigma0majXi1")
weSigma0majXi1c = TensorBuilder("weSigma0majXi1c")
wl3Xi1 = TensorBuilder("wl3Xi1")
wl3Xi1c = TensorBuilder("wl3Xi1c")
wq7Xi1 = TensorBuilder("wq7Xi1")
wq7Xi1c = TensorBuilder("wq7Xi1c")
wq5Xi1 = TensorBuilder("wq5Xi1")
wq5Xi1c = TensorBuilder("wq5Xi1c")


# -- Lagrangian --------------------------------------------------------------

L_VDS = -OpSum(
    Op(deltaB(0, 1), B(2, 0), D(2, S(1))),
    Op(deltaW(0, 1), W(2, 3, 0), D(2, Xi0(3, 1))),
    Op(deltaL1(0, 1), L1c(2, 3, 0), D(2, varphi(3, 1))),
    Op(deltaL1c(0, 1), L1(2, 3, 0), D(2, varphic(3, 1))),
    Op(deltaW1(0, 1), W1c(2, 3, 0), D(2, Xi1(3, 1))),
    Op(deltaW1c(0, 1), W1(2, 3, 0), D(2, Xi1c(3, 1))))

L_VVS = -OpSum(
    Op(epsilonS(0, 1, 2), S(0), L1c(3, 4, 1), L1(3, 4, 2)),
    Op(epsilonXi0(0, 1, 2), Xi0(3, 0), L1c(4, 5, 1), sigmaSU2(3, 5, 6),
       L1(4, 6, 2)),
    Op(epsilonXi1(0, 1, 2), Xi1(3, 0), L1c(4, 5, 1), sigmaSU2(3, 5, 6),
       epsSU2(6, 7), L1c(4, 7, 2)),
    Op(epsilonXi1c(0, 1, 2), Xi1c(3, 0), epsSU2(4, 5), L1(6, 5, 2),
       sigmaSU2(3, 4, 7), L1(6, 7, 1)))

L_VSSM = -OpSum(
    Op(g1Xi1L1(0, 1), epsSU2(2, 3), phi(3), sigmaSU2(4, 2, 5),
       D(6, Xi1c(4, 0)), L1(6, 5, 1)),
    Op(g1Xi1L1c(0, 1), L1c(2, 3, 1), sigmaSU2(4, 3, 5),
       D(2, Xi1(4, 0)), epsSU2(5, 6), phic(6)),
    Op(g1Xi0L1(0, 1), phic(2), sigmaSU2(3, 2, 4), D(5, Xi0(3, 0)),
       L1(5, 4, 1)),
    Op(g1Xi0L1c(0, 1), L1c(2, 3, 1), sigmaSU2(4, 3, 5), D(2, Xi0(4, 0)),
       phi(5)),
    Op(g1SL1(0, 1), phic(2), D(3, S(0)), L1(3, 2, 1)),
    Op(g1SL1c(0, 1), L1c(2, 3, 1), D(2, S(0)), phi(3)),

    Op(g2Xi1L1(0, 1), epsSU2(2, 3), D(4, phi(3)), sigmaSU2(5, 2, 6),
       Xi1c(5, 0), L1(4, 6, 1)),
    Op(g2Xi1L1c(0, 1), L1c(2, 3, 1), sigmaSU2(4, 3, 5), Xi1(4, 0),
       epsSU2(5, 6), D(2, phic(6))),
    Op(g2Xi0L1(0, 1), D(2, phic(3)), sigmaSU2(4, 3, 5), Xi0(4, 0),
       L1(2, 5, 1)),
    Op(g2Xi0L1c(0, 1), L1c(2, 3, 1), sigmaSU2(4, 3, 5), Xi0(4, 0),
       D(2, phi(5))),
    Op(g2SL1(0, 1), D(2, phic(3)), S(0), L1(2, 3, 1)),
    Op(g2SL1c(0, 1), L1c(2, 3, 1), S(0), D(2, phi(3))))

L_VFSM = -OpSum(
    Op(zSigma0L(0, 1, 2), Sigma0Lc(3, 4, 0), sigma4bar(5, 3, 6),
       epsSU2(7, 8), L1(5, 8, 1), sigmaSU2(4, 7, 9), lL(6, 9, 2)),
    Op(zSigma0Lc(0, 1, 2), lLc(3, 4, 2), sigmaSU2(5, 4, 6),
       epsSU2(6, 7), L1c(8, 7, 1), sigma4bar(8, 3, 9), Sigma0L(9, 5, 0)),
    Op(zSigma0R(0, 1, 2), Sigma0R(3, 4, 0), epsDownDot(3, 5),
       sigma4bar(6, 5, 7),
       epsSU2(8, 9), L1(6, 9, 1), sigmaSU2(4, 8, 10), lL(7, 10, 2)),
    Op(zSigma0Rc(0, 1, 2), lLc(3, 4, 2), sigmaSU2(5, 4, 6), epsSU2(6, 7),
       L1c(8, 7, 1), sigma4bar(8, 3, 9), epsDown(9, 10), Sigma0Rc(10, 5, 0)),
    
    Op(zSigma0maj(0, 1, 2), Sigma0majc(3, 4, 0), sigma4bar(5, 3, 6),
       epsSU2(7, 8), L1(5, 8, 1), sigmaSU2(4, 7, 9), lL(6, 9, 2)),
    Op(zSigma0majc(0, 1, 2), lLc(3, 4, 2), sigmaSU2(5, 4, 6),
       epsSU2(6, 7), L1c(8, 7, 1), sigma4bar(8, 3, 9), Sigma0maj(9, 5, 0)),
    
    Op(zSigma1(0, 1, 2), Sigma1Lc(3, 4, 0), sigma4bar(5, 3, 6),
       L1c(5, 7, 1), sigmaSU2(4, 7, 8), lL(6, 8, 2)),
    Op(zSigma1c(0, 1, 2), lLc(3, 4, 2), sigmaSU2(5, 4, 6), L1(7, 6, 1),
       sigma4bar(7, 3, 8), Sigma1L(8, 5, 0)),

    Op(zDelta1(0, 1, 2), Delta1Rc(3, 4, 0), sigma4(5, 3, 6), L1(5, 4, 1),
       eR(6, 2)),
    Op(zDelta1c(0, 1, 2), eRc(3, 2), sigma4(4, 3, 5), L1c(4, 6, 1),
       Delta1R(5, 6, 0)),

    Op(zDelta3(0, 1, 2), Delta3Rc(3, 4, 0), sigma4(5, 3, 6),
       epsSU2(4, 7), L1c(5, 7, 1), eR(6, 2)),
    Op(zDelta3c(0, 1, 2), eRc(3, 2), sigma4(4, 3, 5),
       epsSU2(6, 7), L1(4, 7, 1), Delta3R(5, 6, 0)),

    Op(zNL(0, 1, 2), NLc(3, 0), sigma4bar(4, 3, 5),
       epsSU2(6, 7), L1(4, 7, 1), lL(5, 6, 2)),
    Op(zNLc(0, 1, 2), lLc(3, 4, 2), sigma4bar(5, 3, 6),
       epsSU2(4, 7), L1c(5, 7, 1), NL(6, 0)),
    Op(zNR(0, 1, 2), NR(3, 0), epsDownDot(3, 4), sigma4bar(5, 4, 6),
       epsSU2(7, 8), L1(5, 8, 1), lL(6, 7, 2)),
    Op(zNRc(0, 1, 2), lLc(3, 4, 2), sigma4bar(5, 3, 6),
       epsSU2(4, 7), L1c(5, 7, 1), epsDown(6, 8), NRc(8, 0)),

    Op(zNmaj(0, 1, 2), Nmajc(3, 0), sigma4bar(4, 3, 5),
       epsSU2(6, 7), L1(4, 7, 1), lL(5, 6, 2)),
    Op(zNmajc(0, 1, 2), lLc(3, 4, 2), sigma4bar(5, 3, 6),
       epsSU2(4, 7), L1c(5, 7, 1), Nmaj(6, 0)),

    Op(zE(0, 1, 2), ELc(3, 0), sigma4bar(4, 3, 5), L1c(4, 6, 1), lL(5, 6, 2)),
    Op(zEc(0, 1, 2), lLc(3, 4, 2), L1(5, 4, 1), sigma4bar(5, 3, 6), EL(6, 0)),

    Op(zXUD(0, 1, 2), XUDLc(3, 4, 5, 0), sigma4bar(6, 3, 7),
       epsSU2(8, 9), L1(6, 9, 1), sigmaSU2(5, 8, 10), qL(7, 4, 10, 2)),
    Op(zXUDc(0, 1, 2), qLc(3, 4, 5, 2), sigmaSU2(6, 5, 7),
       epsSU2(7, 8), L1c(9, 8, 1), sigma4bar(9, 3, 10), XUDL(10, 4, 6, 0)),

    Op(zUDY(0, 1, 2), UDYLc(3, 4, 5, 0), sigma4bar(6, 3, 7),
       L1c(6, 8, 1), sigmaSU2(5, 8, 9), qL(7, 4, 9, 2)),
    Op(zUDYc(0, 1, 2), qLc(3, 4, 5, 2), sigmaSU2(6, 5, 7), L1(8, 7, 1),
       sigma4bar(8, 3, 9), UDYL(9, 4, 6, 0)),
    Op(zUDu(0, 1, 2), UDRc(3, 4, 5, 0), sigma4(6, 3, 7),
       epsSU2(5, 8), L1c(6, 8, 1), uR(7, 4, 2)),
    Op(zUDuc(0, 1, 2), uRc(3, 4, 2), sigma4(5, 3, 6),
       epsSU2(7, 8), L1(5, 8, 1), UDR(6, 4, 7, 0)),

    Op(zUDd(0, 1, 2), UDRc(3, 4, 5, 0), sigma4(6, 3, 7), L1(6, 5, 1),
       dR(7, 4, 2)),
    Op(zUDdc(0, 1, 2), dRc(3, 4, 2), sigma4(5, 3, 6), L1c(5, 7, 1),
       UDR(6, 4, 7, 0)),

    Op(zXU(0, 1, 2), XURc(3, 4, 5, 0), sigma4(6, 3, 7),
       L1(6, 5, 1), uR(7, 4, 2)),
    Op(zXUc(0, 1, 2), uRc(3, 4, 2), sigma4(5, 3, 6),
       L1c(5, 7, 1), XUR(6, 4, 7, 0)),

    Op(zDY(0, 1, 2), DYRc(3, 4, 5, 0), sigma4(6, 3, 7),
       epsSU2(5, 8), L1c(6, 8, 1), dR(7, 4, 2)),
    Op(zDYc(0, 1, 2), dRc(3, 4, 2), sigma4(5, 3, 6),
       epsSU2(7, 8), L1(5, 8, 1), DYR(6, 4, 7, 0)),

    Op(zU(0, 1, 2), ULc(3, 4, 0), sigma4bar(5, 3, 6),
       epsSU2(7, 8), L1(5, 8, 1), qL(6, 4, 7, 2)),
    Op(zUc(0, 1, 2), qLc(3, 4, 5, 2), sigma4bar(6, 3, 7),
       epsSU2(5, 8), L1c(6, 8, 1), UL(7, 4, 0)),

    Op(zD(0, 1, 2), DLc(3, 4, 0), sigma4bar(5, 3, 6),
       L1c(5, 7, 1), qL(6, 4, 7, 2)),
    Op(zDc(0, 1, 2), qLc(3, 4, 5, 2), sigma4bar(6, 3, 7),
       L1(6, 5, 1), DL(7, 4, 0))
)

L_SFSM = -OpSum(
    Op(wuS(0, 1, 2), S(0), ULc(3, 4, 1), uR(3, 4, 2)),
    Op(wuSc(0, 1, 2), S(0), uRc(3, 4, 2), UL(3, 4, 1)),
    Op(wdS(0, 1, 2), S(0), DLc(3, 4, 1), dR(3, 4, 2)),
    Op(wdSc(0, 1, 2), S(0), dRc(3, 4, 2), DL(3, 4, 1)),
    Op(weS(0, 1, 2), S(0), ELc(3, 1), eR(3, 2)),
    Op(weSc(0, 1, 2), S(0), eRc(3, 2), EL(3, 1)),
    Op(wqS(0, 1, 2), S(0), UDRc(3, 4, 5, 1), qL(3, 4, 5, 2)),
    Op(wqSc(0, 1, 2), S(0), qLc(3, 4, 5, 2), UDR(3, 4, 5, 1)),
    Op(wlS(0, 1, 2), S(0), Delta1Rc(3, 4, 1), lL(3, 4, 2)),
    Op(wlSc(0, 1, 2), S(0), lLc(3, 4, 2), Delta1R(3, 4, 1)),

    Op(wuXi0(0, 1, 2), Xi0(3, 0), XUDLc(4, 5, 3, 1), uR(4, 5, 2)),
    Op(wuXi0c(0, 1, 2), Xi0(3, 0), uRc(4, 5, 2), XUDL(4, 5, 3, 1)),
    Op(wdXi0(0, 1, 2), Xi0(3, 0), UDYLc(4, 5, 3, 1), dR(4, 5, 2)),
    Op(wdXi0c(0, 1, 2), Xi0(3, 0), dRc(4, 5, 2), UDYL(4, 5, 3, 1)),
    Op(weXi0(0, 1, 2), Xi0(3, 0), Sigma1Lc(4, 3, 1), eR(4, 2)),
    Op(weXi0c(0, 1, 2), Xi0(3, 0), eRc(4, 2), Sigma1L(4, 3, 1)),
    Op(wqXi0(0, 1, 2), Xi0(3, 0), UDRc(4, 5, 6, 1),
       sigmaSU2(3, 6, 7), qL(4, 5, 7, 2)),
    Op(wqXi0c(0, 1, 2), Xi0(3, 0), qLc(4, 5, 6, 2),
       sigmaSU2(3, 6, 7), UDR(4, 5, 7, 1)),
    Op(wlXi0(0, 1, 2), Xi0(3, 0), Delta1Rc(4, 5, 1),
       sigmaSU2(3, 5, 6), lL(4, 6, 2)),
    Op(wlXi0c(0, 1, 2), Xi0(3, 0), lLc(4, 5, 2),
       sigmaSU2(3, 5, 6), Delta1R(4, 6, 1)),

    Op(wuXi1(0, 1, 2), Xi1c(3, 0), UDYLc(4, 5, 3, 1), uR(4, 5, 2)),
    Op(wuXi1c(0, 1, 2), Xi1(3, 0), uRc(4, 5, 2), UDYL(4, 5, 3, 1)),
    Op(wdXi1(0, 1, 2), Xi1c(3, 0), XUDLc(4, 5, 3, 1), dR(4, 5, 2)),
    Op(wdXi1c(0, 1, 2), Xi1(3, 0), dRc(4, 5, 2), XUDL(4, 5, 3, 1)),
    Op(weSigma0RXi1(0, 1, 2), Xi1c(3, 0), Sigma0Rc(4, 3, 1),
       epsDown(4, 6), eRc(6, 2)),
    Op(weSigma0RXi1c(0, 1, 2), Xi1(3, 0), eR(4, 2),
       epsDownDot(4, 5), Sigma0R(5, 3, 1)),
    Op(weSigma0LXi1c(0, 1, 2), Xi1(3, 0), Sigma0Lc(4, 3, 1), eR(4, 2)),
    Op(weSigma0LXi1(0, 1, 2), Xi1c(3, 0), eRc(4, 2), Sigma0L(4, 3, 1)),
    Op(weSigma0majXi1c(0, 1, 2), Xi1(3, 0), Sigma0majc(4, 3, 1), eR(4, 2)),
    Op(weSigma0majXi1(0, 1, 2), Xi1c(3, 0), eRc(4, 2), Sigma0maj(4, 3, 1)),
    Op(wl3Xi1(0, 1, 2), Xi1c(3, 0), Delta3Rc(4, 5, 1),
       sigmaSU2(3, 5, 6), lL(4, 6, 2)),
    Op(wl3Xi1c(0, 1, 2), Xi1(3, 0), lLc(4, 5, 2),
       sigmaSU2(3, 5, 6), Delta3R(4, 6, 1)),
    Op(wq7Xi1(0, 1, 2), Xi1(3, 0), XURc(4, 5, 6, 1),
       sigmaSU2(3, 6, 7), qL(4, 5, 7, 2)),
    Op(wq7Xi1c(0, 1, 2), Xi1c(3, 0), qLc(4, 5, 6, 2),
       sigmaSU2(3, 6, 7), XUR(4, 5, 7, 1)),
    Op(wq5Xi1(0, 1, 2), Xi1c(3, 0), DYRc(4, 5, 6, 1),
       sigmaSU2(3, 6, 7), qL(4, 5, 7, 2)),
    Op(wq5Xi1c(0, 1, 2), Xi1(3, 0), qLc(4, 5, 6, 2),
       sigmaSU2(3, 6, 7), DYR(4, 5, 7, 1))
)

L_total = (
    L_quarks + L_leptons + L_scalars + L_vectors + L_L1_plus_vectors +
    L_VDS + L_VVS + L_VSSM + L_VFSM + L_SFSM)


# -- Heavy fields ------------------------------------------------------------

heavy_fields = (
    heavy_scalars + heavy_vectors + heavy_quarks + heavy_leptons + [heavy_L1])


# -- LaTeX representation ----------------------------------------------------

latex_tensors_mixed = {
    # VDS
    "deltaB": r"\delta^{{{}{}}}_{{\mathcal{{B}}}}",
    "deltaW": r"\delta^{{{}{}}}_{{\mathcal{{W}}}}",
    "deltaL1": r"\delta^{{{}{}}}_{{\mathcal{{L}}^1}}",
    "deltaL1c": r"\delta^{{{}{}*}}_{{\mathcal{{L}}^1}}",
    "deltaW1": r"\delta^{{{}{}}}_{{\mathcal{{W}}^1}}",
    "deltaW1c": r"\delta^{{{}{}*}}_{{\mathcal{{W}}^1}}",

    # VVS
    "epsilonS": r"\varepsilon^{{{}{}{}}}_{{\mathcal{{S}}}}",
    "epsilonXi0": r"\varepsilon^{{{}{}{}}}_{{\Xi_0}}",
    "epsilonXi1": r"\varepsilon^{{{}{}{}}}_{{\Xi_1}}",
    "epsilonXi1c": r"\varepsilon^{{{}{}{}*}}_{{\Xi_1}}",
    
    # VS(SM)
    "g1Xi1L1": r"g^{{(1)}}_{{\Xi_{{1{}}}\mathcal{{L}}^1_{{{}}}}}",
    "g1Xi1L1c": r"g^{{(1)*}}_{{\Xi_{{1{}}}\mathcal{{L}}^1_{{{}}}}}",
    "g1Xi0L1": r"g^{{(1)}}_{{\Xi_{{0{}}}\mathcal{{L}}^1_{{{}}}}}",
    "g1Xi0L1c": r"g^{{(1)*}}_{{\Xi_{{0{}}}\mathcal{{L}}^1_{{{}}}}}",
    "g1SL1": r"g^{{(1)}}_{{\mathcal{{S}}_{{{}}}\mathcal{{L}}^1_{{{}}}}}",
    "g1SL1c": r"g^{{(1)*}}_{{\mathcal{{S}}_{{{}}}\mathcal{{L}}^1_{{{}}}}}",
    "g2Xi1L1": r"g^{{(2)}}_{{\Xi_{{1{}}}\mathcal{{L}}^1_{{{}}}}}",
    "g2Xi1L1c": r"g^{{(2)*}}_{{\Xi_{{1{}}}\mathcal{{L}}^1_{{{}}}}}",
    "g2Xi0L1": r"g^{{(2)}}_{{\Xi_{{0{}}}\mathcal{{L}}^1_{{{}}}}}",
    "g2Xi0L1c": r"g^{{(2)*}}_{{\Xi_{{0{}}}\mathcal{{L}}^1_{{{}}}}}",
    "g2SL1": r"g^{{(2)}}_{{\mathcal{{S}}_{{{}}}\mathcal{{L}}^1_{{{}}}}}",
    "g2SL1c": r"g^{{(2)*}}_{{\mathcal{{S}}_{{{}}}\mathcal{{L}}^1_{{{}}}}}",

    # VF(SM)
    "zSigma0L": r"z^{{\Sigma_{{0L}}}}_{{{}{}{}}}",
    "zSigma0Lc": r"z^{{\Sigma_{{0L}}*}}_{{{}{}{}}}",
    "zSigma0R": r"z^{{\Sigma_{{0R}}}}_{{{}{}{}}}",
    "zSigma0Rc": r"z^{{\Sigma_{{0R}}*}}_{{{}{}{}}}",
    "zSigma0maj": r"z^{{\Sigma^{{(maj)}}_{{0}}}}_{{{}{}{}}}",
    "zSigma0majc": r"z^{{\Sigma^{{(maj)}}_{{0}}*}}_{{{}{}{}}}",
    "zSigma1": r"z^{{\Sigma_1}}_{{{}{}{}}}",
    "zSigma1c": r"z^{{\Sigma_1*}}_{{{}{}{}}}",
    "zDelta1": r"z^{{\Delta_1}}_{{{}{}{}}}",
    "zDelta1c": r"z^{{\Delta_1*}}_{{{}{}{}}}",
    "zDelta3": r"z^{{\Delta_3}}_{{{}{}{}}}",
    "zDelta3c": r"z^{{\Delta_3*}}_{{{}{}{}}}",
    "zNL": r"z^{{N_L}}_{{{}{}{}}}",
    "zNLc": r"z^{{N_L*}}_{{{}{}{}}}",
    "zNR": r"z^{{N_R}}_{{{}{}{}}}",
    "zNRc": r"z^{{N_R*}}_{{{}{}{}}}",
    "zNmaj": r"z^{{N^{{maj}}}}_{{{}{}{}}}",
    "zNmajc": r"z^{{N^{{(maj)}}*}}_{{{}{}{}}}",
    "zE": r"z^E_{{{}{}{}}}",
    "zEc": r"z^{{E*}}_{{{}{}{}}}",
    "zXUD": r"z^{{XUD}}_{{{}{}{}}}",
    "zXUDc": r"z^{{XUD*}}_{{{}{}{}}}",
    "zUDY": r"z^{{UDY}}_{{{}{}{}}}",
    "zUDYc": r"z^{{UDY*}}_{{{}{}{}}}",
    "zUDu": r"z^{{UD}}_{{u,{}{}{}}}",
    "zUDuc": r"z^{{UD*}}_{{u,{}{}{}}}",
    "zUDd": r"z^{{UD}}_{{d,{}{}{}}}",
    "zUDdc": r"z^{{UD}}_{{d,{}{}{}}}",
    "zXU": r"z^{{XU}}_{{{}{}{}}}",
    "zXUc": r"z^{{XU*}}_{{{}{}{}}}",
    "zDY": r"z^{{DY}}_{{{}{}{}}}",
    "zDYc": r"z^{{DY*}}_{{{}{}{}}}",
    "zU": r"z^{{U}}_{{{}{}{}}}",
    "zUc": r"z^{{U*}}_{{{}{}{}}}",
    "zD": r"z^{{D}}_{{{}{}{}}}",
    "zDc": r"z^{{D*}}_{{{}{}{}}}",

    # SF(SM)
    "wuS": r"\left(w^u_{{\mathcal{{S}}_{{{}}}}}\right)_{{{}{}}}",
    "wuSc": r"\left(w^u_{{\mathcal{{S}}_{{{}}}}}\right)^*_{{{}{}}}",
    "wdS": r"\left(w^d_{{\mathcal{{S}}_{{{}}}}}\right)_{{{}{}}}",
    "wdSc": r"\left(w^d_{{\mathcal{{S}}_{{{}}}}}\right)^*_{{{}{}}}",
    "weS": r"\left(w^e_{{\mathcal{{S}}_{{{}}}}}\right)_{{{}{}}}",
    "weSc": r"\left(w^e_{{\mathcal{{S}}_{{{}}}}}\right)^*_{{{}{}}}",
    "wqS": r"\left(w^q_{{\mathcal{{S}}_{{{}}}}}\right)_{{{}{}}}",
    "wqSc": r"\left(w^q_{{\mathcal{{S}}_{{{}}}}}\right)^*_{{{}{}}}",
    "wlS": r"\left(w^l_{{\mathcal{{S}}_{{{}}}}}\right)_{{{}{}}}",
    "wlSc": r"\left(w^l_{{\mathcal{{S}}_{{{}}}}}\right)^*_{{{}{}}}",
    "wuXi0": r"\left(w^u_{{\Xi_{{0{}}}}}\right)_{{{}{}}}",
    "wuXi0c": r"\left(w^u_{{\Xi_{{0{}}}}}\right)^*_{{{}{}}}",
    "wdXi0": r"\left(w^d_{{\Xi_{{0{}}}}}\right)_{{{}{}}}",
    "wdXi0c": r"\left(w^d_{{\Xi_{{0{}}}}}\right)^*_{{{}{}}}",
    "weXi0": r"\left(w^e_{{\Xi_{{0{}}}}}\right)_{{{}{}}}",
    "weXi0c": r"\left(w^e_{{\Xi_{{0{}}}}}\right)^*_{{{}{}}}",
    "wqXi0": r"\left(w^q_{{\Xi_{{0{}}}}}\right)_{{{}{}}}",
    "wqXi0c": r"\left(w^q_{{\Xi_{{0{}}}}}\right)^*_{{{}{}}}",
    "wlXi0": r"\left(w^l_{{\Xi_{{0{}}}}}\right)_{{{}{}}}",
    "wlXi0c": r"\left(w^l_{{\Xi_{{0{}}}}}\right)^*_{{{}{}}}",
    "wuXi1": r"\left(w^u_{{\Xi_{{1{}}}}}\right)_{{{}{}}}",
    "wuXi1c": r"\left(w^u_{{\Xi_{{1{}}}}}\right)^*_{{{}{}}}",
    "wdXi1": r"\left(w^d_{{\Xi_{{1{}}}}}\right)_{{{}{}}}",
    "wdXi1c": r"\left(w^d_{{\Xi_{{1{}}}}}\right)^*_{{{}{}}}",
    "weSigma0LXi1":
    r"\left(w^{{e\Sigma_{{0L}}}}_{{\Xi_{{1{}}}}}\right)_{{{}{}}}",
    "weSigma0LXi1c":
    r"\left(w^{{e\Sigma_{{0L}}}}_{{\Xi_{{1{}}}}}\right)^*_{{{}{}}}",
    "weSigma0RXi1":
    r"\left(w^{{e\Sigma_{{0R}}}}_{{\Xi_{{1{}}}}}\right)_{{{}{}}}",
    "weSigma0RXi1c":
    r"\left(w^{{e\Sigma_{{0R}}}}_{{\Xi_{{1{}}}}}\right)^*_{{{}{}}}",
    "weSigma0majXi1":
    r"\left(w^{{e\Sigma^{{(maj)}}_{{0}}}}_{{\Xi_{{1{}}}}}\right)_{{{}{}}}",
    "weSigma0majXi1c":
    r"\left(w^{{e\Sigma^{{(maj)}}_{{0}}}}_{{\Xi_{{1{}}}}}\right)^*_{{{}{}}}",
    "wl3Xi1": r"\left(w^{{l(3)}}_{{\Xi_{{1{}}}}}\right)_{{{}{}}}",
    "wl3Xi1c": r"\left(w^{{l(3)}}_{{\Xi_{{1{}}}}}\right)^*_{{{}{}}}",
    "wq7Xi1": r"\left(w^{{l(7)}}_{{\Xi_{{1{}}}}}\right)_{{{}{}}}",
    "wq7Xi1c": r"\left(w^{{l(7)}}_{{\Xi_{{1{}}}}}\right)^*_{{{}{}}}",
    "wq5Xi1": r"\left(w^{{l(5)}}_{{\Xi_{{1{}}}}}\right)_{{{}{}}}",
    "wq5Xi1c": r"\left(w^{{l(5)}}_{{\Xi_{{1{}}}}}\right)^*_{{{}{}}}"}



if __name__ == "__main__":

    # -- Integration ---------------------------------------------------------

    eff_lag = integrate(heavy_fields, L_total, 6)


    # Remove operators that are not mixed contributions

    vector_masses = [
        "MB", "MW", "MG", "MH", "MB1", "MW1", "MG1",
        "ML1", "ML3", "MU2", "MU5", "MQ1", "MQ5", "MX",
        "MY1", "MU5"]

    scalar_masses = [
        "MS", "MS1", "MS2", "Mvarphi", "MXi0", "MXi1",
        "MTheta1", "MTheta3", "Momega1", "Momega2", "Momega4",
        "MPi1", "MPi7", "Mzeta", "MOmega1", "MOmega2",
        "MOmega4", "MUpsilon", "MPhi",
    ]

    lepton_masses = [
        "MN", "ME", "MDelta1", "MDelta3", "MSigma0", "MSigma1", 
        "MNmaj", "MSigma0maj"]

    quark_masses = ["MU", "MD", "MXU", "MUD", "MDY", "MXUD", "MUDY"]

    mixed_eff_lag = group_op_sum(OpSum(*[
        op for op in eff_lag.operators
        if ((any(op.contains_symbol(vmass) for vmass in vector_masses) and
             any(op.contains_symbol(smass) for smass in scalar_masses)) or
            (any(op.contains_symbol(vmass) for vmass in vector_masses) and
             any(op.contains_symbol(qmass) for qmass in quark_masses)) or
            (any(op.contains_symbol(vmass) for vmass in vector_masses) and
             any(op.contains_symbol(lmass) for lmass in lepton_masses)) or
            (any(op.contains_symbol(smass) for smass in scalar_masses) and
             any(op.contains_symbol(qmass) for qmass in quark_masses)) or
            (any(op.contains_symbol(smass) for smass in scalar_masses) and
             any(op.contains_symbol(lmass) for lmass in lepton_masses)))]))

    # -- Transformations -----------------------------------------------------

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

        (Op(D(0, phic(1)), D(0, D(2, D(2, phi(1))))),
         OpSum(-Op(D(0, D(0, phic(1))), D(2, D(2, phi(1)))))),

        (Op(D(0, phi(1)), D(0, D(2, D(2, phic(1))))),
         OpSum(-Op(D(0, D(0, phic(1))), D(2, D(2, phi(1)))))),
        

        # Higgs and epsSU2
        
        (Op(phic(0), epsSU2(0, 1), phic(1)), OpSum()),
        (Op(phi(0), epsSU2(0, 1), phi(1)), OpSum()),

        
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

        
        # Four-fermion Fierz reorderings

        (Op(lLc(0, 1, -1), eR(0, -2), eRc(2, -3), lL(2, 1, -4)),
         OpSum(-number_op(0.5) * Ole(-1, -4, -3, -2))),

        (Op(qLc(0, 1, 2, -1), dR(0, 1, -2), dRc(3, 4, -3), qL(3, 4, 2, -4)),
         OpSum(-number_op(1./6) * O1qd(-1, -4, -3, -2),
               -O8qd(-1, -4, -3, -2))),

        (Op(qLc(0, 1, 2, -1), uR(0, 1, -2), uRc(3, 4, -3), qL(3, 4, 2, -4)),
         OpSum(-number_op(1./6) * O1qu(-1, -4, -3, -2),
               -O8qu(-1, -4, -3, -2))),


        # Pass derivatives from fermions to Higgs (int. by parts)

        (Op(D(0, phic(1)), D(0, eRc(2, -1)), lL(2, 1, -2)),
         OpSum(-Op(D(0, phic(1)), eRc(2, -1), D(0, lL(2, 1, -2))),
               -Op(D(0, D(0, phic(1))), eRc(2, -1), lL(2, 1, -2)))),

        (Op(D(0, phi(1)), lLc(2, 1, -2), D(0, eR(2, -1))),
         OpSum(-Op(D(0, phi(1)), D(0, lLc(2, 1, -2)), eR(2, -1)),
               -Op(D(0, D(0, phi(1))), lLc(2, 1, -2), eR(2, -1)))),

        (Op(D(0, phic(1)), D(0, dRc(2, 3, -1)), qL(2, 3, 1, -2)),
         OpSum(-Op(D(0, phic(1)), dRc(2, 3, -1), D(0, qL(2, 3, 1, -2))),
               -Op(D(0, D(0, phic(1))), dRc(2, 3, -1), qL(2, 3, 1, -2)))),

        (Op(D(0, phi(1)), qLc(2, 3, 1, -2), D(0, dR(2, 3, -1))),
         OpSum(-Op(D(0, phi(1)), D(0, qLc(2, 3, 1, -2)), dR(2, 3, -1)),
               -Op(D(0, D(0, phi(1))), qLc(2, 3, 1, -2), dR(2, 3, -1)))),

        (Op(epsSU2(4, 1), D(0, phi(1)), D(0, uRc(2, 3, -1)), qL(2, 3, 4, -2)),
         OpSum(-Op(epsSU2(0, 1), D(2, phi(1)), uRc(3, 4, -1),
                   D(2, qL(3, 4, 0, -2))),
               -Op(epsSU2(4, 1), D(0, D(0, phi(1))),
                   uRc(2, 3, -1), qL(2, 3, 4, -2)))),

        (Op(epsSU2(4, 1), D(0, phic(1)),
            qLc(2, 3, 4, -2), D(0, uR(2, 3, -1))),
         OpSum(-Op(epsSU2(0, 1), D(2, phic(1)),
                   D(2, qLc(3, 4, 0, -2)), uR(3, 4, -1)),
               -Op(epsSU2(4, 1), D(0, D(0, phic(1))),
                   qLc(2, 3, 4, -2), uR(2, 3, -1))))]

    Ofphi2Ophif = [
        (Op(deltaB(-1, -2), phic(0), D(1, phi(0)),
            qLc(2, 3, 4, -3), sigma4bar(1, 2, 5), qL(5, 3, 4, -4)),
         OpSum(-Op(deltaB(-1, -2), D(1, phic(0)), phi(0),
                   qLc(2, 3, 4, -3), sigma4bar(1, 2, 5), qL(5, 3, 4, -4)),
               -Op(deltaB(-1, -2), phic(0), phi(0), D(1, qLc(2, 3, 4, -3)),
                   sigma4bar(1, 2, 5), qL(5, 3, 4, -4)),
               -Op(deltaB(-1, -2), phic(0), phi(0), qLc(2, 3, 4, -3),
                   sigma4bar(1, 2, 5), D(1, qL(5, 3, 4, -4))))),

        (Op(deltaB(-1, -2), phic(0), D(1, phi(0)),
            lLc(2, 3, -3), sigma4bar(1, 2, 5), lL(5, 3, -4)),
         OpSum(-Op(deltaB(-1, -2), D(1, phic(0)), phi(0),
                   lLc(2, 3, -3), sigma4bar(1, 2, 5), lL(5, 3, -4)),
               -Op(deltaB(-1, -2), phic(0), phi(0),
                   D(1, lLc(2, 3, -3)), sigma4bar(1, 2, 5), lL(5, 3, -4)),
               -Op(deltaB(-1, -2), phic(0), phi(0),
                   lLc(2, 3, -3), sigma4bar(1, 2, 5), D(1, lL(5, 3, -4))))),

        (Op(deltaW(-1, -2), phic(0), sigmaSU2(6, 0, 7), D(1, phi(7)),
            qLc(2, 3, 4, -3), sigma4bar(1, 2, 5),
            sigmaSU2(6, 4, 8), qL(5, 3, 8, -4)),
         OpSum(-Op(deltaW(-1, -2), D(1, phic(0)), sigmaSU2(6, 0, 7), phi(7),
                   qLc(2, 3, 4, -3), sigma4bar(1, 2, 5),
                   sigmaSU2(6, 4, 8), qL(5, 3, 8, -4)),
               -Op(deltaW(-1, -2), phic(0), sigmaSU2(6, 0, 7), phi(7),
                   D(1, qLc(2, 3, 4, -3)), sigma4bar(1, 2, 5),
                   sigmaSU2(6, 4, 8), qL(5, 3, 8, -4)),
               -Op(deltaW(-1, -2), phic(0), sigmaSU2(6, 0, 7), phi(7),
                   qLc(2, 3, 4, -3), sigma4bar(1, 2, 5),
                   sigmaSU2(6, 4, 8), D(1, qL(5, 3, 8, -4))))),

        (Op(deltaW(-1, -2), phic(0), sigmaSU2(6, 0, 7), D(1, phi(7)),
            lLc(2, 3, -3), sigma4bar(1, 2, 5),
            sigmaSU2(6, 3, 8), lL(5, 8, -4)),
         OpSum(-Op(deltaW(-1, -2), D(1, phic(0)), sigmaSU2(6, 0, 7), phi(7),
                   lLc(2, 3, -3), sigma4bar(1, 2, 5),
                   sigmaSU2(6, 3, 8), lL(5, 8, -4)),
               -Op(deltaW(-1, -2), phic(0), sigmaSU2(6, 0, 7), phi(7),
                   D(1, lLc(2, 3, -3)), sigma4bar(1, 2, 5),
                   sigmaSU2(6, 3, 8), lL(5, 8, -4)),
               -Op(deltaW(-1, -2), phic(0), sigmaSU2(6, 0, 7), phi(7),
                   lLc(2, 3, -3), sigma4bar(1, 2, 5),
                   sigmaSU2(6, 3, 8), D(1, lL(5, 8, -4))))),

        (Op(deltaB(-1, -2), phic(0), D(1, phi(0)),
            dRc(2, 3, -3), sigma4(1, 2, 5), dR(5, 3, -4)),
         OpSum(-Op(deltaB(-1, -2), D(1, phic(0)), phi(0),
                   dRc(2, 3, -3), sigma4(1, 2, 5), dR(5, 3, -4)),
               -Op(deltaB(-1, -2), phic(0), phi(0),
                   D(1, dRc(2, 3, -3)), sigma4(1, 2, 5), dR(5, 3, -4)),
               -Op(deltaB(-1, -2), phic(0), phi(0),
                   dRc(2, 3, -3), sigma4(1, 2, 5), D(1, dR(5, 3, -4))))),

        (Op(deltaB(-1, -2), phic(0), D(1, phi(0)),
            uRc(2, 3, -3), sigma4(1, 2, 5), uR(5, 3, -4)),
         OpSum(-Op(deltaB(-1, -2), D(1, phic(0)), phi(0),
                   uRc(2, 3, -3), sigma4(1, 2, 5), uR(5, 3, -4)),
               -Op(deltaB(-1, -2), phic(0), phi(0),
                   D(1, uRc(2, 3, -3)), sigma4(1, 2, 5), uR(5, 3, -4)),
               -Op(deltaB(-1, -2), phic(0), phi(0),
                   uRc(2, 3, -3), sigma4(1, 2, 5), D(1, uR(5, 3, -4))))),

        (Op(deltaB(-1, -2), phic(0), D(1, phi(0)),
            eRc(2, -3), sigma4(1, 2, 5), eR(5, -4)),
         OpSum(-Op(deltaB(-1, -2), D(1, phic(0)), phi(0),
                   eRc(2, -3), sigma4(1, 2, 5), eR(5, -4)),
               -Op(deltaB(-1, -2), phic(0), phi(0),
                   D(1, eRc(2, -3)), sigma4(1, 2, 5), eR(5, -4)),
               -Op(deltaB(-1, -2), phic(0), phi(0),
                   eRc(2, -3), sigma4(1, 2, 5), D(1, eR(5, -4)))))]

    transpose_epsSU2 = [(Op(epsSU2(-1, -2)), OpSum(-Op(epsSU2(-2, -1))))]

    all_rules = (specific_rules + Ofphi2Ophif
                 + rules_SU2 + rules_Lorentz
                 + eoms_SM + rules_basis_definitions + transpose_epsSU2)

    transf_eff_lag = apply_rules(mixed_eff_lag, all_rules, 4)

    
    # -- Output --------------------------------------------------------------
    
    transf_eff_lag_writer = Writer(transf_eff_lag,
                                   latex_basis_coefs.keys())

    sys.stdout.write(str(transf_eff_lag_writer) + "\n")

    latex_tensors = {}
    latex_tensors.update(latex_tensors_mixed)
    latex_tensors.update(latex_tensors_scalars)
    latex_tensors.update(latex_tensors_leptons)
    latex_tensors.update(latex_tensors_quarks)
    latex_tensors.update(latex_tensors_vectors)
    latex_tensors.update(latex_tensors_L1)
    latex_tensors.update(latex_SM)
    latex_tensors.update(latex_SU2)
    latex_tensors.update(latex_SU3)
    latex_tensors.update(latex_Lorentz)

    transf_eff_lag_writer.show_pdf(
        "mixed", "open", latex_tensors, latex_basis_coefs,
        list(map(chr, range(ord('a'), ord('z')))))
