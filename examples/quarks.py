"""
This script defines all the heavy quarks (color-triplet fermions)
that couple linearly through renormalizable interactions to the 
Standard Model. It specifies their interaction lagrangian and 
integrates them out.
"""

from __future__ import print_function


# -- Core tools --------------------------------------------------------------

from effective.core import (
    TensorBuilder, FieldBuilder, D, Op, OpSum,
    number_op, symbol_op, tensor_op, flavor_tensor_op,
    boson, fermion, kdelta)

from effective.integration import integrate, VectorLikeFermion

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
lambdaP1 = TensorBuilder("lambdaP1")
lambdaP1c = TensorBuilder("lambdaP1c")
lambdaP2 = TensorBuilder("lambdaP2")
lambdaP2c = TensorBuilder("lambdaP2c")
lambdaP3u = TensorBuilder("lambdaP3u")
lambdaP3uc = TensorBuilder("lambdaP3uc")
lambdaP3d = TensorBuilder("lambdaP3d")
lambdaP3dc = TensorBuilder("lambdaP3dc")
lambdaP4 = TensorBuilder("lambdaP4")
lambdaP4c = TensorBuilder("lambdaP4c")
lambdaP5 = TensorBuilder("lambdaP5")
lambdaP5c = TensorBuilder("lambdaP5c")
lambdaP6 = TensorBuilder("lambdaP6")
lambdaP6c = TensorBuilder("lambdaP6c")
lambdaP7 = TensorBuilder("lambdaP7")
lambdaP7c = TensorBuilder("lambdaP7c")

# Heavy - heavy
lambdaTUDU = TensorBuilder("lambdaTUDU")
lambdaTUDUc = TensorBuilder("lambdaTUDUc")
lambdaTUDD = TensorBuilder("lambdaTUDD")
lambdaTUDDc = TensorBuilder("lambdaTUDDc")
lambdaTUDXUD = TensorBuilder("lambdaTUDXUD")
lambdaTUDXUDc = TensorBuilder("lambdaTUDXUDc")
lambdaTUDUDY = TensorBuilder("lambdaTUDUDY")
lambdaTUDUDYc = TensorBuilder("lambdaTUDUDYc")
lambdaTXUU = TensorBuilder("lambdaTXUU")
lambdaTXUUc = TensorBuilder("lambdaTXUUc")
lambdaTXUXUD = TensorBuilder("lambdaTXUXUD")
lambdaTXUXUDc = TensorBuilder("lambdaTXUXUDc")
lambdaTDYD = TensorBuilder("lambdaTDYD")
lambdaTDYDc = TensorBuilder("lambdaTDYDc")
lambdaTDYUDY = TensorBuilder("lambdaTDYUDY")
lambdaTDYUDYc = TensorBuilder("lambdaTDYUDYc")


# -- Fields ------------------------------------------------------------------

UL = FieldBuilder("UL", 1, boson)
UR = FieldBuilder("UR", 1, boson)
ULc = FieldBuilder("ULc", 1, boson)
URc = FieldBuilder("URc", 1, boson)

DL = FieldBuilder("DL", 1, boson)
DR = FieldBuilder("DR", 1, boson)
DLc = FieldBuilder("DLc", 1, boson)
DRc = FieldBuilder("DRc", 1, boson)

XUL = FieldBuilder("XUL", 1, boson)
XUR = FieldBuilder("XUR", 1, boson)
XULc = FieldBuilder("XULc", 1, boson)
XURc = FieldBuilder("XURc", 1, boson)

UDL = FieldBuilder("UDL", 1, boson)
UDR = FieldBuilder("UDR", 1, boson)
UDLc = FieldBuilder("UDLc", 1, boson)
UDRc = FieldBuilder("UDRc", 1, boson)

DYL = FieldBuilder("DYL", 1, boson)
DYR = FieldBuilder("DYR", 1, boson)
DYLc = FieldBuilder("DYLc", 1, boson)
DYRc = FieldBuilder("DYRc", 1, boson)

XUDL = FieldBuilder("XUDL", 1, boson)
XUDR = FieldBuilder("XUDR", 1, boson)
XUDLc = FieldBuilder("XUDLc", 1, boson)
XUDRc = FieldBuilder("XUDRc", 1, boson)

UDYL = FieldBuilder("UDYL", 1, boson)
UDYR = FieldBuilder("UDYR", 1, boson)
UDYLc = FieldBuilder("UDYLc", 1, boson)
UDYRc = FieldBuilder("UDYRc", 1, boson)


# -- Lagrangian --------------------------------------------------------------

L_quarks = -OpSum(
    # U
    Op(lambdaP1(0, 1), V(1, 2), URc(3, 4, 0),
       epsSU2(5, 6), phi(6), qL(3, 4, 5, 2)),
    Op(lambdaP1c(0, 1), Vc(1, 2), qLc(3, 4, 5, 2),
       epsSU2(5, 6), phic(6), UR(3, 4, 0)),

    # D
    Op(lambdaP2(0, 1), DRc(2, 3, 0), phic(4), qL(2, 3, 4, 1)),
    Op(lambdaP2c(0, 1), qLc(2, 3, 4, 1), phi(4), DR(2, 3, 0)),

    # UD
    Op(lambdaP3u(0, 1), UDLc(2, 3, 4, 0), epsSU2(4, 5), phic(5), uR(2, 3, 1)),
    Op(lambdaP3uc(0, 1), uRc(2, 3, 1), epsSU2(4, 5), phi(5), UDL(2, 3, 4, 0)),
    Op(lambdaP3d(0, 1), UDLc(2, 3, 4, 0), phi(4), dR(2, 3, 1)),
    Op(lambdaP3dc(0, 1), dRc(2, 3, 1), phic(4), UDL(2, 3, 4, 0)),

    # XU
    Op(lambdaP4(0, 1), XULc(2, 3, 4, 0), phi(4), uR(2, 3, 1)),
    Op(lambdaP4c(0, 1), uRc(2, 3, 1), phic(4), XUL(2, 3, 4, 0)),

    # DY
    Op(lambdaP5(0, 1), DYLc(2, 3, 4, 0), epsSU2(4, 5), phic(5), dR(2, 3, 1)),
    Op(lambdaP5c(0, 1), dRc(2, 3, 1), epsSU2(4, 5), phi(5), DYL(2, 3, 4, 0)),

    # XUD
    number_op(0.5) * Op(lambdaP6(0, 1), XUDRc(2, 3, 4, 0), epsSU2(5, 6),
                        phi(6), sigmaSU2(4, 5, 7), qL(2, 3, 7, 1)),
    number_op(0.5) * Op(lambdaP6c(0, 1), qLc(2, 3, 7, 1), sigmaSU2(4, 7, 5),
                        epsSU2(5, 6), phic(6), XUDR(2, 3, 4, 0)),

    # UDY
    number_op(0.5) * Op(lambdaP7(0, 1), UDYRc(2, 3, 4, 0), phic(5),
                        sigmaSU2(4, 5, 6), qL(2, 3, 6, 1)),
    number_op(0.5) * Op(lambdaP7c(0, 1), qLc(2, 3, 6, 1), sigmaSU2(4, 6, 5),
                        phi(5), UDYR(2, 3, 4, 0)),

    # UD and U
    Op(lambdaTUDU(0, 1), UDRc(2, 3, 4, 0),
       epsSU2(4, 5), phic(5), UL(2, 3, 1)),
    Op(lambdaTUDUc(0, 1), ULc(2, 3, 1),
       epsSU2(4, 5), phi(5), UDR(2, 3, 4, 0)),

    # UD and D
    Op(lambdaTUDD(0, 1), UDRc(2, 3, 4, 0), phi(4), DL(2, 3, 1)),
    Op(lambdaTUDDc(0, 1), DLc(2, 3, 1), phic(4), UDR(2, 3, 4, 0)),

    # UD and XUD
    number_op(0.5) * Op(lambdaTUDXUD(0, 1), UDRc(2, 3, 4, 0),
                        sigmaSU2(5, 4, 6), epsSU2(6, 7), phic(7),
                        XUDL(2, 3, 5, 1)),
    number_op(0.5) * Op(lambdaTUDXUDc(0, 1), XUDLc(2, 3, 5, 1),
                        epsSU2(6, 7), phi(7), sigmaSU2(5, 6, 4),
                        UDR(2, 3, 4, 0)),

    # UD and UDY
    number_op(0.5) * Op(lambdaTUDUDY(0, 1), UDRc(2, 3, 4, 0),
                        sigmaSU2(5, 4, 6), phi(6), UDYL(2, 3, 5, 1)),
    number_op(0.5) * Op(lambdaTUDUDYc(0, 1), UDYLc(2, 3, 5, 1),
                        phic(6), sigmaSU2(5, 6, 4), UDR(2, 3, 4, 0)),

    # XU and U
    Op(lambdaTXUU(0, 1), XURc(2, 3, 4, 0), phi(4), UL(2, 3, 1)),
    Op(lambdaTXUUc(0, 1), ULc(2, 3, 1), phic(4), XUR(2, 3, 4, 0)),

    # XU and XUD
    number_op(0.5) * Op(lambdaTXUXUD(0, 1), XURc(2, 3, 4, 0),
                        sigmaSU2(5, 4, 6), phi(6), XUDL(2, 3, 5, 1)),
    number_op(0.5) * Op(lambdaTXUXUDc(0, 1), XUDLc(2, 3, 5, 1),
                        phic(6), sigmaSU2(5, 6, 4), XUR(2, 3, 4, 0)),

    # DY and D
    Op(lambdaTDYD(0, 1), DYRc(2, 3, 4, 0),
       epsSU2(4, 5), phic(5), DL(2, 3, 1)),
    Op(lambdaTDYDc(0, 1), DLc(2, 3, 1),
       epsSU2(4, 5), phi(5), DYR(2, 3, 4, 0)),

    # DY and UDY
    number_op(0.5) * Op(lambdaTDYUDY(0, 1), DYRc(2, 3, 4, 0),
                        sigmaSU2(5, 4, 6), epsSU2(6, 7), phic(7),
                        UDYL(2, 3, 5, 1)),
    number_op(0.5) * Op(lambdaTDYUDYc(0, 1), UDYLc(2, 3, 5, 1),
                        epsSU2(6, 7), phi(7), sigmaSU2(5, 6, 4),
                        DYR(2, 3, 4, 0)))

heavy_U = VectorLikeFermion("U", "UL", "UR", "ULc", "URc", 3)
heavy_D = VectorLikeFermion("D", "DL", "DR", "DLc", "DRc", 3)
heavy_XU = VectorLikeFermion("XU", "XUL", "XUR", "XULc", "XURc", 4)
heavy_UD = VectorLikeFermion("UD", "UDL", "UDR", "UDLc", "UDRc", 4)
heavy_DY = VectorLikeFermion("DY", "DYL", "DYR", "DYLc", "DYRc", 4)
heavy_XUD = VectorLikeFermion("XUD", "XUDL", "XUDR", "XUDLc", "XUDRc", 4)
heavy_UDY = VectorLikeFermion("UDY", "UDYL", "UDYR", "UDYLc", "UDYRc", 4)
heavy_quarks = [heavy_U, heavy_D, heavy_XU, heavy_UD, heavy_DY,
                heavy_XUD, heavy_UDY]
"""
All the heavy quark (SU(3) triplet fermion) fields that couple linearly 
through renormalizable interactions to the Standard Model.
"""


# -- LaTeX representation ----------------------------------------------------

latex_tensors_quarks = {
    # Heavy - light
    "lambdaP1": r"\lambda'^{{(1)}}_{{{}{}}}",
    "lambdaP1c": r"\lambda'^{{(1)*}}_{{{}{}}}",
    "lambdaP2": r"\lambda'^{{(2)}}_{{{}{}}}",
    "lambdaP2c": r"\lambda'^{{(2)*}}_{{{}{}}}",
    "lambdaP3u": r"\lambda'^{{(3)u}}_{{{}{}}}",
    "lambdaP3uc": r"\lambda'^{{(3)u*}}_{{{}{}}}",
    "lambdaP3d": r"\lambda'^{{(3)d}}_{{{}{}}}",
    "lambdaP3dc": r"\lambda'^{{(3)d*}}_{{{}{}}}",
    "lambdaP4": r"\lambda'^{{(4)}}_{{{}{}}}",
    "lambdaP4c": r"\lambda'^{{(4)*}}_{{{}{}}}",
    "lambdaP5": r"\lambda'^{{(5)}}_{{{}{}}}",
    "lambdaP5c": r"\lambda'^{{(5)*}}_{{{}{}}}",
    "lambdaP6": r"\lambda'^{{(6)}}_{{{}{}}}",
    "lambdaP6c": r"\lambda'^{{(6)*}}_{{{}{}}}",
    "lambdaP7": r"\lambda'^{{(7)}}_{{{}{}}}",
    "lambdaP7c": r"\lambda'^{{(7)*}}_{{{}{}}}",

    # Heavy - heavy
    "lambdaTUDU": r"(\tilde{{\lambda}}^{{UD}}_{{U}})_{{{}{}}}",
    "lambdaTUDUc": r"(\tilde{{\lambda}}^{{UD}}_{{U}})^*_{{{}{}}}",
    "lambdaTUDD": r"(\tilde{{\lambda}}^{{UD}}_{{D}})_{{{}{}}}",
    "lambdaTUDDc": r"(\tilde{{\lambda}}^{{UD}}_{{D}})^*_{{{}{}}}",
    "lambdaTUDXUD": r"(\tilde{{\lambda}}^{{UD}}_{{XUD}})_{{{}{}}}",
    "lambdaTUDXUDc": r"(\tilde{{\lambda}}^{{UD}}_{{XUD}})^*_{{{}{}}}",
    "lambdaTUDUDY": r"(\tilde{{\lambda}}^{{UD}}_{{UDY}})_{{{}{}}}",
    "lambdaTUDUDYc": r"(\tilde{{\lambda}}^{{UD}}_{{UDY}})^*_{{{}{}}}",
    "lambdaTXUU": r"(\tilde{{\lambda}}^{{XU}}_{{U}})_{{{}{}}}",
    "lambdaTXUUc": r"(\tilde{{\lambda}}^{{XU}}_{{U}})^*_{{{}{}}}",
    "lambdaTXUXUD": r"(\tilde{{\lambda}}^{{XU}}_{{XUD}})_{{{}{}}}",
    "lambdaTXUXUDc": r"(\tilde{{\lambda}}^{{XU}}_{{XUD}})^*_{{{}{}}}",
    "lambdaTDYD": r"(\tilde{{\lambda}}^{{DY}}_{{D}})_{{{}{}}}",
    "lambdaTDYDc": r"(\tilde{{\lambda}}^{{DY}}_{{D}})^*_{{{}{}}}",
    "lambdaTDYUDY": r"(\tilde{{\lambda}}^{{DY}}_{{UDY}})_{{{}{}}}",
    "lambdaTDYUDYc": r"(\tilde{{\lambda}}^{{DY}}_{{UDY}})^*_{{{}{}}}",

    # Masses
    "MU": r"M_{{U_{}}}",
    "MD": r"M_{{D_{}}}",
    "MXU": r"M_{{XU_{}}}",
    "MUD": r"M_{{UD_{}}}",
    "MDY": r"M_{{DY_{}}}",
    "MXUD": r"M_{{XUD_{}}}",
    "MUDY": r"M_{{UDY_{}}}"}


if __name__ == "__main__":
    
    # -- Integration ---------------------------------------------------------
    
    eff_lag = integrate(heavy_quarks, L_quarks, 6)

    
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
    latex_tensors.update(latex_tensors_quarks)
    latex_tensors.update(latex_SM)
    latex_tensors.update(latex_SU2)
    latex_tensors.update(latex_Lorentz)

    eff_lag_writer.write_latex(
        "quarks", latex_tensors, {},
        list(map(chr, range(ord('a'), ord('z')))))
