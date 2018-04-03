"""
This module defines the Standard Model tensors and fields and the rules
for substituting their equations of motion.
"""

from fractions import Fraction

from matchingtools.core import (
    Op, OpSum, TensorBuilder, FieldBuilder, D,
    i_op, number_op, sigma4, sigma4bar, boson, fermion)

from matchingtools.extras.SU2 import sigmaSU2, epsSU2

from matchingtools.extras.Lorentz import eps4

# Coupling constants
gb = TensorBuilder("gb")
r""":math:`U(1)` coupling constant :math:`g'`"""
gw = TensorBuilder("gw")
r""":math:`SU(2)` coupling constant :math:`g`"""
mu2phi = TensorBuilder("mu2phi")
r"""Higgs quadratic coupling :math:`\mu^2_\phi`"""
lambdaphi = TensorBuilder("lambdaphi")
r"""Higgs quartic coupling :math:`\lambda_\phi`"""
ye = TensorBuilder("ye")
r"""Yukawa coupling for leptons :math:`y^e_{ij}`"""
yec = TensorBuilder("yec")
r"""Conjugate of the Yukawa coupling for leptons :math:`y^{e*}_{ij}`"""
yd = TensorBuilder("yd")
r"""Yukawa coupling for down quarks :math:`y^d_{ij}`"""
ydc = TensorBuilder("ydc")
r"""Conjugate of the Yukawa coupling for down quarks :math:`y^{d*}_{ij}`"""
yu = TensorBuilder("yu")
r"""Yukawa coupling for up quarks :math:`y^u_{ij}`"""
yuc = TensorBuilder("yuc")
r"""Conjugate of the Yukawa coupling for up quarks :math:`y^{u*}_{ij}`"""
V = TensorBuilder("V")
r"""CKM matrix"""
Vc = TensorBuilder("Vc")
r"""Conjugate of the CKM matrix"""

deltaFlavor = TensorBuilder("deltaFlavor")
r"""Kronecker delta for flavor indices"""

# Field. Indices appear in this order when needed:
# Lorentz, SU(3), SU(2), U(1), flavor 
phi = FieldBuilder("phi", 1, boson)
r"""Higgs doublet"""
phic = FieldBuilder("phic", 1, boson)
r"""Conjugate of the Higgs doublet"""

lL = FieldBuilder("lL", 1.5, fermion)
r"""Lepton left-handed doublet"""
lLc = FieldBuilder("lLc", 1.5, fermion)
r"""Conjugate of the lepton left-handed doublet"""

qL = FieldBuilder("qL", 1.5, fermion)
r"""Quark left-handed doublet"""
qLc = FieldBuilder("qLc", 1.5, fermion)
r"""Conjugate of the quark left-handed doublet"""

eR = FieldBuilder("eR", 1.5, fermion)
r"""Electron right-handed singlet"""
eRc = FieldBuilder("eRc", 1.5, fermion)
r"""Conjugate of the electron right-handed singlet"""

dR = FieldBuilder("dR", 1.5, fermion)
r"""Down quark right-handed doublet"""
dRc = FieldBuilder("dRc", 1.5, fermion)
r"""Conjugate of the down quark right-handed doublet"""

uR = FieldBuilder("uR", 1.5, fermion)
r"""Up quark right-handed doublet"""
uRc = FieldBuilder("uRc", 1.5, fermion)
r"""Conjugate of the up quark right-handed doublet"""

bFS = FieldBuilder("bFS", 2, boson)
r""":math:`U(1)` gauge field strength :math:`B_{\mu\nu}`"""
wFS = FieldBuilder("wFS", 2, boson)
r""":math:`SU(2)` gauge field strength :math:`W^a_{\mu\nu}`"""
gFS = FieldBuilder("gFS", 2, boson)
r""":math:`SU(3)` gauge field strength :math:`G^A_{\mu\nu}`"""

# Equations of motion

eom_phi = (
    Op(D(0, D(0, phi(-1)))),
    OpSum(Op(mu2phi(), phi(-1)),
          -number_op(2) * Op(lambdaphi(), phic(0), phi(0), phi(-1)),
          -Op(yec(0, 1), eRc(2, 1), lL(2, -1, 0)),
          -Op(ydc(0, 1), dRc(2, 3, 1), qL(2, 3, -1, 0)),
          -Op(Vc(0, 1), yu(0, 2), qLc(3, 4, 5, 1), uR(3, 4, 2), epsSU2(5, -1))))
r"""
Rule using the Higgs doublet equation of motion. 
Substitute :math:`D^2\phi` by

.. math::
    \mu^2_\phi \phi- 2\lambda_\phi (\phi^\dagger\phi) \phi
    - y^{e*}_{ij} \bar{e}_{Rj} l_{Li} - y^{d*}_{ij} \bar{d}_{Rj} q_{Li}
    + V^*_{ki} y^u_{kj} i\sigma^2 \bar{q}^T_{Li} u_{Rj}
"""

eom_phic = (
    Op(D(0, D(0, phic(-1)))),
    OpSum(Op(mu2phi(), phic(-1)),
          -number_op(2) * Op(lambdaphi(), phic(0), phi(0), phic(-1)),
          -Op(ye(0, 1), lLc(2, -1, 0), eR(2, 1)),
          -Op(yd(0, 1), qLc(2, 3, -1, 0), dR(2, 3, 1)),
          -Op(V(0, 1), yuc(0, 2), uRc(3, 4, 2), qL(3, 4, 5, 1), epsSU2(5, -1))))
r"""
Rule using the conjugate of the Higgs doublet equation of motion. 
Substitute :math:`D^2\phi^\dagger` by

.. math::
    \mu^2_\phi \phi^\dagger- 2\lambda_\phi (\phi^\dagger\phi) \phi^\dagger
    - y^e_{ij} \bar{l}_{Li} e_{Rj} - y^d_{ij} \bar{q}_{Li} d_{Rj}
    - V_{ki} y^{u*}_{kj} \bar{u}_{Rj} q^T_{Li} i\sigma^2
"""

eom_bFS = (
    Op(D(0, bFS(0, -1))),
    -OpSum(number_op(-Fraction(1, 2)) * Op(gb(), deltaFlavor(2, 4), lLc(0, 1, 2),
                                 sigma4bar(-1, 0, 3), lL(3, 1, 4)),
           number_op(Fraction(1, 6)) * Op(gb(), deltaFlavor(3, 5),  qLc(0, 1, 2, 3),
                                sigma4bar(-1, 0, 4), qL(4, 1, 2, 5)),
           number_op(-1) * Op(gb(), deltaFlavor(1, 3), eRc(0, 1),
                               sigma4(-1, 0, 2), eR(2, 3)),
           number_op(-Fraction(1, 3)) * Op(gb(), deltaFlavor(2, 4), dRc(0, 1, 2),
                                 sigma4(-1, 0, 3), dR(3, 1, 4)),
           number_op(Fraction(2, 3)) * Op(gb(), deltaFlavor(2, 4), uRc(0, 1, 2),
                                sigma4(-1, 0, 3), uR(3, 1, 4)),
           number_op(Fraction(1, 2)) * i_op * Op(gb(), phic(0), D(-1, phi(0))),
           - number_op(Fraction(1, 2)) * i_op * Op(gb(), D(-1, phic(0)), phi(0))))
r"""
Rule using the :math:`U(1)` gauge field strength equation of motion. 
Substitute :math:`D_\mu B^{\mu\nu}` by

.. math::
    -g' \sum_f Y_f \bar{f} \gamma^\nu f
    -(\frac{i}{2} g' \phi^\dagger D^\nu \phi + h.c.)

where :math:`\sum_f` runs over all SM fermions :math:`f` and the  :math:`Y_f` are
the hypercharges.
"""

eom_wFS = (
    Op(D(0, wFS(0, -1, -2))),
    -OpSum(number_op(Fraction(1, 2)) * Op(gw(), deltaFlavor(0, 1), lLc(3, 4, 0),
                                sigma4bar(-1, 3, 5), sigmaSU2(-2, 4, 6),
                                lL(5, 6, 1)),
           number_op(Fraction(1, 2)) * Op(gw(), deltaFlavor(0, 1), qLc(3, 4, 5, 0),
                                sigma4bar(-1, 3, 6), sigmaSU2(-2, 5, 7),
                                qL(6, 4, 7, 1)),
           number_op(Fraction(1, 2)) * i_op * Op(gw(), phic(0), sigmaSU2(-2, 0, 1), D(-1, phi(1))),
           number_op(-Fraction(1, 2)) * i_op * Op(gw(), D(-1, phic(0)), sigmaSU2(-2, 0, 1), phi(1))))
r"""
Rule using the :math:`SU(2)` gauge field strength equation of motion.
Substitute :math:`D_\mu W^{a,\mu\nu}` by

.. math::
    -\frac{1}{2} g \sum_F \bar{F} \sigma^a \gamma^\nu F
    -(\frac{i}{2} g \phi^\dagger \sigma^a D^\nu \phi + h.c.)

where :math:`\sum_F` runs over the SM fermion doublets :math:`F`.
"""

eom_lL = (
    Op(sigma4bar(0, -1, 1), D(0, lL(1, -2, -3))),
    OpSum(- i_op * Op(ye(-3, 0), phi(-2), eR(-1, 0))))
r"""
Rule using the equation of motion of the left-handed lepton doublet.
Substitute :math:`\gamma^\mu D_\mu l_{Li}` by
:math:`-iy^e_{ij}\phi e_{Rj}`.
"""

eom_lLc = (
    Op(sigma4bar(0, 1, -1), D(0, lLc(1, -2, -3))),
    OpSum(i_op * Op(yec(-3, 0), phic(-2), eRc(-1, 0))))
r"""
Rule using the equation of motion of conjugate of the left-handed lepton doublet.
Substitute :math:`D_\mu\bar{l}_{Li}\gamma^\mu` by
:math:`iy^{e*}_{ij} \bar{e}_{Rj}\phi^\dagger`.
"""
    
eom_qL = (
    Op(sigma4bar(0, -1, 1), D(0, qL(1, -2, -3, -4))),
    OpSum(- i_op * Op(yd(-4, 0), phi(-3), dR(-1, -2, 0)),
          - i_op * Op(Vc(0, -4), yu(0, 1), epsSU2(-3, 2),
                              phic(2), uR(-1, -2, 1))))
r"""
Rule using the equation of motion of the left-handed quark doublet.
Substitute :math:`\gamma^\mu D_\mu q_{Li}` by
:math:`-iy^d_{ij}\phi d_{Rj}-i V^*_{ji}y^u_{kj} \tilde{\phi} u_{Rj}`.
"""

eom_qLc = (
    Op(sigma4bar(0, 1, -1), D(0, qLc(1, -2, -3, -4))),
    OpSum(i_op * Op(ydc(-4, 0), phic(-3), dRc(-1, -2, 0)),
          i_op * Op(V(0, -4), yuc(0, 1), epsSU2(-3, 2),
                             phi(2), uRc(-1, -2, 1))))
r"""
Rule using the equation of motion of the conjugate of the left-handed quark doublet.
Substitute :math:`D_\mu \bar{q}_{Li}\gamma^\mu` by
:math:`iy^{d*}_{ij} \bar{d}_{Rj}\phi^\dagger+
i V_{ji}y^{u*}_{kj} \bar{u}_{Rj}\tilde{\phi}^\dagger`.
"""

eom_eR = (
    Op(sigma4(0, -1, 1), D(0, eR(1, -2))),
    OpSum(- i_op * Op(yec(0, -2), phic(1), lL(-1, 1, 0))))
r"""
Rule using the equation of motion of the right-handed electron singlet.
Substitute :math:`\gamma^\mu D_\mu e_{Rj}` by
:math:`-i y^{e*}_{ij} \phi^\dagger l_{Li}`.
"""

eom_eRc = (
    Op(sigma4(0, 1, -1), D(0, eRc(1, -2))),
    OpSum(i_op * Op(ye(0, -2), phi(1), lLc(-1, 1, 0))))
r"""
Rule using the equation of motion of the conjugate of the right-handed
electron singlet. Substitute :math:`D_\mu \bar{e}_{Rj} \gamma^\mu` by
:math:`i y^e_{ij} \bar{l}_{Li}\phi`.
"""

eom_dR = (
    Op(sigma4(0, -1, 1), D(0, dR(1, -2, -3))),
    OpSum(- i_op * Op(ydc(0, -3), phic(1), qL(-1, -2, 1, 0))))
r"""
Rule using the equation of motion of the right-handed down quark singlet.
Substitute :math:`\gamma^\mu D_\mu d_{Rj}` by
:math:`-i y^{d*}_{ij} \phi^\dagger q_{Li}`.
"""

eom_dRc = (
    Op(sigma4(0, 1, -1), D(0, dRc(1, -2, -3))),
    OpSum(i_op * Op(yd(0, -3), phi(1), qLc(-1, -2, 1, 0))))
r"""
Rule using the equation of motion of the conjugate of the right-handed
down quark singlet. Substitute :math:`D_\mu \bar{d}_{Rj} \gamma^\mu` by
:math:`i y^d_{ij} \bar{q}_{Li}\phi`.
"""

eom_uR = (
    Op(sigma4(0, -1, 1), D(0, uR(1, -2, -3))),
    OpSum(- i_op * Op(V(0, 1), yuc(0, -3), epsSU2(2, 3),
                              phi(3), qL(-1, -2, 2, 1))))
r"""
Rule using the equation of motion of the right-handed up quark singlet.
Substitute :math:`\gamma^\mu D_\mu u_{Rj}` by
:math:`-i V_{ki} y^{u*}_{kj} \tilde{\phi}^\dagger q_{Li}`.
"""

eom_uRc = (
    Op(sigma4(0, 1, -1), D(0, uRc(1, -2, -3))),
    OpSum(i_op * Op(Vc(0, 1), yu(0, -3), epsSU2(2, 3),
                             phic(3), qLc(-1, -2, 2, 1))))
r"""
Rule using the equation of motion of the conjugate of the right-handed up 
quark singlet. Substitute :math:`D_\mu \bar{u}_{Rj}\gamma^\mu` by
:math:`i V^*_{ki} y^u_{kj} \bar{q}_{Li} \tilde{\phi}`.
"""

eoms_SM = [eom_phi, eom_phic, eom_bFS, eom_wFS,
           eom_lL, eom_lLc, eom_qL, eom_qLc,
           eom_eR, eom_eRc, eom_dR, eom_dRc, eom_uR, eom_uRc]
r"""
Rules that use the equations of motion of the Standard Model fields
to substitute some expressions involving their derivatives by other
combinations of the fields.
"""

latex_SM = {
    # Tensors
    "gb": "g'",
    "gw": "g",
    "mu2phi": r"\mu^2_\phi",
    "lambdaphi": r"\lambda_\phi",
    "ye": r"\delta_{{{0}{1}}}y^e_{{{0}{0}}}",
    "yec": r"\delta_{{{0}{1}}}y^{{e*}}_{{{0}{0}}}",
    "yd": r"\delta_{{{0}{1}}}y^d_{{{0}{0}}}",
    "ydc": r"\delta_{{{0}{1}}}y^{{d*}}_{{{0}{0}}}",
    "yu": r"y^u_{{{}{}}}",
    "yuc": r"y^{{u*}}_{{{}{}}}",
    "V": r"V_{{{}{}}}",
    "Vc": r"V^\dagger_{{{1}{0}}}",
    "deltaFlavor": r"\delta_{{{}{}}}",
    
    # Fields
    "phi": r"\phi_{}",
    "phic": r"\phi^*_{}",
    "lL": r"l_{{L{}{}{}}}",
    "lLc": r"l^c_{{L\dot{{{}}}{}{}}}",
    "qL": r"q_{{L{}{}{}{}}}",
    "qLc": r"q^c_{{L\dot{{{}}}{}{}{}}}",
    "eR": r"e^{{\dot{{{}}}}}_{{R{}}}",
    "eRc": r"e^{{c{}}}_{{R{}}}",
    "dR": r"d^{{\dot{{{}}}}}_{{R{}{}}}",
    "dRc": r"d^{{c{}}}_{{R{}{}}}",
    "uR": r"u^{{\dot{{{}}}}}_{{R{}{}}}",
    "uRc": r"u^{{c{}}}_{{R{}{}}}",
    "bFS": r"B_{{{}{}}}",
    "wFS": r"W^{{{2}}}_{{{0}{1}}}",
    "gFS": r"G^{{{2}}}_{{{0}{1}}}"}
r"""
LaTeX code representation of the tensors and fields of the Standard Model.
"""
