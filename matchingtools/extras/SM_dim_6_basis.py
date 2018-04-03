"""
This module defines a basis of operators for the Standar Model
effective lagrangian up to dimension 6. 


The basis is the one in arXiv:1412.8480v2_.

.. _arXiv:1412.8480v2: https://arxiv.org/abs/1412.8480v2.
"""

from fractions import Fraction

from matchingtools.core import (
    tensor_op, flavor_tensor_op, D, Op, OpSum, i_op, number_op)

from matchingtools.extras.SM import (
    phi, phic, lL, lLc, eR, eRc, qL, qLc, uR, uRc, dR, dRc, bFS, wFS, gFS,
    ye, yec, yd, ydc, yu, yuc, V, Vc, mu2phi, lambdaphi)

from matchingtools.extras.SU2 import epsSU2, sigmaSU2, epsSU2triplets

from matchingtools.extras.SU3 import epsSU3, TSU3, fSU3

from matchingtools.extras.Lorentz import (
    epsUp, epsUpDot, epsDown, epsDownDot, sigma4, sigma4bar,
    eps4, sigmaTensor)


# -- Standard Model dimension 4 operators --

Okinphi = tensor_op("Okinphi")
r"""
Higgs kinetic term
:math:`\mathcal{O}_{kin,\phi} = (D_\mu \phi)^\dagger D^\mu \phi`.
"""

Ophi4 = tensor_op("Ophi4")
r"""
Higgs quartic coupling
:math:`\mathcal{O}_{\phi 4} = (\phi^\dagger\phi)^2`.
"""

Ophi2 = tensor_op("Ophi2")
r"""
Higgs quadratic coupling
:math:`\mathcal{O}_{\phi 2} = \phi^\dagger\phi`.
"""

Oye = flavor_tensor_op("Oye")
r"""
Lepton Yukawa coupling
:math:`(\mathcal{O}_{y^e})_{ij} = \bar{l}_{Li}\phi e_{Rj}`.
"""

Oyec = flavor_tensor_op("Oyec")
r"""
Conjugate lepton Yukawa coupling
:math:`(\mathcal{O}_{y^e})^*_{ij} = \bar{e}_{Rj}\phi^\dagger l_{Li}`.
"""

Oyd = flavor_tensor_op("Oyd")
r"""
Down quark Yukawa coupling
:math:`(\mathcal{O}_{y^d})_{ij} = \bar{q}_{Li}\phi d_{Rj}`.
"""

Oydc = flavor_tensor_op("Oydc")
r"""
Conjugate down quark Yukawa coupling
:math:`(\mathcal{O}_{y^d})^*_{ij} = \bar{d}_{Rj}\phi^\dagger q_{Li}`.
"""

Oyu = flavor_tensor_op("Oyu")
r"""
Up quark Yukawa coupling
:math:`(\mathcal{O}_{y^u})_{ij} = \bar{q}_{Li}\tilde{\phi} u_{Rj}`.
"""

Oyuc = flavor_tensor_op("Oyuc")
r"""
Conjugate up quark Yukawa coupling
:math:`(\mathcal{O}_{y^d})^*_{ij} = 
\bar{d}_{Rj}\tilde{\phi}^\dagger q_{Li}`.
"""

# -- Standard Model dimension 5 operators --

O5 = flavor_tensor_op("O5")
r"""
Weinberg operator
:math:`\mathcal{O}_5 = 
\overline{l^c}_L\tilde{\phi}^*\tilde{\phi}^\dagger l_L`.
"""

O5c = flavor_tensor_op("O5c")
r"""
Conjugate Weinberg operator
:math:`\mathcal{O}_5 = 
\bar{l}_L\tilde{\phi}\tilde{\phi}^T l^c_L`.
"""

# -- Standard Model dimension 6 four-fermion operators --

# *** LLLL ***

O1ll = flavor_tensor_op("O1ll")
r"""
LLLL type four-fermion operator
:math:`(\mathcal{O}^{(1)}_{ll})_{ijkl}=
\frac{1}{2}(\bar{l}_{Li}\gamma_\mu l_{Lj})
(\bar{l}_{Lk}\gamma^\mu l_{Ll})`.
"""

O1qq = flavor_tensor_op("O1qq")
r"""
LLLL type four-fermion operator
:math:`(\mathcal{O}^{(1)}_{qq})_{ijkl}=
\frac{1}{2}(\bar{q}_{Li}\gamma_\mu q_{Lj})
(\bar{q}_{Lk}\gamma^\mu q_{Ll})`.
"""

O8qq = flavor_tensor_op("O8qq")
r"""
LLLL type four-fermion operator 
:math:`(\mathcal{O}^{(8)}_{qq})_{ijkl}=
\frac{1}{2}(\bar{q}_{Li}T_A \gamma_\mu q_{Lj})
(\bar{q}_{Lk}T_A \gamma^\mu q_{Ll})`.
"""

O1lq = flavor_tensor_op("O1lq")
r"""
LLLL type four-fermion operator
:math:`(\mathcal{O}^{(1)}_{lq})_{ijkl}=
(\bar{l}_{Li}\gamma_\mu l_{Lj})
(\bar{q}_{Lk}\gamma^\mu q_{Ll})`.
"""

O3lq = flavor_tensor_op("O3lq")
r"""
LLLL type four-fermion operator
:math:`(\mathcal{O}^{(8)}_{qq})_{ijkl}=
(\bar{l}_{Li}\sigma_a \gamma_\mu l_{Lj})
(\bar{q}_{Lk}\sigma_a \gamma^\mu q_{Ll})`.
"""

# *** RRRR ***

Oee = flavor_tensor_op("Oee")
r"""
RRRR type four-fermion operator
:math:`(\mathcal{O}_{ee})_{ijkl}=
\frac{1}{2}(\bar{e}_{Ri}\gamma_\mu e_{Rj})
(\bar{e}_{Rk}\gamma^\mu e_{Rl})`.
"""

O1uu = flavor_tensor_op("O1uu")
r"""
RRRR type four-fermion operator
:math:`(\mathcal{O}^{(1)}_{uu})_{ijkl}=
\frac{1}{2}(\bar{u}_{Ri}\gamma_\mu u_{Rj})
(\bar{u}_{Rk}\gamma^\mu u_{Rl})`.
"""

O1dd = flavor_tensor_op("O1dd")
r"""
RRRR type four-fermion operator
:math:`(\mathcal{O}^{(1)}_{dd})_{ijkl}=
\frac{1}{2}(\bar{d}_{Ri}\gamma_\mu d_{Rj})
(\bar{d}_{Rk}\gamma^\mu d_{Rl})`.
"""

O1ud = flavor_tensor_op("O1ud")
r"""
RRRR type four-fermion operator
:math:`(\mathcal{O}^{(1)}_{uu})_{ijkl}=
(\bar{u}_{Ri}\gamma_\mu u_{Rj})
(\bar{d}_{Rk}\gamma^\mu d_{Rl})`.
"""

O8ud = flavor_tensor_op("O8ud")
r"""
RRRR type four-fermion operator
:math:`(\mathcal{O}^{(8)}_{uu})_{ijkl}=
(\bar{u}_{Ri}T_A \gamma_\mu u_{Rj})
(\bar{d}_{Rk}T_A \gamma^\mu d_{Rl})`.
"""

Oeu = flavor_tensor_op("Oeu")
r"""
RRRR type four-fermion operator
:math:`(\mathcal{O}_{eu})_{ijkl}=
(\bar{e}_{Ri}\gamma_\mu e_{Rj})
(\bar{u}_{Rk}\gamma^\mu u_{Rl})`.
"""

Oed = flavor_tensor_op("Oed")
r"""
RRRR type four-fermion operator
:math:`(\mathcal{O}_{ed})_{ijkl}=
(\bar{e}_{Ri}\gamma_\mu e_{Rj})
(\bar{d}_{Rk}\gamma^\mu d_{Rl})`.
"""

# *** LLRR and LRRL ***

Ole = flavor_tensor_op("Ole")
r"""
LLRR type four-fermion operator
:math:`(\mathcal{O}_{le})_{ijkl}=
(\bar{l}_{Li}\gamma_\mu l_{Lj})
(\bar{e}_{Rk}\gamma^\mu e_{Rl})`.
"""

Oqe = flavor_tensor_op("Oqe")
r"""
LLRR type four-fermion operator
:math:`(\mathcal{O}_{qe})_{ijkl}=
(\bar{q}_{Li}\gamma_\mu q_{Lj})
(\bar{e}_{Rk}\gamma^\mu e_{Rl})`.
"""

Olu = flavor_tensor_op("Olu")
r"""
LLRR type four-fermion operator
:math:`(\mathcal{O}_{lu})_{ijkl}=
(\bar{l}_{Li}\gamma_\mu l_{Lj})
(\bar{u}_{Rk}\gamma^\mu u_{Rl})`.
"""

Old = flavor_tensor_op("Old")
r"""
LLRR type four-fermion operator
:math:`(\mathcal{O}_{ld})_{ijkl}=
(\bar{l}_{Li}\gamma_\mu l_{Lj})
(\bar{d}_{Rk}\gamma^\mu d_{Rl})`.
"""

O1qu = flavor_tensor_op("O1qu")
r"""
LLRR type four-fermion operator
:math:`(\mathcal{O}^{(1)}_{qu})_{ijkl}=
(\bar{q}_{Li}\gamma_\mu q_{Lj})
(\bar{u}_{Rk}\gamma^\mu u_{Rl})`.
"""

O8qu = flavor_tensor_op("O8qu")
r"""
LLRR type four-fermion operator
:math:`(\mathcal{O}^{(8)}_{qu})_{ijkl}=
(\bar{q}_{Li}T_A\gamma_\mu q_{Lj})
(\bar{u}_{Rk}T_A\gamma^\mu u_{Rl})`.
"""

O1qd = flavor_tensor_op("O1qd")
r"""
LLRR type four-fermion operator
:math:`(\mathcal{O}^{(1)}_{qd})_{ijkl}=
(\bar{q}_{Li}\gamma_\mu q_{Lj})
(\bar{d}_{Rk}\gamma^\mu d_{Rl})`.
"""

O8qd = flavor_tensor_op("O8qd")
r"""
LLRR type four-fermion operator
:math:`(\mathcal{O}^{(8)}_{qd})_{ijkl}=
(\bar{q}_{Li}T_A\gamma_\mu q_{Lj})
(\bar{d}_{Rk}T_A\gamma^\mu d_{Rl})`.
"""

Oledq = flavor_tensor_op("Oledq")
r"""
LRRL type four-fermion operator
:math:`(\mathcal{O}_{leqd})_{ijkl}=
(\bar{l}_{Li} e_{Rj})
(\bar{d}_{Rk} q_{Ll})`.
"""

Oledqc = flavor_tensor_op("Oledqc")
r"""
LRRL type four-fermion operator
:math:`(\mathcal{O}_{leqd})^*_{ijkl}=
(\bar{e}_{Rj} l_{Li})
(\bar{q}_{Ll} d_{Rk})`.
"""

# *** LRLR ***

O1qud = flavor_tensor_op("O1qud")
r"""
LRLR type four-fermion operator
:math:`(\mathcal{O}^{(1)}_{qud})_{ijkl}=
(\bar{q}_{Li} u_{Rj})i\sigma_2
(\bar{q}_{Lk} d_{Rl})^T`.
"""

O1qudc = flavor_tensor_op("O1qudc")
r"""
LRLR type four-fermion operator
:math:`(\mathcal{O}^{(1)}_{qud})^*_{ijkl}=
(\bar{u}_{Rj} q_{Li})i\sigma_2
(\bar{d}_{Rl} q_{Lk})^T`.
"""

O8qud = flavor_tensor_op("O8qud")
r"""
LRLR type four-fermion operator
:math:`(\mathcal{O}^{(8)}_{qud})_{ijkl}=
(\bar{q}_{Li}T_A u_{Rj})i\sigma_2
(\bar{q}_{Lk}T_A d_{Rl})^T`.
"""

O8qudc = flavor_tensor_op("O8qudc")
r"""
LRLR type four-fermion operator
:math:`(\mathcal{O}^{(8)}_{qud})^*_{ijkl}=
(\bar{u}_{Rj} T_A q_{Li})i\sigma_2
(\bar{d}_{Rl} T_A q_{Lk}})^T`.
"""

Olequ = flavor_tensor_op("Olequ")
r"""
LRLR type four-fermion operator
:math:`(\mathcal{O}_{lequ})_{ijkl}=
(\bar{l}_{Li} e_{Rj})i\sigma_2
(\bar{q}_{Lk} u_{Rl})^T`.
"""

Olequc = flavor_tensor_op("Olequc")
r"""
LRLR type four-fermion operator
:math:`(\mathcal{O}_{lequ})^*_{ijkl}=
(\bar{e}_{Rj} l_{Li})i\sigma_2
(\bar{u}_{Rl} q_{Lk})^T`.
"""

Oluqe = flavor_tensor_op("Oluqe")
r"""
LRLR type four-fermion operator
:math:`(\mathcal{O}_{luqe})_{ijkl}=
(\bar{l}_{Li} u_{Rj})i\sigma_2
(\bar{q}_{Lk} e_{Rl})^T`.
"""

Oluqec = flavor_tensor_op("Oluqec")
r"""
LRLR type four-fermion operator
:math:`(\mathcal{O}_{luqe})^*_{ijkl}=
(\bar{l}_{Li} u_{Rj})i\sigma_2
(\bar{q}_{Lk} e_{Rl})^T`.
"""

# *** \slashed{B} and slashed{L} ***

Olqdu = flavor_tensor_op("Olqdu")
r"""
Four-fermion operator
:math:`(\mathcal{O}_{lqdu})_{ijkl}=
\epsilon_{ABC}(\bar{l}_{Li} i\sigma_2 q^{c,A}_{Lj})
(\bar{d}^B_{Rk} u^{c,C}_{Rl})`.
"""

Olqduc = flavor_tensor_op("Olqduc")
r"""
Four-fermion operator
:math:`(\mathcal{O}_{lqdu})^*_{ijkl}=
-\epsilon_{ABC}(\bar{q}^{c,A}_{Lj}i\sigma_2 l_{Li})
(\bar{u}^{c,C}_{Rl} d^B_{Rk})`.
"""

Oqqeu = flavor_tensor_op("Oqqeu")
r"""
Four-fermion operator
:math:`(\mathcal{O}_{lqdu})_{ijkl}=
\epsilon_{ABC}(\bar{q}^A_{Li} i\sigma_2 q^{c,B}_{Lj})
(\bar{e}_{Rk} u^{c,C}_{Rl})`.
"""

Oqqeuc = flavor_tensor_op("Oqqeuc")
r"""
Four-fermion operator
:math:`(\mathcal{O}_{lqdu})^*_{ijkl}=
-\epsilon_{ABC}(\bar{q}^{c,B}_{Lj}i\sigma_2 q^A_{Li})
(\bar{u}^{c,C}_{Rl} e_{Rk})`.
"""

O1lqqq = flavor_tensor_op("O1lqqq")
r"""
Four-fermion operator
:math:`(\mathcal{O}^{{(1)}}_{lqqq})_{ijkl}=
\epsilon_{ABC}(\bar{l}_{Li} i\sigma_2 q^{c,A}_{Lj})
(\bar{q}^B_{Lk} i\sigma_2 q^{c,C}_{Ll})`.
"""

O1lqqqc = flavor_tensor_op("O1lqqqc")
r"""
Four-fermion operator
:math:`(\mathcal{O}^{{(1)}}_{lqqq})^*_{ijkl}=
\epsilon_{ABC}(\bar{q}^{c,A}_{Lj} i\sigma_2 l_{i})
(\bar{q}^{c,C}_{Ll} i\sigma_2 q^B_{Lk})`.
"""

Oudeu = flavor_tensor_op("Oudeu")
r"""
Four-fermion operator
:math:`(\mathcal{O}_{udeu})_{ijkl}=
\epsilon_{ABC}(\bar{u}^A_{Ri} d^{c,B}_{Rj})
(\bar{e}_{Rk} u^{c,C}_{Rl})`.
"""

Oudeuc = flavor_tensor_op("Oudeuc")
r"""
Four-fermion operator
:math:`(\mathcal{O}_{udeu})^*_{ijkl}=
\epsilon_{ABC}(\bar{d}^{c,B}_{Rj} u^A_{Ri})
(\bar{u}^{c,C}_{Rl} e_{Rk})`.
"""

O3lqqq = flavor_tensor_op("O3lqqq")
r"""
Four-fermion operator
:math:`(\mathcal{O}^{{(3)}}_{lqqq})_{ijkl}=
\epsilon_{ABC}(\bar{l}_{Li} \sigma_a i\sigma_2 q^{c,A}_{Lj})
(\bar{q}^B_{Lk} \sigma_a i\sigma_ 2 q^{c,C}_{Ll})`.
"""

O3lqqqc = flavor_tensor_op("O3lqqqc")
r"""
Four-fermion operator
:math:`(\mathcal{O}^{{(3)}}_{lqqq})^*_{ijkl}=
\epsilon_{ABC}(\bar{q}^{c,A}_{Lj} i sigma_2 \sigma_a l_{Li} )
(\bar{q}^{c,C}_{Ll} i\sigma_2 \sigma_a q^B_{Lk})`.
"""

# -- Standard Model dimension six operators other than four-fermion --

# *** S ***

Ophisq = tensor_op("Ophisq")
r"""
S type operator
:math:`\mathcal{O}_{\phi\square}=\phi^\dagger\phi\square(\phi^\dagger\phi)`.
"""

Ophi = tensor_op("Ophi")
r"""
S type six Higgs interaction operator
:math:`\mathcal{O}_\phi = \frac{1}{3}(\phi^\dagger\phi)^3`.
"""

# *** SVF ***

O1phil = flavor_tensor_op("O1phil")
r"""
SVF type operator :math:`(\mathcal{O}^{(1)}_{\phi l})_{ij}=
(\phi^\dagger i D_\mu \phi)(\bar{l}_{Li}\gamma^\mu l_{Lj})`.
"""

O1philc = flavor_tensor_op("O1philc")
r"""
SVF type operator :math:`(\mathcal{O}^{(1)}_{\phi l})^*_{ij}=
(-i (D_\mu \phi)^\dagger \phi)(\bar{l}_{Lj}\gamma^\mu l_{Li})`.
"""

O1phiq = flavor_tensor_op("O1phiq")
r"""
SVF type operator :math:`(\mathcal{O}^{(1)}_{\phi q})_{ij}=
(\phi^\dagger i D_\mu \phi)(\bar{q}_{Li}\gamma^\mu q_{Lj})`.
"""

O1phiqc = flavor_tensor_op("O1phiqc")
r"""
SVF type operator :math:`(\mathcal{O}^{(1)}_{\phi q})^*_{ij}=
(-i (D_\mu \phi)^\dagger \phi)(\bar{q}_{Lj}\gamma^\mu q_{Li})`.
"""

O3phil = flavor_tensor_op("O3phil")
r"""
SVF type operator :math:`(\mathcal{O}^{(3)}_{\phi l})_{ij}=
(\phi^\dagger i D_\mu \phi)(\bar{l}_{Li}\gamma^\mu l_{Lj})`.
"""

O3philc = flavor_tensor_op("O3philc")
r"""
SVF type operator :math:`(\mathcal{O}^{(3)}_{\phi l})^*_{ij}=
(-i (D_\mu \phi)^\dagger \phi)(\bar{l}_{Lj}\gamma^\mu l_{Li})`.
"""

O3phiq = flavor_tensor_op("O3phiq")
r"""
SVF type operator :math:`(\mathcal{O}^{(3)}_{\phi q})_{ij}=
(\phi^\dagger i D_\mu \phi)(\bar{q}_{Li}\gamma^\mu q_{Lj})`.
"""

O3phiqc = flavor_tensor_op("O3phiqc")
r"""
SVF type operator :math:`(\mathcal{O}^{(3)}_{\phi q})^*_{ij}=
(-i (D_\mu \phi)^\dagger \phi)(\bar{q}_{Lj}\gamma^\mu q_{Li})`.
"""

O1phie = flavor_tensor_op("O1phie")
r"""
SVF type operator :math:`(\mathcal{O}^{(1)}_{\phi e})_{ij}=
(\phi^\dagger i D_\mu \phi)(\bar{e}_{Ri}\gamma^\mu e_{Rj})`.
"""

O1phiec = flavor_tensor_op("O1phiec")
r"""
SVF type operator :math:`(\mathcal{O}^{(1)}_{\phi e})^*_{ij}=
(-i (D_\mu \phi)^\dagger \phi)(\bar{e}_{Rj}\gamma^\mu e_{Ri})`.
"""

O1phid = flavor_tensor_op("O1phid")
r"""
SVF type operator :math:`(\mathcal{O}^{(1)}_{\phi d})_{ij}=
(\phi^\dagger i D_\mu \phi)(\bar{d}_{Ri}\gamma^\mu d_{Rj})`.
"""

O1phidc = flavor_tensor_op("O1phidc")
r"""
SVF type operator :math:`(\mathcal{O}^{(1)}_{\phi d})^*_{ij}=
(-i (D_\mu \phi)^\dagger \phi)(\bar{d}_{Rj}\gamma^\mu d_{Ri})`.
"""

O1phiu = flavor_tensor_op("O1phiu")
r"""
SVF type operator :math:`(\mathcal{O}^{(1)}_{\phi u})_{ij}=
(\phi^\dagger i D_\mu \phi)(\bar{u}_{Ri}\gamma^\mu u_{Rj})`.
"""

O1phiuc = flavor_tensor_op("O1phiuc")
r"""
SVF type operator :math:`(\mathcal{O}^{(1)}_{\phi u})^*_{ij}=
(-i (D_\mu \phi)^\dagger \phi)(\bar{u}_{Rj}\gamma^\mu u_{Ri})`.
"""

Ophiud = flavor_tensor_op("Ophiud")
r"""
SVF type operator :math:`(\mathcal{O}^{(1)}_{\phi ud})_{ij}=
-(\tilde{\phi}^\dagger i D_\mu \phi)(\bar{u}_{Ri}\gamma^\mu d_{Rj})`.
"""

Ophiudc = flavor_tensor_op("Ophiudc")
r"""
SVF type operator :math:`(\mathcal{O}_{\phi ud})^*_{ij}=
(i (D_\mu \phi)^\dagger \tilde{\phi})(\bar{u}_{Rj}\gamma^\mu d_{Ri})`.
"""

# *** STF ***

OeB = flavor_tensor_op("OeB")
r"""
STF type operator :math:`(\mathcal{O}_{eB})_{ij}=
(\bar{l}_{Li}\sigma^{\mu\nu}e_{Rj})\phi B_{\mu\nu}`.
"""

OeBc = flavor_tensor_op("OeBc")
r"""
STF type operator :math:`(\mathcal{O}_{eB})^*_{ij}=
\phi^\dagger (\bar{e}_{Rj}\sigma^{\mu\nu}l_{Li}) B_{\mu\nu}`.
"""

OeW = flavor_tensor_op("OeW")
r"""
STF type operator :math:`(\mathcal{O}_{eW})_{ij}=
(\bar{l}_{Li}\sigma^{\mu\nu}e_{Rj})\sigma^a\phi W^a_{\mu\nu}`.
"""

OeWc = flavor_tensor_op("OeWc")
r"""
STF type operator :math:`(\mathcal{O}_{eW})^*_{ij}=
\phi^\dagger\sigma^a(\bar{e}_{Rj}\sigma^{\mu\nu}l_{Li}) W^a_{\mu\nu}`.
"""

OuB = flavor_tensor_op("OuB")
r"""
STF type operator :math:`(\mathcal{O}_{uB})_{ij}=
(\bar{q}_{Li}\sigma^{\mu\nu}u_{Rj})\tilde{\phi} B_{\mu\nu}`.
"""

OuBc = flavor_tensor_op("OuBc")
r"""
STF type operator :math:`(\mathcal{O}_{uB})^*_{ij}=
\tilde{\phi}^\dagger(\bar{u}_{Rj}\sigma^{\mu\nu}q_{Li}) B_{\mu\nu}`.
"""

OuW = flavor_tensor_op("OuW")
r"""
STF type operator :math:`(\mathcal{O}_{uW})_{ij}=
(\bar{q}_{Li}\sigma^{\mu\nu}u_{Rj})\sigma^a\tilde{\phi} W^a_{\mu\nu}`.
"""

OuWc = flavor_tensor_op("OuWc")
r"""
STF type operator :math:`(\mathcal{O}_{uW})^*_{ij}=
\tilde{\phi}\sigma^a(\bar{u}_{Rj}\sigma^{\mu\nu}q_{Li}) W^a_{\mu\nu}`.
"""

OdB = flavor_tensor_op("OdB")
r"""
STF type operator :math:`(\mathcal{O}_{dB})_{ij}=
(\bar{q}_{Li}\sigma^{\mu\nu}d_{Rj})\phi B_{\mu\nu}`.
"""

OdBc = flavor_tensor_op("OdBc")
r"""
STF type operator :math:`(\mathcal{O}_{dB})^*_{ij}=
\phi^\dagger(\bar{d}_{Rj}\sigma^{\mu\nu}q_{Li}) B_{\mu\nu}`.
"""

OdW = flavor_tensor_op("OdW")
r"""
STF type operator :math:`(\mathcal{O}_{dW})_{ij}=
(\bar{q}_{Li}\sigma^{\mu\nu}d_{Rj})\sigma^a\phi W^a_{\mu\nu}`.
"""

OdWc = flavor_tensor_op("OdWc")
r"""
STF type operator :math:`(\mathcal{O}_{dW})^*_{ij}=
\phi^\dagger\sigma^a(\bar{d}_{Rj}\sigma^{\mu\nu}q_{Li}) W^a_{\mu\nu}`.
"""

OuG = flavor_tensor_op("OuG")
r"""
STF type operator :math:`(\mathcal{O}_{uG})_{ij}=
(\bar{q}_{Li}\sigma^{\mu\nu}T_A u_{Rj})\tilde{\phi} G^A_{\mu\nu}`.
"""

OuGc = flavor_tensor_op("OuGc")
r"""
STF type operator :math:`(\mathcal{O}_{uG})^*_{ij}=
\tilde{\phi}^\dagger(\bar{u}_{Rj}\sigma^{\mu\nu}T_A q_{Li}) G^A_{\mu\nu}`.
"""

OdG = flavor_tensor_op("OdG")
r"""
STF type operator :math:`(\mathcal{O}_{dG})_{ij}=
(\bar{q}_{Li}\sigma^{\mu\nu}T_A d_{Rj})\phi G^A_{\mu\nu}`.
"""

OdGc = flavor_tensor_op("OdGc")
r"""
STF type operator :math:`(\mathcal{O}_{dG})^*_{ij}=
\phi^\dagger(\bar{d}_{Rj}\sigma^{\mu\nu}T_A q_{Li}) G^A_{\mu\nu}`.
"""

# *** SF ***

Oephi = flavor_tensor_op("Oephi")
r"""
SF type operator :math:`(\mathcal{O}_{e\phi})_{ij}=
(\phi^\dagger\phi)(\bar{l}_{Li}\phi e_{Rj})`.
"""

Odphi = flavor_tensor_op("Odphi")
r"""
SF type operator :math:`(\mathcal{O}_{d\phi})_{ij}=
(\phi^\dagger\phi)(\bar{q}_{Li}\phi d_{Rj})`.
"""

Ouphi = flavor_tensor_op("Ouphi")
r"""
SF type operator :math:`(\mathcal{O}_{u\phi})_{ij}=
(\phi^\dagger\phi)(\bar{q}_{Li}\tilde{\phi} u_{Rj})`.
"""

Oephic = flavor_tensor_op("Oephic")
r"""
SF type operator :math:`(\mathcal{O}_{e\phi})^*_{ij}=
(\phi^\dagger\phi)(\bar{e}_{Rj}\phi^\dagger l_{Li})`.
"""

Odphic = flavor_tensor_op("Odphic")
r"""
SF type operator :math:`(\mathcal{O}_{d\phi})^*_{ij}=
(\phi^\dagger\phi)(\bar{d}_{Rj}\phi^\dagger q_{Li})`.
"""

Ouphic = flavor_tensor_op("Ouphic")
r"""
SF type operator :math:`(\mathcal{O}_{u\phi})^*_{ij}=
(\phi^\dagger\phi)(\bar{u}_{Rj}\tilde{\phi}^\dagger q_{Li})`.
"""

# *** Oblique ***

OphiD = tensor_op("OphiD")
r"""
Oblique operator :math:`\mathcal{O}_{\phi D}=(\phi^\dagger D_\mu \phi)((D^\mu\phi)^\dagger\phi)`.
"""

OphiB = tensor_op("OphiB")
r"""
Oblique operator 
:math:`\mathcal{O}_{\phi B}=\phi^\dagger\phi B_{\mu\nu}B^{\mu\nu}`.
"""

OphiBTilde = tensor_op("OphiBTilde")
r"""
Oblique operator 
:math:`\mathcal{O}_{\phi \tilde{B}}=
\phi^\dagger\phi \tilde{B}_{\mu\nu}B^{\mu\nu}`.
"""

OphiW = tensor_op("OphiW")
r"""
Oblique operator 
:math:`\mathcal{O}_{\phi W}=
\phi^\dagger\phi W^a_{\mu\nu}W^{a,\mu\nu}`.
"""

OphiWTilde = tensor_op("OphiWTilde")
r"""
Oblique operator 
:math:`\mathcal{O}_{\phi \tilde{W}}=
\phi^\dagger\phi \tilde{W}^a_{\mu\nu}W^{a,\mu\nu}`.
"""

OWB = tensor_op("OWB")
r"""
Oblique operator 
:math:`\mathcal{O}_{W B}=
\phi^\dagger\sigma^a\phi W^a_{\mu\nu}B^{\mu\nu}`.
"""

OWBTilde = tensor_op("OWBTilde")
r"""
Oblique operator 
:math:`\mathcal{O}_{\tilde{W} B}=
\phi^\dagger\sigma^a\phi \tilde{W}^a_{\mu\nu}B^{\mu\nu}`.
"""

OphiG = tensor_op("OphiG")
r"""
Oblique operator 
:math:`\mathcal{O}_{\phi W}=
\phi^\dagger\phi G^A_{\mu\nu}G^{A,\mu\nu}`.
"""

OphiGTilde = tensor_op("OphiGTilde")
r"""
Oblique operator 
:math:`\mathcal{O}_{\phi \tilde{W}}=
\phi^\dagger\phi \tilde{G}^A_{\mu\nu}G^{A,\mu\nu}`.
"""

# *** Gauge ***

OW = tensor_op("OW")
r"""
Gauge type operator
:math:`\mathcal{O}_W=
\varepsilon_{abc}W^{a,\nu}_\mu W^{b,\rho}_\nu W^{c,\mu}_\rho`.
"""

OWTilde = tensor_op("OWTilde")
r"""
Gauge type operator
:math:`\mathcal{O}_{\tilde{W}}=
\varepsilon_{abc}\tilde{W}^{a,\nu}_\mu
W^{b,\rho}_\nu W^{c,\mu}_\rho`.
"""

OG = tensor_op("OG")
r"""
Gauge type operator
:math:`\mathcal{O}_G=
f_{ABC}G^{A,\nu}_\mu G^{B,\rho}_\nu G^{C,\mu}_\rho`.
"""

OGTilde = tensor_op("OGTilde")
r"""
Gauge type operator
:math:`\mathcal{O}_{\tilde{G}}=
f_{ABC}\tilde{G}^{A,\nu}_\mu
G^{B,\rho}_\nu G^{C,\mu}_\rho`.
"""

# Auxiliary operators for intermediate calculations

O5aux = flavor_tensor_op("O5aux")
O5auxc = flavor_tensor_op("O5auxc")

Olqqqaux = flavor_tensor_op("Olqqqaux")
Olqqqauxc = flavor_tensor_op("Olqqqauxc")


rules_basis_defs_dim_6_5 = [
    
    # Standard Model dimension 6 four-fermion operators

    # LLLL type
    
    (Op(lLc(0, 1, -1), sigma4bar(2, 0, 3), lL(3, 1, -2),
        lLc(4, 5, -3), sigma4bar(2, 4, 6), lL(6, 5, -4)),
     OpSum(number_op(2) * O1ll(-1, -2, -3, -4))),

    (Op(qLc(0, 1, 2, -1), sigma4bar(3, 0, 4), qL(4, 1, 2, -2),
        qLc(5, 6, 7, -3), sigma4bar(3, 5, 8), qL(8, 6, 7, -4)),
     OpSum(number_op(2) * O1qq(-1, -2, -3, -4))),

    (Op(qLc(0, 1, 2, -1), sigma4bar(3, 0, 4),
        TSU3(5, 1, 6), qL(4, 6, 2, -2),
        qLc(7, 8, 9, -3), sigma4bar(3, 7, 10),
        TSU3(5, 8, 11), qL(10, 11, 9, -4)),
     OpSum(number_op(2) * O8qq(-1, -2, -3, -4))),

    (Op(lLc(0, 1, -1), sigma4bar(2, 0, 3), lL(3, 1, -2),
        qLc(4, 5, 6, -3), sigma4bar(2, 4, 7), qL(7, 5, 6, -4)),
     OpSum(O1lq(-1, -2, -3, -4))),

    (Op(lLc(0, 1, -1), sigma4bar(2, 0, 3),
        sigmaSU2(4, 1, 5), lL(3, 5, -2),
        qLc(6, 7, 8, -3), sigma4bar(2, 6, 9),
        sigmaSU2(4, 8, 10), qL(9, 7, 10, -4)),
     OpSum(O3lq(-1, -2, -3, -4))),

    # RRRR type

    (Op(eRc(0, -1), sigma4(1, 0, 2), eR(2, -2),
        eRc(3, -3), sigma4(1, 3, 4), eR(4, -4)),
     OpSum(number_op(2) * Oee(-1, -2, -3, -4))),

    (Op(uRc(0, 1, -1), sigma4(2, 0, 3), uR(3, 1, -2),
        uRc(4, 5, -3), sigma4(2, 4, 6), uR(6, 5, -4)),
     OpSum(number_op(2) * O1uu(-1, -2, -3, -4))),

    (Op(dRc(0, 1, -1), sigma4(2, 0, 3), dR(3, 1, -2),
        dRc(4, 5, -3), sigma4(2, 4, 6), dR(6, 5, -4)),
     OpSum(number_op(2) * O1dd(-1, -2, -3, -4))),

    (Op(uRc(0, 1, -1), sigma4(2, 0, 3), uR(3, 1, -2),
        dRc(4, 5, -3), sigma4(2, 4, 6), dR(6, 5, -4)),
     OpSum(O1ud(-1, -2, -3, -4))),

    (Op(uRc(0, 1, -1), sigma4(2, 0, 3),
        TSU3(4, 1, 5), uR(3, 5, -2),
        dRc(6, 7, -3), sigma4(2, 6, 8),
        TSU3(4, 7, 9), dR(8, 9, -4)),
     OpSum(O8ud(-1, -2, -3, -4))), 
    
    (Op(eRc(0, -1), sigma4(2, 0, 3), eR(3, -2),
        uRc(4, 5, -3), sigma4(2, 4, 6), uR(6, 5, -4)),
     OpSum(Oeu(-1, -2, -3, -4))),
    
    (Op(eRc(0, -1), sigma4(2, 0, 3), eR(3, -2),
        dRc(4, 5, -3), sigma4(2, 4, 6), dR(6, 5, -4)),
     OpSum(Oed(-1, -2, -3, -4))),

    # LLRR and LRRL type

    (Op(lLc(0, 1, -1), sigma4bar(2, 0, 3), lL(3, 1, -2),
        eRc(4, -3), sigma4(2, 4, 5), eR(5, -4)),
     OpSum(Ole(-1, -2, -3, -4))),

    (Op(qLc(0, 1, 2, -1), sigma4bar(3, 0, 4), qL(4, 1, 2, -2),
        eRc(5, -3), sigma4(3, 5, 6), eR(6, -4)),
     OpSum(Oqe(-1, -2, -3, -4))),

    (Op(lLc(0, 1, -1), sigma4bar(2, 0, 3), lL(3, 1, -2),
        uRc(4, 5, -3), sigma4(2, 4, 6), uR(6, 5, -4)),
     OpSum(Olu(-1, -2, -3, -4))),

    (Op(lLc(0, 1, -1), sigma4bar(2, 0, 3), lL(3, 1, -2),
        dRc(4, 5, -3), sigma4(2, 4, 6), dR(6, 5, -4)),
     OpSum(Old(-1, -2, -3, -4))),

    (Op(qLc(0, 1, 2, -1), sigma4bar(3, 0, 4), qL(4, 1, 2, -2),
        uRc(5, 6, -3), sigma4(3, 5, 7), uR(7, 6, -4)),
     OpSum(O1qu(-1, -2, -3, -4))),

    (Op(qLc(0, 1, 2, -1), sigma4bar(3, 0, 4),
        TSU3(5, 1, 6), qL(4, 6, 2, -2),
        uRc(7, 8, -3), sigma4(3, 7, 9),
        TSU3(5, 8, 10), uR(9, 10, -4)),
     OpSum(O8qu(-1, -2, -3, -4))),
    
    (Op(qLc(0, 1, 2, -1), sigma4bar(3, 0, 4), qL(4, 1, 2, -2),
        dRc(5, 6, -3), sigma4(3, 5, 7), dR(7, 6, -4)),
     OpSum(O1qd(-1, -2, -3, -4))),

    (Op(qLc(0, 1, 2, -1), sigma4bar(3, 0, 4),
        TSU3(5, 1, 6), qL(4, 6, 2, -2),
        dRc(7, 8, -3), sigma4(3, 7, 9),
        TSU3(5, 8, 10), dR(9, 10, -4)),
     OpSum(O8qd(-1, -2, -3, -4))),

    (Op(lLc(0, 1, -1), eR(0, -2), dRc(2, 3, -3), qL(2, 3, 1, -4)),
     OpSum(Oledq(-1, -2, -3, -4))),

    (Op(eRc(0, -2), lL(0, 1, -1), qLc(2, 3, 1, -4), dR(2, 3, -3)),
     OpSum(Oledqc(-1, -2, -3, -4))),

    # LRLR type

    (Op(qLc(0, 1, 2, -1), uR(0, 1, -2), epsSU2(2, 3),
        qLc(4, 5, 3, -3), dR(4, 5, -4)),
     OpSum(O1qud(-1, -2, -3, -4))),

    (Op(uRc(0, 1, -2), qL(0, 1, 2, -1), epsSU2(2, 3),
        dRc(4, 5, -4), qL(4, 5, 3, -3)),
     OpSum(O1qudc(-1, -2, -3, -4))),

    (Op(qLc(0, 1, 2, -1), TSU3(3, 1, 4), uR(0, 4, -2),
        epsSU2(2, 5),
        qLc(6, 7, 5, -3), TSU3(3, 7, 8), dR(0, 8, -4)),
     OpSum(O8qud(-1, -2, -3, -4))),

    (Op(uRc(0, 4, -2), TSU3(3, 4, 1), qLc(0, 1, 2, -1),
        epsSU2(2, 5),
        dRc(0, 8, -4), TSU3(3, 8, 7), qL(6, 7, 5, -3)),
     OpSum(O8qudc(-1, -2, -3, -4))),
    
    (Op(lLc(0, 1, -1), eR(0, -2), epsSU2(1, 2),
        qLc(3, 4, 2, -3), uR(3, 4, -4)),
     OpSum(Olequ(-1, -2, -3, -4))),

    (Op(eRc(0, -2), lL(0, 1, -1), epsSU2(1, 2),
        uRc(3, 4, -4), qL(3, 4, 2, -3)),
     OpSum(Olequc(-1, -2, -3, -4))),

    (Op(lLc(0, 1, -1), uR(0, 2, -2), epsSU2(1, 3),
        qLc(4, 2, 3, -3), eR(4, -4)),
     OpSum(Oluqe(-1, -2, -3, -4))),

    (Op(uRc(0, 2, -2), lL(0, 1, -1), epsSU2(1, 3),
        eRc(4, -4), qLc(4, 2, 3, -3)),
     OpSum(Oluqec(-1, -2, -3, -4))),

    # \slashed{B} and \slashed{L} type

    (Op(epsSU3(0, 1, 2), lLc(3, 4, -1), epsSU2(4, 5),
        epsUpDot(3, 6), qLc(6, 0, 5, -2),
        dRc(7, 1, -3), epsDown(7, 8), uRc(8, 2, -4)),
     OpSum(Olqdu(-1, -2, -3, -4))),

    (Op(epsSU3(0, 1, 2), qL(6, 0, 5, -2), epsSU2(5, 4),
        epsUp(3, 6), lL(3, 4, -1),
        uR(8, 2, -4), epsDownDot(8, 7), dR(7, 1, -3)),
     OpSum(Olqduc(-1, -2, -3, -4))),

    (Op(epsSU3(0, 1, 2), qLc(3, 1, 4, -1), epsSU2(4, 5),
        epsUpDot(3, 6), qLc(6, 2, 5, -2),
        eRc(7, -3), epsDown(7, 8), uRc(8, 0, -4)),
     OpSum(Oqqeu(-1, -2, -3, -4))),

    (Op(epsSU3(0, 1, 2), qL(6, 1, 5, -2), epsSU2(4, 5),
        epsUp(6, 3), qL(3, 0, 4, -1),
        uR(8, 2, -4), epsDownDot(8, 7), eR(7, -3)),
     OpSum(Oqqeuc(-1, -2, -3, -4))),

    (Op(epsSU3(0, 1, 2), lLc(3, 4, -1), epsSU2(4, 5),
        epsUpDot(3, 6), qLc(6, 0, 5, -2),
        qLc(7, 1, 8, -3), epsSU2(8, 9),
        epsUpDot(7, 10), qLc(10, 2, 9, -4)),
     OpSum(O1lqqq(-1, -2, -3, -4))),

    (Op(epsSU3(0, 1, 2), qL(6, 0, 5, -2), epsSU2(4, 5),
        epsUp(6, 4), lL(3, 4, -1),
        qL(10, 2, 9, -4), epsSU2(8, 9),
        epsUp(10, 7), qL(7, 1, 8, -3)),
     OpSum(O1lqqqc(-1, -2, -3, -4))),

    (Op(epsSU3(0, 1, 2), uRc(3, 0, -1), epsDown(3, 4),
        dRc(4, 1, -2), eRc(5, -3), epsDown(5, 6),
        uRc(6, 2, -4)),
     OpSum(Oudeu(-1, -2, -3, -4))),

    (Op(epsSU3(0, 1, 2), dR(4, 1, -2), epsDownDot(4, 3),
        uR(3, 0, -1), uR(6, 2, -4), epsDownDot(6, 5),
        eR(5, -3)),
     OpSum(Oudeuc(-1, -2, -3, -4))),

    (Op(epsSU3(0, 1, 2), lLc(3, 4, -1), sigmaSU2(5, 4, 6),
        epsSU2(6, 7), epsUpDot(3, 8), qLc(8, 0, 7, -2),
        qLc(9, 1, 10, -3), sigmaSU2(5, 10, 11),
        epsSU2(11, 12), epsUpDot(9, 13), qLc(13, 2, 12, -4)),
     OpSum(O3lqqq(-1, -2, -3, -4))),

    (Op(epsSU3(0, 1, 2), epsSU2(6, 7), qL(8, 0, 7, -2),
        sigmaSU2(5, 6, 4), epsUp(8, 3), lL(3, 4, -1),
        epsSU2(11, 12), qL(13, 2, 12, -4),
        sigmaSU2(5, 11, 10), epsUp(13, 9), qL(9, 1, 10, -3)),
     OpSum(O3lqqqc(-1, -2, -3, -4))),
    
    # Standard Model dimension 6 operators other than four-fermion

    # S type

    (Op(D(0, phic(1)), D(0, phi(1)), phic(2), phi(2)),
     OpSum(number_op(Fraction(1, 2)) * Ophisq,
           -Op(mu2phi()) * Ophi4,
           number_op(6) * Op(lambdaphi()) * Ophi,
           number_op(Fraction(1, 2)) * Op(ye(0, 1)) * Oephi(0, 1),
           number_op(Fraction(1, 2)) * Op(yd(0, 1)) * Odphi(0, 1),
           number_op(Fraction(1, 2)) * Op(Vc(0, 1), yu(0, 2)) * Ouphi(1, 2),
           number_op(Fraction(1, 2)) * Op(yec(0, 1)) * Oephic(0, 1),
           number_op(Fraction(1, 2)) * Op(ydc(0, 1)) * Odphic(0, 1),
           number_op(Fraction(1, 2)) * Op(V(0, 1), yuc(0, 2)) * Ouphic(1, 2))),
           
    
    (Op(phic(0), phi(0), phic(1), phi(1), phic(2), phi(2)),
     OpSum(number_op(3) * Ophi)),

    (Op(phic(0), phi(0), phic(1), phic(1), phi(2), phi(2)),
     OpSum(number_op(3) * Ophi)),

    # SVF type

    (Op(phic(0), D(1, phi(0)),
        lLc(2, 3, -1), sigma4bar(1, 2, 4), lL(4, 3, -2)),
     OpSum(- i_op * O1phil(-1, -2))),

    (Op(D(1, phic(0)), phi(0),
        lLc(2, 3, -2), sigma4bar(1, 2, 4), lL(4, 3, -1)),
     OpSum(i_op * O1philc(-1, -2))),

    (Op(phic(0), sigmaSU2(1, 0, 2), D(3, phi(2)),
        lLc(4, 5, -1), sigma4bar(3, 4, 6),
        sigmaSU2(1, 5, 7), lL(6, 7, -2)),
     OpSum(- i_op * O3phil(-1, -2))),

    (Op(D(3, phic(0)), sigmaSU2(1, 0, 2), phi(2),
        lLc(4, 5, -2), sigma4bar(3, 4, 6),
        sigmaSU2(1, 5, 7), lL(6, 7, -1)),
     OpSum(i_op * O3philc(-1, -2))),
    
    (Op(phic(0), D(1, phi(0)),
        qLc(2, 3, 4, -1), sigma4bar(1, 2, 5), qL(5, 3, 4, -2)),
     OpSum(- i_op * O1phiq(-1, -2))),

    (Op(D(1, phic(0)), phi(0),
        qLc(2, 3, 4, -2), sigma4bar(1, 2, 5), qL(5, 3, 4, -1)),
     OpSum(i_op * O1phiqc(-1, -2))),

    (Op(phic(0), sigmaSU2(1, 0, 2), D(3, phi(2)),
        qLc(4, 5, 6, -1), sigma4bar(3, 4, 7),
        sigmaSU2(1, 6, 8), qL(7, 5, 8, -2)),
     OpSum(- i_op * O3phiq(-1, -2))),

    (Op(D(3, phic(0)), sigmaSU2(1, 0, 2), phi(2),
        qLc(4, 5, 6, -2), sigma4bar(3, 4, 7),
        sigmaSU2(1, 6, 8), qL(7, 5, 8, -1)),
     OpSum(i_op * O3phiqc(-1, -2))),

    (Op(phic(0), D(1, phi(0)),
        eRc(2, -1), sigma4(1, 2, 3), eR(3, -2)),
     OpSum(- i_op * O1phie(-1, -2))),

    (Op(D(1, phic(0)), phi(0),
        eRc(2, -2), sigma4(1, 2, 3), eR(3, -1)),
     OpSum(i_op * O1phiec(-1, -2))),

    (Op(phic(0), D(1, phi(0)),
        dRc(2, 3, -1), sigma4(1, 2, 4), dR(4, 3, -2)),
     OpSum(- i_op * O1phid(-1, -2))),

    (Op(D(1, phic(0)), phi(0),
        dRc(2, 3, -2), sigma4(1, 2, 4), dR(4, 3, -1)),
     OpSum(i_op * O1phidc(-1, -2))),

    (Op(phic(0), D(1, phi(0)),
        uRc(2, 3, -1), sigma4(1, 2, 4), uR(4, 3, -2)),
     OpSum(- i_op * O1phiu(-1, -2))),

    (Op(D(1, phic(0)), phi(0),
        uRc(2, 3, -2), sigma4(1, 2, 4), uR(4, 3, -1)),
     OpSum(i_op * O1phiuc(-1, -2))),
    
    (Op(phi(0), epsSU2(0, 1), D(2, phi(1)),
        uRc(3, 4, -1), sigma4(2, 3, 5), dR(5, 4, -2)),
     OpSum(- i_op * Ophiud(-1, -2))),

    (Op(phic(0), epsSU2(0, 1), D(2, phic(1)),
        dRc(3, 4, -2), sigma4(2, 3, 5), uR(5, 4, -1)),
     OpSum(i_op * Ophiudc(-1, -2))),

    # STF type

    (Op(lLc(0, 1, -1), sigmaTensor(2, 3, 0, 4), eR(4, -2),
        phi(1), bFS(2, 3)),
     OpSum(OeB(-1, -2))),

    (Op(eRc(4, -2), sigmaTensor(2, 3, 4, 0), lL(0, 1, -1),
        phic(1), bFS(2, 3)),
     OpSum(OeBc(-1, -2))),

    (Op(lLc(0, 1, -1), sigmaTensor(2, 3, 0, 4), eR(4, -2),
        sigmaSU2(5, 1, 6), phi(6), wFS(2, 3, 5)),
     OpSum(OeW(-1, -2))),

    (Op(eRc(4, -2), sigmaTensor(2, 3, 4, 0), lL(0, 1, -1),
        sigmaSU2(5, 6, 1), phic(6), wFS(2, 3, 5)),
     OpSum(OeWc(-1, -2))),

    (Op(qLc(0, 1, 2, -1), sigmaTensor(3, 4, 0, 5), uR(5, 1, -2),
        epsSU2(2, 6), phic(6), bFS(3, 4)),
     OpSum(OuB(-1, -2))),

    (Op(uRc(5, 1, -2), sigmaTensor(3, 4, 5, 0), qL(0, 1, 2, -1),
        epsSU2(2, 6), phi(6), bFS(3, 4)),
     OpSum(OuBc(-1, -2))),

    (Op(qLc(0, 1, 2, -1), sigmaTensor(3, 4, 0, 5), uR(5, 1, -2),
        sigmaSU2(6, 2, 7), epsSU2(7, 8), phic(8), wFS(3, 4, 6)),
     OpSum(OuW(-1, -2))),

    (Op(uRc(5, 1, -2), sigmaTensor(3, 4, 5, 0), qL(0, 1, 2, -1),
        sigmaSU2(6, 7, 2), epsSU2(7, 8), phi(8), wFS(3, 4, 6)),
     OpSum(OuWc(-1, -2))),

    (Op(qLc(0, 1, 2, -1), sigmaTensor(3, 4, 0, 5), dR(5, 1, -2),
        phi(2), bFS(3, 4)),
     OpSum(OdB(-1, -2))),

    (Op(dRc(5, 1, -2), sigmaTensor(3, 4, 5, 0), qL(0, 1, 2, -1),
        phic(2), bFS(3, 4)),
     OpSum(OdBc(-1, -2))),

    (Op(qLc(0, 1, 2, -1), sigmaTensor(3, 4, 0, 5), dR(5, 1, -2),
        sigmaSU2(6, 2, 7), phi(7), wFS(3, 4, 6)),
     OpSum(OdW(-1, -2))),

    (Op(dRc(5, 1, -2), sigmaTensor(3, 4, 5, 0), qL(0, 1, 2, -1),
        sigmaSU2(6, 7, 2), phic(7), wFS(3, 4, 6)),
     OpSum(OdWc(-1, -2))),

    (Op(qLc(0, 1, 2, -1), sigmaTensor(3, 4, 0, 5),
        TSU3(6, 1, 7), uR(5, 7, -2),
        epsSU2(2, 8), phic(8), gFS(3, 4, 6)),
     OpSum(OuG(-1, -2))),

    (Op(uRc(5, 7, -2), sigmaTensor(3, 4, 5, 0),
        TSU3(6, 7, 1), qL(0, 1, 2, -1),
        epsSU2(2, 8), phi(8), gFS(3, 4, 6)),
     OpSum(OuGc(-1, -2))),

    (Op(qLc(0, 1, 2, -1), sigmaTensor(3, 4, 0, 5),
        TSU3(6, 1, 7), dR(5, 7, -2),
        phi(2), gFS(3, 4, 6)),
     OpSum(OdG(-1, -2))),

    (Op(dRc(5, 1, -2), sigmaTensor(3, 4, 5, 0),
        TSU3(6, 1, 7), qL(0, 7, 2, -1),
        phic(2), gFS(3, 4, 6)),
     OpSum(OdGc(-1, -2))),
     
    # SF type

    (Op(phic(0), phi(0), lLc(1, 2, -1), phi(2), eR(1, -2)),
     OpSum(Oephi(-1, -2))),

    (Op(phic(0), phi(0), eRc(1, -2), phic(2), lL(1, 2, -1)),
     OpSum(Oephic(-1, -2))),

    (Op(phic(0), phi(0), qLc(1, 2, 3, -1), phi(3), dR(1, 2, -2)),
     OpSum(Odphi(-1, -2))),

    (Op(phic(0), phi(0), dRc(1, 2, -2), phic(3), qL(1, 2, 3, -1)),
     OpSum(Odphic(-1, -2))),

    (Op(phic(0), phi(0), qLc(1, 2, 3, -1), epsSU2(3, 4),
        phic(4), uR(1, 2, -2)),
     OpSum(Ouphi(-1, -2))),

    (Op(phic(0), phi(0), qLc(1, 2, 3, -1), epsSU2(4, 3),
        phic(4), uR(1, 2, -2)),
     OpSum(-Ouphi(-1, -2))),

    (Op(phic(0), phi(0), uRc(1, 2, -2), qL(1, 2, 3, -1),
        epsSU2(3, 4), phi(4)),
     OpSum(Ouphic(-1, -2))),

    (Op(phic(0), phi(0), uRc(1, 2, -2), qL(1, 2, 3, -1),
        epsSU2(4, 3), phi(4)),
     OpSum(-Ouphic(-1, -2))),

    # Oblique type

    (Op(phic(0), D(1, phi(0)), D(1, phic(2)), phi(2)),
     OpSum(OphiD)),

    (Op(phic(0), phi(0), bFS(1, 2), bFS(1, 2)),
     OpSum(OphiB)),

    (Op(phic(0), phi(0), eps4(1, 2, 3, 4), bFS(3, 4), bFS(1, 2)),
     OpSum(OphiBTilde)),

    (Op(phic(0), sigmaSU2(1, 0, 2), phi(2), wFS(3, 4, 1), bFS(3, 4)),
     OpSum(OWB)),

    (Op(phic(0), sigmaSU2(1, 0, 2), phi(2),
        eps4(3, 4, 5, 6), wFS(5, 6, 1), bFS(3, 4)),
     OpSum(OWBTilde)),

    (Op(phic(0), phi(0), wFS(1, 2, 3), wFS(1, 2, 3)),
     OpSum(OphiW)),

    (Op(phic(0), phi(0), eps4(1, 2, 4, 5), wFS(4, 5, 3), wFS(1, 2, 3)),
     OpSum(OphiWTilde)),

    (Op(phic(0), phi(0), gFS(1, 2, 3), gFS(1, 2, 3)),
     OpSum(OphiG)),
    
    (Op(phic(0), phi(0), eps4(1, 2, 4, 5), gFS(4, 5, 3), gFS(1, 2, 3)),
     OpSum(OphiGTilde)),

    # Gauge type

    (Op(epsSU2triplets(0, 1, 2),
        wFS(3, 4, 0), wFS(4, 5, 1), wFS(5, 3, 2)),
     OpSum(OW)),

    (Op(epsSU2triplets(0, 1, 2),
        eps4(3, 4, 6, 7), wFS(6, 7, 0), wFS(4, 5, 1), wFS(5, 3, 2)),
     OpSum(OWTilde)),

    (Op(fSU3(0, 1, 2),
        gFS(3, 4, 0), gFS(4, 5, 1), gFS(5, 3, 2)),
     OpSum(OG)),

    (Op(fSU3(0, 1, 2),
        eps4(3, 4, 6, 7), gFS(6, 7, 0), gFS(4, 5, 1), gFS(5, 3, 2)),
     OpSum(OGTilde)),

    # Standard Model dimension 5 operators
    
    (Op(lL(0, 1, -1), epsSU2(1, 2), phi(2),
        epsSU2(3, 4), phi(4), epsUp(5, 0), lL(5, 3, -2)),
     OpSum(O5(-1, -2))),

    (Op(lLc(0, 1, -2), epsSU2(1, 2), phic(2), epsUpDot(0, 3),
        epsSU2(4, 5), phic(5), lLc(3, 4, -1)),
     OpSum(O5c(-1, -2)))]

rules_basis_defs_dim_4 = [
    # Standard Model dimension 4 operators
    
    (Op(D(0, phic(1)), D(0, phi(1))),
     OpSum(Okinphi)),

    (Op(phic(0), phi(0), phic(1), phi(1)),
     OpSum(Ophi4)),

    (Op(phic(0), phi(0)),
     OpSum(Ophi2)),

    (Op(lLc(0, 1, -1), phi(1), eR(0, -2)),
     OpSum(Oye(-1, -2))),

    (Op(eRc(0, -2), phic(1), lL(0, 1, -1)),
     OpSum(Oyec(-1, -2))),

    (Op(qLc(0, 1, 2, -1), epsSU2(2, 3), phic(3), uR(0, 1, -2)),
     OpSum(Oyu(-1, -2))),

    (Op(uRc(0, 1, -2), epsSU2(2, 3), phi(3), qL(0, 1, 2, -1)),
     OpSum(Oyuc(-1, -2))),

    (Op(qLc(0, 1, 2, -1), phi(2), dR(0, 1, -2)),
     OpSum(Oyd(-1, -2))),

    (Op(dRc(0, 1, -2), phic(2), qL(0, 1, 2, -1)),
     OpSum(Oydc(-1, -2)))]

rules_basis_definitions = rules_basis_defs_dim_6_5 + rules_basis_defs_dim_4
"""
Rules defining the operators in the basis in terms of 
Standard Model fields.
"""


latex_basis_coefs = {
    # Dimension 4
    "Okinphi": r"\alpha_{{kin,\phi}}", 
    "Ophi4": r"\alpha_{{\phi 4}}",
    "Ophi2": r"\alpha_{{\phi 2}}",
    "Oye": r"\left(\alpha_{{{{y^e}}}}\right)_{{{}{}}}",
    "Oyec": r"\left(\alpha_{{{{y^e}}}}\right)^*_{{{}{}}}",
    "Oyd": r"\left(\alpha_{{{{y^d}}}}\right)_{{{}{}}}",
    "Oydc": r"\left(\alpha_{{{{y^d}}}}\right)^*_{{{}{}}}",
    "Oyu": r"\left(\alpha_{{{{y^u}}}}\right)_{{{}{}}}",
    "Oyuc": r"\left(\alpha_{{{{y^u}}}}\right)^*_{{{}{}}}",

    # Dimension 5
    "O5": r"\frac{{\left(\alpha_5\right)_{{{}{}}}}}{{\Lambda}}",
    "O5c": r"\frac{{\left(\alpha_5\right)^*_{{{}{}}}}}{{\Lambda}}",

    # Auxiliary
    
    "O5aux": r"\frac{{\left(\alpha^{{AUX}}_5\right)_{{{}{}}}}}{{\Lambda}}",
    "O5auxc": r"\frac{{\left(\alpha^{{AUX}}_5\right)^*_{{{}{}}}}}{{\Lambda}}",

    "Olqqqaux":
    r"\frac{{\left(\alpha^{{AUX}}_{{lqqq}}\right)_{{{}{}{}{}}}}}{{\Lambda^2}}",
    "Olqqqauxc":
    r"\frac{{\left(\alpha^{{AUX}}_{{lqqq}}\right)^*_{{{}{}{}{}}}}}{{\Lambda^2}}",

    # Dimension 6 four-fermion

    # LLLL
    
    "O1ll":
    (r"\frac{{\left(\alpha^{{(1)}}_{{ll}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),
    
    "O1qq":
    (r"\frac{{\left(\alpha^{{(1)}}_{{qq}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),
    
    "O8qq":
    (r"\frac{{\left(\alpha^{{(8)}}_{{qq}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "O1lq":
    (r"\frac{{\left(\alpha^{{(1)}}_{{lq}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "O3lq":
    (r"\frac{{\left(\alpha^{{(3)}}_{{lq}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    # RRRR
    
    "Oee":
    (r"\frac{{\left(\alpha_{{ee}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "O1uu":
    (r"\frac{{\left(\alpha^{{(1)}}_{{uu}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "O1dd":
    (r"\frac{{\left(\alpha^{{(1)}}_{{dd}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "O1ud":
    (r"\frac{{\left(\alpha^{{(1)}}_{{ud}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "O8ud":
    (r"\frac{{\left(\alpha^{{(8)}}_{{ud}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "Oeu":
    (r"\frac{{\left(\alpha_{{eu}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "Oed":
    (r"\frac{{\left(\alpha_{{ed}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),
    
    # LLRR and LRRL
    
    "Ole":
    (r"\frac{{\left(\alpha_{{le}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "Oqe":
    (r"\frac{{\left(\alpha_{{qe}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "Olu":
    (r"\frac{{\left(\alpha_{{lu}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "Old":
    (r"\frac{{\left(\alpha_{{ld}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "O1qu":
    (r"\frac{{\left(\alpha^{{(1)}}_{{qu}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "O8qu":
    (r"\frac{{\left(\alpha^{{(8)}}_{{qu}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "O1qd":
    (r"\frac{{\left(\alpha^{{(1)}}_{{qd}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "O8qd":
    (r"\frac{{\left(\alpha^{{(8)}}_{{qd}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),
    
    "Oledq":
    (r"\frac{{\left(\alpha_{{ledq}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "Oledqc":
    (r"\frac{{\left(\alpha_{{ledq}}\right)^*"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    # LRLR

    "O1qud":
    (r"\frac{{\left(\alpha^{{(1)}}_{{qud}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "O1qudc":
    (r"\frac{{\left(\alpha^{{(1)}}_{{qud}}\right)^*"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "O8qud":
    (r"\frac{{\left(\alpha^{{(8)}}_{{qud}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "O8qudc":
    (r"\frac{{\left(\alpha^{{(8)}}_{{qud}}\right)^*"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "Olequ":
    (r"\frac{{\left(\alpha_{{lequ}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "Olequc":
    (r"\frac{{\left(\alpha_{{lequ}}\right)^*"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "Oluqe":
    (r"\frac{{\left(\alpha_{{luqe}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "Oluqec":
    (r"\frac{{\left(\alpha_{{luqe}}\right)^*"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    # \slashed{B} and \slashed{L} type
    
    "Olqdu":
    (r"\frac{{\left(\alpha_{{lqdu}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "Olqduc":
    (r"\frac{{\left(\alpha_{{lqdu}}\right)^*"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "Oqqeu":
    (r"\frac{{\left(\alpha_{{qqeu}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "Oqqeuc":
    (r"\frac{{\left(\alpha_{{qqeu}}\right)^*"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "O1lqqq":
    (r"\frac{{\left(\alpha^{{(1)}}_{{lqqq}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "O1lqqqc":
    (r"\frac{{\left(\alpha^{{(1)}}_{{lqqq}}\right)^*"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "Oudeu":
    (r"\frac{{\left(\alpha_{{udeu}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "Oudeuc":
    (r"\frac{{\left(\alpha_{{udeu}}\right)^*"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "O3lqqq":
    (r"\frac{{\left(\alpha^{{(3)}}_{{lqqq}}\right)"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    "O3lqqqc":
    (r"\frac{{\left(\alpha^{{(3)}}_{{lqqq}}\right)^*"
     r"_{{{}{}{}{}}}}}{{\Lambda^2}}"),

    # Dimesion 6 other than four-fermion

    # S type
    
    "Ophi": r"\frac{{\alpha_\phi}}{{\Lambda^2}}",
    "Ophisq": r"\frac{{\alpha_{{\phi\square}}}}{{\Lambda^2}}",

    # SVF type

    "O1phil":
    (r"\frac{{\left(\alpha^{{(1)}}_{{\phi l}}\right)"
     r"_{{{}{}}}}}{{\Lambda^2}}"),
    
    "O1philc":
    (r"\frac{{\left(\alpha^{{(1)}}_{{\phi l}}\right)^*"
     r"_{{{}{}}}}}{{\Lambda^2}}"),
    
    "O3phil":
    (r"\frac{{\left(\alpha^{{(3)}}_{{\phi l}}\right)"
     r"_{{{}{}}}}}{{\Lambda^2}}"),
    
    "O3philc":
    (r"\frac{{\left(\alpha^{{(3)}}_{{\phi l}}\right)^*"
     r"_{{{}{}}}}}{{\Lambda^2}}"),
    
    "O1phiq":
    (r"\frac{{\left(\alpha^{{(1)}}_{{\phi q}}\right)"
     r"_{{{}{}}}}}{{\Lambda^2}}"),
    
    "O1phiqc":
    (r"\frac{{\left(\alpha^{{(1)}}_{{\phi q}}\right)^*"
     r"_{{{}{}}}}}{{\Lambda^2}}"),
    
    "O3phiq":
    (r"\frac{{\left(\alpha^{{(3)}}_{{\phi q}}\right)"
     r"_{{{}{}}}}}{{\Lambda^2}}"),
    
    "O3phiqc":
    (r"\frac{{\left(\alpha^{{(3)}}_{{\phi q}}\right)^*"
     r"_{{{}{}}}}}{{\Lambda^2}}"),
    
    "O1phie":
    (r"\frac{{\left(\alpha^{{(1)}}_{{\phi e}}\right)"
     r"_{{{}{}}}}}{{\Lambda^2}}"),
    
    "O1phiec":
    (r"\frac{{\left(\alpha^{{(1)}}_{{\phi e}}\right)^*"
     r"_{{{}{}}}}}{{\Lambda^2}}"),
    
    "O1phid":
    (r"\frac{{\left(\alpha^{{(1)}}_{{\phi d}}\right)"
     r"_{{{}{}}}}}{{\Lambda^2}}"),
    
    "O1phidc":
    (r"\frac{{\left(\alpha^{{(1)}}_{{\phi d}}\right)^*"
     r"_{{{}{}}}}}{{\Lambda^2}}"),
    
    "O1phiu":
    (r"\frac{{\left(\alpha^{{(1)}}_{{\phi u}}\right)"
     r"_{{{}{}}}}}{{\Lambda^2}}"),
    
    "O1phiuc":
    (r"\frac{{\left(\alpha^{{(1)}}_{{\phi u}}\right)^*"
     r"_{{{}{}}}}}{{\Lambda^2}}"),
    
    "Ophiud":
    (r"\frac{{\left(\alpha_{{\phi ud}}\right)"
     r"_{{{}{}}}}}{{\Lambda^2}}"),
    
    "Ophiudc":
    (r"\frac{{\left(\alpha_{{\phi ud}}\right)^*"
     r"_{{{}{}}}}}{{\Lambda^2}}"),

    # STF type
    
    "OeB": r"\frac{{\left(\alpha_{{eB}}\right)_{{{}{}}}}}{{\Lambda^2}}",
    "OeBc": r"\frac{{\left(\alpha_{{eB}}\right)^*_{{{}{}}}}}{{\Lambda^2}}",

    "OeW": r"\frac{{\left(\alpha_{{eW}}\right)_{{{}{}}}}}{{\Lambda^2}}",
    "OeWc": r"\frac{{\left(\alpha_{{eW}}\right)^*_{{{}{}}}}}{{\Lambda^2}}",

    "OuB": r"\frac{{\left(\alpha_{{uB}}\right)_{{{}{}}}}}{{\Lambda^2}}",
    "OuBc": r"\frac{{\left(\alpha_{{uB}}\right)^*_{{{}{}}}}}{{\Lambda^2}}",

    "OuW": r"\frac{{\left(\alpha_{{uW}}\right)_{{{}{}}}}}{{\Lambda^2}}",
    "OuWc": r"\frac{{\left(\alpha_{{uW}}\right)^*_{{{}{}}}}}{{\Lambda^2}}",

    "OdB": r"\frac{{\left(\alpha_{{dB}}\right)_{{{}{}}}}}{{\Lambda^2}}",
    "OdBc": r"\frac{{\left(\alpha_{{dB}}\right)^*_{{{}{}}}}}{{\Lambda^2}}",

    "OdW": r"\frac{{\left(\alpha_{{dW}}\right)_{{{}{}}}}}{{\Lambda^2}}",
    "OdWc": r"\frac{{\left(\alpha_{{dW}}\right)^*_{{{}{}}}}}{{\Lambda^2}}",

    "OuG": r"\frac{{\left(\alpha_{{uG}}\right)_{{{}{}}}}}{{\Lambda^2}}",
    "OuGc": r"\frac{{\left(\alpha_{{uG}}\right)^*_{{{}{}}}}}{{\Lambda^2}}",

    "OdG": r"\frac{{\left(\alpha_{{dG}}\right)_{{{}{}}}}}{{\Lambda^2}}",
    "OdGc": r"\frac{{\left(\alpha_{{dG}}\right)^*_{{{}{}}}}}{{\Lambda^2}}",

    # SF type

    "Oephi":
    r"\frac{{\left(\alpha_{{e\phi}}\right)_{{{}{}}}}}{{\Lambda^2}}",

    "Oephic":
    r"\frac{{\left(\alpha_{{e\phi}}\right)^*_{{{}{}}}}}{{\Lambda^2}}",

    "Odphi":
    r"\frac{{\left(\alpha_{{d\phi}}\right)_{{{}{}}}}}{{\Lambda^2}}",

    "Odphic":
    r"\frac{{\left(\alpha_{{d\phi}}\right)^*_{{{}{}}}}}{{\Lambda^2}}",

    "Ouphi":
    r"\frac{{\left(\alpha_{{u\phi}}\right)_{{{}{}}}}}{{\Lambda^2}}",

    "Ouphic":
    r"\frac{{\left(\alpha_{{u\phi}}\right)^*_{{{}{}}}}}{{\Lambda^2}}",

    # Oblique type
    
    "OphiD": r"\frac{{\alpha_{{\phi D}}}}{{\Lambda^2}}",
    "OphiB": r"\frac{{\alpha_{{\phi B}}}}{{\Lambda^2}}",
    "OWB": r"\frac{{\alpha_{{WB}}}}{{\Lambda^2}}",
    "OphiW": r"\frac{{\alpha_{{\phi W}}}}{{\Lambda^2}}",
    "OphiBTilde": r"\frac{{\alpha_{{\phi\tilde{{B}}}}}}{{\Lambda^2}}",
    "OWBTilde": r"\frac{{\alpha_{{W\tilde{{B}}}}}}{{\Lambda^2}}",
    "OphiWTilde": r"\frac{{\alpha_{{\phi\tilde{{W}}}}}}{{\Lambda^2}}",
    "OphiG": r"\frac{{\alpha_{{\phi G}}}}{{\Lambda^2}}",
    "OphiGTilde": r"\frac{{\alpha_{{\phi\tilde{{G}}}}}}{{\Lambda^2}}",

    # Gauge type

    "OW": r"\frac{{\alpha_W}}{{\Lambda^2}}",
    "OWTilde": r"\frac{{\alpha_{{\tilde{{W}}}}}}{{\Lambda^2}}",
    "OG": r"\frac{{\alpha_G}}{{\Lambda^2}}",
    "OGTilde": r"\frac{{\alpha_{{\tilde{{G}}}}}}{{\Lambda^2}}"}
"""
LaTeX representation of the coefficients of the
operators in the basis.
"""
