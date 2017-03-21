import sys

import context

from efttools.operators import (
    Tensor, Op, OpSum,
    TensorBuilder, FieldBuilder, D,
    number_op, symbol_op, tensor_op, flavor_tensor_op,
    boson, fermion, kdelta,
    sigma4, sigma4bar, epsUp, epsUpDot, epsDown, epsDownDot)

from efttools.transformations import (
    collect_numbers_and_symbols, collect_by_tensors,
    apply_rules_until, group_op_sum)

from efttools.integration import (
    integrate, RealScalar, ComplexScalar, RealVector,
    ComplexVector, VectorLikeFermion, MajoranaFermion)

from efttools.output import Writer

# -- Flavor tensors --

deltaFlavor = TensorBuilder("deltaFlavor")

# Standard Model
gb = TensorBuilder("gb")
gw = TensorBuilder("gw")
mu2phi = TensorBuilder("mu2phi")
lambdaphi = TensorBuilder("lambdaphi")
ye = TensorBuilder("ye")
yec = TensorBuilder("yec")
yd = TensorBuilder("yd")
ydc = TensorBuilder("ydc")
yu = TensorBuilder("yu")
yuc = TensorBuilder("yuc")
V = TensorBuilder("V")
Vc = TensorBuilder("Vc")

# Quarks
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

# Leptons
lambdaDelta1e = TensorBuilder("lambdaDelta1e")
lambdaDelta1ec = TensorBuilder("lambdaDelta1ec")
lambdaDelta3e = TensorBuilder("lambdaDelta3e")
lambdaDelta3ec = TensorBuilder("lambdaDelta3ec")
lambdaNRl = TensorBuilder("lambdaNRl")
lambdaNRlc = TensorBuilder("lambdaNRlc")
lambdaNLl = TensorBuilder("lambdaNLl")
lambdaNLlc = TensorBuilder("lambdaNLlc")
lambdaEl = TensorBuilder("lambdaEl")
lambdaElc = TensorBuilder("lambdaElc")
lambdaSigma0Rl = TensorBuilder("lambdaSigma0Rl")
lambdaSigma0Rlc = TensorBuilder("lambdaSigma0Rlc")
lambdaSigma0Ll = TensorBuilder("lambdaSigma0Ll")
lambdaSigma0Llc = TensorBuilder("lambdaSigma0Llc")
lambdaSigma1l = TensorBuilder("lambdaSigma1l")
lambdaSigma1lc = TensorBuilder("lambdaSigma1lc")
lambdaNmajl = TensorBuilder("lambdaNmajl")
lambdaNmajlc = TensorBuilder("lambdaNmajlc")
lambdaSigma0majl = TensorBuilder("lambdaSigma0majl")
lambdaSigma0majlc = TensorBuilder("lambdaSigma0majlc")

# Scalars
kappaS = TensorBuilder("kappaS")
lambdaS = TensorBuilder("lambdaS")
kappaS3 = TensorBuilder("kappaS3")
kappaXi0 = TensorBuilder("kappaXi0")
lambdaXi0 = TensorBuilder("kappaXi0")
ylS1 = TensorBuilder("ylS1")
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
ylXi1 = TensorBuilder("ylXi1")
ylXi1c = TensorBuilder("ylXi1c")
kappaXi1 = TensorBuilder("kappaXi1")
kappaXi1c = TensorBuilder("kappaXi1c")
lambdaXi1 = TensorBuilder("lambdaXi1")
lambdaXi1c = TensorBuilder("lambdaXi1c")
lambdaTildeXi1 = TensorBuilder("lambdaTildeXi1")
lambdaTildeXi1c = TensorBuilder("lambdaTildeXi1c")
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
ydOmega4 = TensorBuilder("ydOmega4")
ydOmega4c = TensorBuilder("ydOmega4c")
yqUpsilon = TensorBuilder("yqUpsilon")
yqUpsilonc = TensorBuilder("yqUpsilonc")
yquPhi = TensorBuilder("yquPhi")
yquPhic = TensorBuilder("yquPhic")
ydqPhi = TensorBuilder("ydqPhi")
ydqPhic = TensorBuilder("ydqPhic")
kappaSXi0 = TensorBuilder("kappaSXi0")
kappaSXi1 = TensorBuilder("kappaSXi1")
kappaXi0Xi1 = TensorBuilder("kappaXi0Xi1")
lambdaSXi0 = TensorBuilder("lambdaSXi0")
kappaSvarphi = TensorBuilder("kappaSvarphi")
kappaSvarphic = TensorBuilder("kappaSvarphic")
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
lambdaSXi1 = TensorBuilder("lambdaSXi1")
lambdaSXi1c = TensorBuilder("lambdaSXi1c")
lambdaXi1Xi0 = TensorBuilder("lambdaXi1Xi0")
lambdaXi1Xi0c = TensorBuilder("lambdaXi1Xi0c")

# Vectors 1
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

# Vectors 2
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

# -- Group tensors --

epsSU2 = TensorBuilder("epsSU2")
sigmaSU2 = TensorBuilder("sigmaSU2")
fSU2 = TensorBuilder("fSU2")

lambdaColor = TensorBuilder("lambdaColor")

eps4 = TensorBuilder("eps4")

# -- Fields --

# Standard Model
phi = FieldBuilder("phi", 1, boson)
phic = FieldBuilder("phic", 1, boson)
lL = FieldBuilder("lL", 1.5, fermion)
lLc = FieldBuilder("lLc", 1.5, fermion)
qL = FieldBuilder("qL", 1.5, fermion)
qLc = FieldBuilder("qLc", 1.5, fermion)
eR = FieldBuilder("eR", 1.5, fermion)
eRc = FieldBuilder("eRc", 1.5, fermion)
dR = FieldBuilder("dR", 1.5, fermion)
dRc = FieldBuilder("dRc", 1.5, fermion)
uR = FieldBuilder("uR", 1.5, fermion)
uRc = FieldBuilder("uRc", 1.5, fermion)
bFS = FieldBuilder("bFS", 2, boson)
wFS = FieldBuilder("wFS", 2, boson)

# Quarks
UL = FieldBuilder("UL", 1.5, fermion)
UR = FieldBuilder("UR", 1.5, fermion)
ULc = FieldBuilder("ULc", 1.5, fermion)
URc = FieldBuilder("URc", 1.5, fermion)
DL = FieldBuilder("DL", 1.5, fermion)
DR = FieldBuilder("DR", 1.5, fermion)
DLc = FieldBuilder("DLc", 1.5, fermion)
DRc = FieldBuilder("DRc", 1.5, fermion)
UDL = FieldBuilder("UDL", 1.5, fermion)
UDR = FieldBuilder("UDR", 1.5, fermion)
UDLc = FieldBuilder("UDLc", 1.5, fermion)
UDRc = FieldBuilder("UDRc", 1.5, fermion)
XUL = FieldBuilder("XUL", 1.5, fermion)
XUR = FieldBuilder("XUR", 1.5, fermion)
XULc = FieldBuilder("XULc", 1.5, fermion)
XURc = FieldBuilder("XURc", 1.5, fermion)
DYL = FieldBuilder("DYL", 1.5, fermion)
DYR = FieldBuilder("DYR", 1.5, fermion)
DYLc = FieldBuilder("DYLc", 1.5, fermion)
DYRc = FieldBuilder("DYRc", 1.5, fermion)
UDL = FieldBuilder("UDL", 1.5, fermion)
UDR = FieldBuilder("UDR", 1.5, fermion)
UDLc = FieldBuilder("UDLc", 1.5, fermion)
UDRc = FieldBuilder("UDRc", 1.5, fermion)
XUDL = FieldBuilder("XUDL", 1.5, fermion)
XUDR = FieldBuilder("XUDR", 1.5, fermion)
XUDLc = FieldBuilder("XUDLc", 1.5, fermion)
XUDRc = FieldBuilder("XUDRc", 1.5, fermion)
UDYL = FieldBuilder("UDYL", 1.5, fermion)
UDYR = FieldBuilder("UDYR", 1.5, fermion)
UDYLc = FieldBuilder("UDYLc", 1.5, fermion)
UDYRc = FieldBuilder("UDYRc", 1.5, fermion)

# Leptons
Delta1L = FieldBuilder("Delta1L", 1.5, fermion)
Delta1R = FieldBuilder("Delta1R", 1.5, fermion)
Delta1Lc = FieldBuilder("Delta1Lc", 1.5, fermion)
Delta1Rc = FieldBuilder("Delta1Rc", 1.5, fermion)
Delta3L = FieldBuilder("Delta3L", 1.5, fermion)
Delta3R = FieldBuilder("Delta3R", 1.5, fermion)
Delta3Lc = FieldBuilder("Delta3Lc", 1.5, fermion)
Delta3Rc = FieldBuilder("Delta3Rc", 1.5, fermion)
NL = FieldBuilder("NL", 1.5, fermion)
NR = FieldBuilder("NR", 1.5, fermion)
NLc = FieldBuilder("NLc", 1.5, fermion)
NRc = FieldBuilder("NRc", 1.5, fermion)
Nmaj = FieldBuilder("Nmaj", 1.5, fermion)
Nmajc = FieldBuilder("Nmajc", 1.5, fermion)
EL = FieldBuilder("EL", 1.5, fermion)
ER = FieldBuilder("ER", 1.5, fermion)
ELc = FieldBuilder("ELc", 1.5, fermion)
ERc = FieldBuilder("ERc", 1.5, fermion)
Sigma0L = FieldBuilder("Sigma0L", 1.5, fermion)
Sigma0R = FieldBuilder("Sigma0R", 1.5, fermion)
Sigma0Lc = FieldBuilder("Sigma0Lc", 1.5, fermion)
Sigma0Rc = FieldBuilder("Sigma0Rc", 1.5, fermion)
Sigma0maj = FieldBuilder("Sigma0maj", 1.5, fermion)
Sigma0majc = FieldBuilder("Sigma0majc", 1.5, fermion)
Sigma1L = FieldBuilder("Sigma1L", 1.5, fermion)
Sigma1R = FieldBuilder("Sigma1R", 1.5, fermion)
Sigma1Lc = FieldBuilder("Sigma1Lc", 1.5, fermion)
Sigma1Rc = FieldBuilder("Sigma1Rc", 1.5, fermion)

# Scalars
S = FieldBuilder("S", 1, boson)
Xi0 = FieldBuilder("Xi0", 1, boson)
S1 = FieldBuilder("S1", 1, boson)
S1c = FieldBuilder("S1c", 1, boson)
S2 = FieldBuilder("S2", 1, boson)
S2c = FieldBuilder("S2c", 1, boson)
varphi = FieldBuilder("varphi", 1, boson)
varphic = FieldBuilder("varphic", 1, boson)
Xi1 = FieldBuilder("Xi1", 1, boson)
Xi1c = FieldBuilder("Xi1c", 1, boson)
Theta1 = FieldBuilder("Theta1", 1, boson)
Theta1c = FieldBuilder("Theta1c", 1, boson)
Theta3 = FieldBuilder("Theta3", 1, boson)
Theta3c = FieldBuilder("Theta3c", 1, boson)
omega1 = FieldBuilder("omega1", 1, boson)
omega1c = FieldBuilder("omega1c", 1, boson)
omega2 = FieldBuilder("omega2", 1, boson)
omega2c = FieldBuilder("omega2c", 1, boson)
omega4 = FieldBuilder("omega4", 1, boson)
omega4c = FieldBuilder("omega4c", 1, boson)
Pi1 = FieldBuilder("Pi1", 1, boson)
Pi1c = FieldBuilder("Pi1c", 1, boson)
Pi7 = FieldBuilder("Pi7", 1, boson)
Pi7c = FieldBuilder("Pi7c", 1, boson)
zeta = FieldBuilder("zeta", 1, boson)
zetac = FieldBuilder("zetac", 1, boson)
Omega1 = FieldBuilder("Omega1", 1, boson)
Omega1c = FieldBuilder("Omega1c", 1, boson)
Omega2 = FieldBuilder("Omega2", 1, boson)
Omega2c = FieldBuilder("Omega2c", 1, boson)
Omega4 = FieldBuilder("Omega4", 1, boson)
Omega4c = FieldBuilder("Omega4c", 1, boson)
Upsilon = FieldBuilder("Upsilon", 1, boson)
Upsilonc = FieldBuilder("Upsilonc", 1, boson)
Phi = FieldBuilder("Phi", 1, boson)
Phic = FieldBuilder("Phic", 1, boson)

# Vectors
B = FieldBuilder("B", 1, boson)
W = FieldBuilder("W", 1, boson)
G = FieldBuilder("G", 1, boson)
H = FieldBuilder("H", 1, boson)
B1 = FieldBuilder("B1", 1, boson)
B1c = FieldBuilder("B1c", 1, boson)
W1 = FieldBuilder("W1", 1, boson)
W1c = FieldBuilder("W1c", 1, boson)
G1 = FieldBuilder("G1", 1, boson)
G1c = FieldBuilder("G1c", 1, boson)
L1 = FieldBuilder("L1", 1, boson)
L1c = FieldBuilder("L1c", 1, boson)
L3 = FieldBuilder("L3", 1, boson)
L3c = FieldBuilder("L3c", 1, boson)
U2 = FieldBuilder("U2", 1, boson)
U2c = FieldBuilder("U2c", 1, boson)
U5 = FieldBuilder("U5", 1, boson)
U5c = FieldBuilder("U5c", 1, boson)
Q1 = FieldBuilder("Q1", 1, boson)
Q1c = FieldBuilder("Q1c", 1, boson)
Q5 = FieldBuilder("Q5", 1, boson)
Q5c = FieldBuilder("Q5c", 1, boson)

# -- Lagrangian --

half = number_op(0.5)

L_quarks = -OpSum(
    Op(lambdaP1(0, 1), V(1, 2), URc(3, 4, 0), epsSU2(5, 6), phi(6), qL(3, 4, 5, 2)),
    Op(lambdaP1c(0, 1), Vc(1, 2), qLc(3, 4, 5, 2), epsSU2(5, 6), phic(6), UR(3, 4, 0)),

    Op(lambdaP2(0, 1), DRc(2, 3, 0), phic(4), qL(2, 3, 4, 1)),
    Op(lambdaP2c(0, 1), qLc(2, 3, 4, 1), phi(4), DR(2, 3, 0)),

    Op(lambdaP3u(0, 1), UDLc(2, 3, 4, 0), epsSU2(4, 5), phic(5), uR(2, 3, 1)),
    Op(lambdaP3uc(0, 1), uRc(2, 3, 1), epsSU2(4, 5), phi(5), UDL(2, 3, 4, 1)),

    Op(lambdaP3d(0, 1), UDLc(2, 3, 4, 0), phi(4), dR(2, 3, 1)),
    Op(lambdaP3dc(0, 1), dRc(2, 3, 1), phic(4), UDL(2, 3, 4, 0)),

    Op(lambdaP4(0, 1), XULc(2, 3, 4, 0), phi(4), uR(2, 3, 1)),
    Op(lambdaP4c(0, 1), uRc(2, 3, 1), phic(4), XUL(2, 3, 4, 0)),

    Op(lambdaP5(0, 1), DYLc(2, 3, 4, 0), epsSU2(4, 5), phic(5), dR(2, 3, 1)),
    Op(lambdaP5c(0, 1), dRc(2, 3, 1), epsSU2(4, 5), phi(5), DYL(2, 3, 4, 0)),

    half * Op(lambdaP6(0, 1), V(1, 2), XUDRc(3, 4, 5, 0),
              epsSU2(6, 7), phi(7), sigmaSU2(5, 6, 8), qL(3, 4, 8, 2)),
    half * Op(lambdaP6c(0, 1), Vc(1, 2), qLc(3, 4, 5, 2),
              sigmaSU2(6, 5, 7), epsSU2(7, 8), phic(8), XUDR(3, 4, 6, 0)),

    half * Op(lambdaP7(0, 1), V(1, 2), UDYRc(3, 4, 5, 0),
              phic(6), sigmaSU2(5, 6, 7), qL(3, 4, 7, 2)),
    half * Op(lambdaP7c(0, 1), Vc(1, 2), qLc(3, 4, 5, 2),
              sigmaSU2(6, 5, 7), phi(7), UDYR(3, 4, 6, 0)))

L_leptons = -OpSum(
    Op(lambdaDelta1e(0, 1), Delta1Lc(2, 3, 0), phi(3), eR(2, 1)),
    Op(lambdaDelta1ec(0, 1), eRc(2, 1), phic(3), Delta1L(2, 3, 0)),

    Op(lambdaDelta3e(0, 1), Delta3Lc(2, 3, 0), epsSU2(3, 4), phic(4), eR(2, 1)),
    Op(lambdaDelta3ec(0, 1), eRc(2, 1), epsSU2(3, 4), phi(4), Delta3L(2, 3, 0)),

    Op(lambdaNRl(0, 1), NRc(2, 0), epsSU2(3, 4), phi(4), lL(2, 3, 1)),
    Op(lambdaNRlc(0, 1), lLc(2, 3, 1), epsSU2(3, 4), phic(4), NR(2, 0)),

    Op(lambdaNLl(0, 1), NL(2, 0), epsUp(2, 3), epsSU2(4, 5), phi(5), lL(3, 4, 1)),
    Op(lambdaNLlc(0, 1), lLc(2, 3, 1), epsUpDot(2, 4),
       epsSU2(3, 5), phic(5), NLc(4, 0)),

    Op(lambdaNmajl(0, 1), Nmaj(2, 0), epsUp(2, 3), epsSU2(4, 5), phi(5), lL(3, 4, 1)),
    Op(lambdaNmajlc(0, 1), lLc(2, 3, 1), epsUpDot(2, 4),
       epsSU2(3, 5), phic(5), Nmajc(4, 0)),

    Op(lambdaEl(0, 1), ERc(2, 0), phic(3), lL(2, 3, 1)),
    Op(lambdaElc(0, 1), lLc(2, 3, 1), phi(3), ER(2, 0)),

    half * Op(lambdaSigma0Rl(0, 1), Sigma0Rc(2, 3, 0),
              epsSU2(4, 5), phi(5), sigmaSU2(3, 4, 6), lL(2, 6, 1)),
    half * Op(lambdaSigma0Rlc(0, 1), lLc(2, 3, 1),
              sigmaSU2(4, 3, 5), epsSU2(5, 6), phic(6), Sigma0R(2, 4, 0)),

    half * Op(lambdaSigma0Ll(0, 1), Sigma0L(2, 3, 0), epsUp(2, 4),
              epsSU2(5, 6), phi(6), sigmaSU2(3, 5, 7), lL(4, 7, 1)),
    half * Op(lambdaSigma0Llc(0, 1), lLc(2, 3, 1), epsUpDot(2, 4),
              sigmaSU2(5, 3, 6), epsSU2(6, 7), phic(7), Sigma0Lc(4, 5, 0)),

    half * Op(lambdaSigma0majl(0, 1), Sigma0maj(2, 3, 0), epsUp(2, 4),
              epsSU2(5, 6), phi(6), sigmaSU2(3, 5, 7), lL(4, 7, 1)),
    half * Op(lambdaSigma0majlc(0, 1), lLc(2, 3, 1), epsUpDot(2, 4),
              sigmaSU2(5, 3, 6), epsSU2(6, 7), phic(7), Sigma0majc(4, 5, 0)),

    half * Op(lambdaSigma1l(0, 1), Sigma1Rc(2, 3, 0),
              phic(4), sigmaSU2(3, 4, 5), lL(2, 5, 1)),
    half * Op(lambdaSigma1lc(0, 1), lLc(2, 3, 1),
              sigmaSU2(4, 3, 5), phi(5), Sigma1R(2, 4, 0)))

L_scalars = -OpSum(
    Op(kappaS(0), S(0), phic(1), phi(1)),
    Op(lambdaS(0, 1), S(0), S(1), phic(2), phi(2)),
    Op(kappaS3(0, 1, 2), S(0), S(1), S(2)),

    Op(kappaXi0(0), phic(1), Xi0(2, 0), sigmaSU2(2, 1, 3), phi(3)),
    Op(lambdaXi0(0, 1), Xi0(2, 0), Xi0(2, 1), phic(3), phi(3)),

    Op(yevarphi(0, 1, 2), varphic(3, 0), eRc(4, 1), lL(4, 3, 2)),
    Op(yevarphic(0, 1, 2), varphi(3, 0), lLc(4, 3, 2), eR(4, 1)),
    Op(ydvarphi(0, 1, 2), varphic(3, 0), dRc(4, 5, 1), qL(4, 5, 3, 2)),
    Op(ydvarphic(0, 1, 2), varphi(3, 0), qLc(4, 5, 3, 2), dR(4, 5, 1)),
    Op(yuvarphi(0, 1, 2), varphic(3, 0), epsSU2(3, 4), qLc(5, 6, 4, 1), uR(5, 6, 2)),
    Op(yuvarphic(0, 1, 2), varphi(3, 0), epsSU2(3, 4), uRc(5, 6, 2), qL(5, 6, 4, 1)),
    Op(lambdavarphi(0), varphic(1, 0), phi(1), phic(2), phi(2)),
    Op(lambdavarphic(0), phic(1), varphi(1, 0), phic(2), phi(2)),

    Op(ylXi1(0, 1, 2), Xi1c(3, 0), lLc(4, 5, 1), sigmaSU2(3, 5, 6),
       epsUpDot(4, 7), epsSU2(6, 8), lLc(7, 8, 2)),
    Op(ylXi1c(0, 1, 2), Xi1(3, 0), epsSU2(4, 5), lL(6, 5, 2),
       sigmaSU2(3, 4, 7), epsUp(6, 8), lL(8, 7, 1)),
    Op(kappaXi1(0), Xi1c(1, 0), epsSU2(2, 3), phi(3), sigmaSU2(1, 2, 4), phi(4)),
    Op(kappaXi1c(0), Xi1(1, 0), phic(2), sigmaSU2(1, 2, 3), epsSU2(3, 4), phic(4)),
    half * Op(lambdaXi1(0, 1), Xi1c(2, 0), Xi1(2, 1), phic(3), phi(3)),
    half * Op(lambdaTildeXi1(0, 1), fSU2(2, 3, 4), Xi1c(2, 0), Xi1(3, 1),
              phic(5), sigmaSU2(4, 5, 6), phi(6)),

    Op(kappaSXi0(0, 1, 2), S(0), Xi0(3, 1), Xi0(3, 2)),
    Op(kappaSXi1(0, 1, 2), S(0), Xi1c(3, 1), Xi1(3, 2)),
    Op(kappaXi0Xi1(0, 1, 2), fSU2(3, 4, 5), Xi0(3, 0), Xi1c(4, 1), Xi1(5, 2)),
    Op(lambdaSXi0(0, 1), S(0), Xi0(2, 1), phic(3), sigmaSU2(2, 3, 4), phi(4)),

    Op(kappaSvarphi(0, 1), S(0), varphic(2, 1), phi(2)),
    Op(kappaSvarphic(0, 1), S(0), phic(2), varphi(2, 1)),
    Op(kappaXi0varphi(0, 1), Xi0(2, 0), varphic(3, 1), sigmaSU2(2, 3, 4), phi(4)),
    Op(kappaXi0varphic(0, 1), Xi0(2, 0), phic(3), sigmaSU2(2, 3, 4), varphi(4, 1)),
    Op(kappaXi1varphi(0, 1), Xi1c(2, 0), epsSU2(5, 3), varphi(3, 1),
       sigmaSU2(2, 5, 4), phi(4)),
    Op(kappaXi1varphic(0, 1), Xi1(2, 0), phic(3), sigmaSU2(2, 3, 5),
       epsSU2(5, 4), varphic(4, 1)),
    Op(lambdaSXi1(0, 1), S(0), Xi1c(2, 1), epsSU2(3, 4), phi(4),
       sigmaSU2(2, 3, 5), phi(5)),
    Op(lambdaSXi1c(0, 1), S(0), Xi1(2, 1), phic(3),
       sigmaSU2(2, 3, 4), epsSU2(4, 5), phic(5)),
    Op(lambdaXi1Xi0(0, 1), fSU2(2, 3, 4), Xi1c(2, 0), Xi0(3, 1),
       epsSU2(5, 6), phi(6), sigmaSU2(4, 5, 7), phi(7)),
    Op(lambdaXi1Xi0c(0, 1), fSU2(2, 3, 4), Xi1(2, 0), Xi0(3, 1),
       phic(5), sigmaSU2(4, 5, 6), epsSU2(6, 7), phic(7)),
    
    # -----------------------------------------------------------------------
    # These don't contribute:
    # Op(ylS1(0, 1, 2), S1c(0), lLc(3, 4, 1), epsUpDot(3, 5),
    #    epsSU2(4, 6), lLc(5, 6, 2)),
    # Op(ylS1c(0, 1, 2), S1(0), lL(3, 4, 2), epsUp(3, 5),
    #    epsSU2(6, 4), lL(5, 6, 1)),

    # Op(yeS2(0, 1, 2), S2c(0), eRc(3, 1), epsDown(3, 4), eRc(4, 2)),
    # Op(yeS2c(0, 1, 2), S2(0), eR(3, 2), epsDownDot(3, 4), eR(4, 1)),
    
    # Op(lambdaTheta1(0), phic(1), sigmaSU2(2, 1, 3), phi(3), CSU2(4, 2, 5),
    #    epsSU2(5, 6), phic(6), epsSU2Q(4, 7), Theta1(7, 0)),
    # Op(lambdaTheta1c(0), phic(1), sigmaSU2(2, 1, 3), phi(3), CSU2c(
    # ...
    # ------------------------------------------------------------------------
)

L_1vectors = -OpSum(
    Op(glB(0, 1, 2), B(3, 0), lLc(4, 5, 1), sigma4bar(3, 4, 6), lL(6, 5, 2)),
    Op(gqB(0, 1, 2), B(3, 0), qLc(4, 5, 6, 1), sigma4bar(3, 4, 7), qL(7, 5, 6, 2)),
    Op(geB(0, 1, 2), B(3, 0), eRc(4, 1), sigma4(3, 4, 5), eR(5, 2)),
    Op(gdB(0, 1, 2), B(3, 0), dRc(4, 5, 1), sigma4(3, 4, 6), dR(6, 5, 2)),
    Op(guB(0, 1, 2), B(3, 0), uRc(4, 5, 1), sigma4(3, 4, 6), uR(6, 5, 2)),
    number_op(1j) * Op(gphiB(0), B(1, 0), phic(2), D(1, phi(2))),
    number_op(-1j) * Op(gphiBc(0), B(1, 0), D(1, phic(2)), phi(2)),

    half * Op(glW(0, 1, 2), W(3, 4, 0), lLc(5, 6, 1), sigma4bar(3, 5, 7),
              sigmaSU2(4, 6, 8), lL(7, 8, 2)),
    half * Op(gqW(0, 1, 2), W(3, 4, 0), qLc(5, 6, 7, 1), sigma4bar(3, 5, 8),
              sigmaSU2(4, 7, 9), qL(8, 6, 9, 2)),
    number_op(0.5j) * Op(gphiW(0), W(1, 2, 0), phic(3), sigmaSU2(2, 3, 4),
                         D(1, phi(4))),
    number_op(-0.5j) * Op(gphiWc(0), W(1, 2, 0), D(1, phic(3)),
                          sigmaSU2(2, 3, 4), phi(4)),
    
    Op(gduB1(0, 1, 2), B1c(3, 0), dRc(4, 5, 1), sigma4(3, 4, 6), uR(6, 5, 2)),
    Op(gduB1c(0, 1, 2), B1(3, 0), uRc(4, 5, 2), sigma4(3, 4, 6), dR(6, 5, 1)),
    number_op(1j) * Op(gphiB1(0), B1c(1, 0), D(1, phi(2)), epsSU2(2, 3), phi(3)),
    number_op(-1j) * Op(gphiB1c(0), B1(1, 0), D(1, phic(2)), epsSU2(2, 3), phic(3)),

    number_op(0.5j) * Op(gphiW1(0), W1c(1, 2, 0), D(1, phi(3)),
                         epsSU2(3, 4), sigmaSU2(2, 4, 5), phi(5)),
    number_op(-0.5j) * Op(gphiW1c(0), W1(1, 2, 0), D(1, phic(3)),
                          epsSU2(3, 4), sigmaSU2(2, 5, 4), phic(5))
    
    # ------ (The rest doesn't contribute) ------
)

L_2vectors = -OpSum(
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

L_VDS = -OpSum(
    Op(deltaB(0, 1), B(2, 0), D(2, S(1))),
    Op(deltaW(0, 1), W(2, 3, 0), D(2, Xi0(3, 1))),
    Op(deltaL1(0, 1), L1c(2, 3, 0), D(2, varphi(3, 1))),
    Op(deltaL1c(0, 1), L1(2, 3, 0), D(2, varphic(3, 1))),
    Op(deltaW1(0, 1), W1c(2, 3, 0), D(2, Xi1(3, 1))),
    Op(deltaW1c(0, 1), W1(2, 3, 0), D(2, Xi1c(3, 1))))

L_VVS = -OpSum(
    Op(epsilonS(0, 1, 2), S(0), L1c(3, 4, 1), L1(3, 4, 2)),
    Op(epsilonXi0(0, 1, 2), Xi0(3, 0), L1c(4, 5, 1), sigmaSU2(3, 5, 6), L1(4, 6, 2)),
    Op(epsilonXi1(0, 1, 2), Xi1(3, 0), L1c(4, 5, 1), sigmaSU2(3, 5, 6),
       epsSU2(6, 7), L1c(4, 7, 2)),
    Op(epsilonXi1c(0, 1, 2), Xi1c(3, 0), epsSU2(4, 5), L1(6, 5, 2),
       sigmaSU2(3, 4, 7), L1(6, 7, 1)))

L_VSSM = -OpSum(
    Op(g1Xi1L1(0, 1), epsSU2(2, 3), phi(3), sigmaSU2(4, 2, 5),
       D(6, Xi1c(4, 0)), L1(6, 5, 1)),
    Op(g1Xi1L1c(0, 1), L1c(2, 3, 1), sigmaSU2(4, 3, 5),
       D(2, Xi1(4, 0)), epsSU2(5, 6), phic(6)),
    Op(g1Xi0L1(0, 1), phic(2), sigmaSU2(3, 2, 4), D(5, Xi0(3, 0)), L1(5, 4, 1)),
    Op(g1Xi0L1c(0, 1), L1c(2, 3, 1), sigmaSU2(4, 3, 5), D(2, Xi0(4, 0)), phi(5)),
    Op(g1SL1(0, 1), phic(2), D(3, S(0)), L1(3, 2, 1)),
    Op(g1SL1c(0, 1), L1c(2, 3, 1), D(2, S(0)), phi(3)),

    Op(g2Xi1L1(0, 1), epsSU2(2, 3), D(4, phi(3)), sigmaSU2(5, 2, 6),
       Xi1c(5, 0), L1(4, 6, 1)),
    Op(g2Xi1L1c(0, 1), L1c(2, 3, 1), sigmaSU2(4, 3, 5), Xi1(4, 0),
       epsSU2(5, 6), D(2, phic(6))),
    Op(g2Xi0L1(0, 1), D(2, phic(3)), sigmaSU2(4, 3, 5), Xi0(4, 0), L1(2, 5, 1)),
    Op(g2Xi0L1c(0, 1), L1c(2, 3, 1), sigmaSU2(4, 3, 5), Xi0(4, 0), D(2, phi(5))),
    Op(g2SL1(0, 1), D(2, phic(3)), S(0), L1(2, 3, 1)),
    Op(g2SL1c(0, 1), L1c(2, 3, 1), S(0), D(2, phi(3))))

L_VFSM = -OpSum(
    Op(zSigma0L(0, 1, 2), Sigma0Lc(3, 4, 0), sigma4bar(5, 3, 6),
       epsSU2(7, 8), L1(5, 8, 1), sigmaSU2(4, 7, 9), lL(6, 9, 2)),
    Op(zSigma0Lc(0, 1, 2), lLc(3, 4, 2), sigmaSU2(5, 4, 6),
       epsSU2(6, 7), L1c(8, 7, 1), sigma4bar(8, 3, 9), Sigma0L(9, 5, 0)),
    Op(zSigma0R(0, 1, 2), Sigma0R(3, 4, 0), epsDownDot(3, 5), sigma4bar(6, 5, 7),
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

    Op(zDelta1(0, 1, 2), Delta1Rc(3, 4, 0), sigma4(5, 3, 6), L1(5, 4, 1), eR(6, 2)),
    Op(zDelta1c(0, 1, 2), eRc(3, 2), sigma4(4, 3, 5), L1c(4, 6, 1), Delta1R(5, 6, 0)),

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

    Op(zUDd(0, 1, 2), UDRc(3, 4, 5, 0), sigma4(6, 3, 7), L1(6, 5, 1), dR(7, 4, 2)),
    Op(zUDdc(0, 1, 2), dRc(3, 4, 2), sigma4(5, 3, 6), L1c(5, 7, 1), UDR(6, 4, 7, 0)),

    Op(zXU(0, 1, 2), XURc(3, 4, 5, 0), sigma4(6, 3, 7), L1(6, 5, 1), uR(7, 4, 2)),
    Op(zXUc(0, 1, 2), uRc(3, 4, 2), sigma4(5, 3, 6), L1c(5, 7, 1), XUR(6, 4, 7, 0)),

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
    Op(wlXi0(0, 1, 2), Xi0(3, 0), Delta1Rc(4, 5, 1), sigmaSU2(3, 5, 6), lL(4, 6, 2)),
    Op(wlXi0c(0, 1, 2), Xi0(3, 0), lLc(4, 5, 2), sigmaSU2(3, 5, 6), Delta1R(4, 6, 1)),

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

interaction_lagrangian = (
    L_quarks + L_leptons + L_scalars + L_1vectors + L_2vectors +
    L_VDS + L_VVS + L_VSSM + L_VFSM + L_SFSM)


# -- Heavy fields --

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
heavy_Omega1 = ComplexScalar("Omega1", "Omega1c", 2)
heavy_Omega2 = ComplexScalar("Omega2", "Omega2c", 2)
heavy_Omega4 = ComplexScalar("Omega4", "Omega4c", 2)
heavy_Upsilon = ComplexScalar("Upsilon", "Upsilonc", 3)
heavy_Phi = ComplexScalar("Phi", "Phic", 3)

heavy_scalars = [
    heavy_S, heavy_S1, heavy_S2, heavy_varphi, heavy_Xi0, heavy_Xi1,
    heavy_Theta1, heavy_Theta3, heavy_omega1, heavy_omega2, heavy_omega4,
    heavy_Pi1, heavy_Pi7, heavy_zeta, heavy_Omega1, heavy_Omega2,
    heavy_Omega4, heavy_Upsilon, heavy_Phi
]


heavy_B = RealVector("B", 2)
heavy_W = RealVector("W", 3)
heavy_G = RealVector("G", 3)
heavy_H = RealVector("H", 4)
heavy_B1 = ComplexVector("B1", "B1c", 2)
heavy_W1 = ComplexVector("W1", "W1c", 3)
heavy_G1 = ComplexVector("G1", "G1c", 3)
heavy_L1 = ComplexVector("L1", "L1c", 3)
heavy_L3 = ComplexVector("L3", "L3c", 3)
heavy_U2 = ComplexVector("U2", "U2c", 3)
heavy_U5 = ComplexVector("U5", "U5c", 3)
heavy_Q1 = ComplexVector("Q1", "Q1c", 4)
heavy_Q5 = ComplexVector("Q5", "Q5c", 4)
heavy_X = ComplexVector("X", "Xc", 4)
heavy_Y1 = ComplexVector("Y1", "Y1c", 4)
heavy_Y5 = ComplexVector("Y5", "Y5c", 4)

heavy_vectors = [
    heavy_B, heavy_W, heavy_G, heavy_H, heavy_B1, heavy_W1, heavy_G1,
    heavy_L1, heavy_L3, heavy_U2, heavy_U5, heavy_Q1, heavy_Q5, heavy_X,
    heavy_Y1, heavy_U5]


heavy_U = VectorLikeFermion("U", "UL", "UR", "ULc", "URc", 3)
heavy_D = VectorLikeFermion("D", "DL", "DR", "DLc", "DRc", 3)
heavy_XU = VectorLikeFermion("XU", "XUL", "XUR", "XULc", "XURc", 4)
heavy_UD = VectorLikeFermion("UD", "UDL", "UDR", "UDLc", "UDRc", 4)
heavy_DY = VectorLikeFermion("DY", "DYL", "DYR", "DYLc", "DYRc", 4)
heavy_XUD = VectorLikeFermion("XUD", "XUDL", "XUDR", "XUDLc", "XUDRc", 4)
heavy_UDY = VectorLikeFermion("UDY", "UDYL", "UDYR", "UDYLc", "UDYRc", 4)

heavy_quarks = [heavy_U, heavy_D, heavy_XU, heavy_UD, heavy_DY, heavy_XUD, heavy_UDY]


heavy_N = VectorLikeFermion("N", "NL", "NR", "NLc", "NRc", 2)
heavy_E = VectorLikeFermion("E", "EL", "ER", "ELc", "ERc", 2)
heavy_Delta1 = VectorLikeFermion("Delta1", "Delta1L", "Delta1R",
                                 "Delta1Lc", "Delta1Rc", 3)
heavy_Delta3 = VectorLikeFermion("Delta3", "Delta3L", "Delta3R",
                                 "Delta3Lc", "Delta3Rc", 3)
heavy_Sigma0 = VectorLikeFermion("Sigma0", "Sigma0L", "Sigma0R",
                                 "Sigma0Lc", "Sigma0Rc", 3)
heavy_Sigma1 = VectorLikeFermion("Sigma1", "Sigma1L", "Sigma1R",
                                 "Sigma1Lc", "Sigma1Rc", 3)

heavy_Nmaj = MajoranaFermion("Nmaj", "Nmajc", 2)
heavy_Sigma0maj = MajoranaFermion("Sigma0maj", "Sigma0majc", 3)

heavy_leptons = [heavy_N, heavy_E, heavy_Delta1, heavy_Delta3,
                 heavy_Sigma0, heavy_Sigma1,
                 heavy_Nmaj, heavy_Sigma0maj]


heavy_fields = heavy_scalars + heavy_vectors + heavy_quarks + heavy_leptons

# *** L1 vector ***

# -- Integration --

print "Integrating...",
sys.stdout.flush()
eff_lag = integrate(heavy_fields, interaction_lagrangian, max_dim=6)
print "done."

# -- Remove operators without ML1 --

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

eff_lag = group_op_sum(OpSum(*[
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

# -- Transformations --

Okinphi = tensor_op("Okinphi")
Ophi4 = tensor_op("Ophi4")
Ophi2 = tensor_op("Ophi2")
Oye = flavor_tensor_op("Oye")
Oyec = flavor_tensor_op("Oyec")
Oyd = flavor_tensor_op("Oyd")
Oydc = flavor_tensor_op("Oydc")
Oyu = flavor_tensor_op("Oyu")
Oyuc = flavor_tensor_op("Oyuc")

O5 = flavor_tensor_op("O5")
O5c = flavor_tensor_op("O5c")
O5aux = flavor_tensor_op("O5aux")
O5auxc = flavor_tensor_op("O5auxc")

Ophi = tensor_op("Ophi")
OphiD = tensor_op("OphiD")
Ophisq = tensor_op("Ophisq")
OD2 = tensor_op("OD2")
OD2c = tensor_op("OD2c")

OphiB = tensor_op("OphiB")
OWB = tensor_op("OWB")
OphiW = tensor_op("OphiW")
OphiBTilde = tensor_op("OphiBTilde")
OWBTilde = tensor_op("OWBTilde")
OphiWTilde = tensor_op("OphiWTilde")

Oephi = flavor_tensor_op("Oephi")
Odphi = flavor_tensor_op("Odphi")
Ouphi = flavor_tensor_op("Ouphi")
Oephic = flavor_tensor_op("Oephic")
Odphic = flavor_tensor_op("Odphic")
Ouphic = flavor_tensor_op("Ouphic")

OephiDaux = flavor_tensor_op("OephiDaux")
OephiDauxc = flavor_tensor_op("OephiDauxc")
OdphiDaux = flavor_tensor_op("OdphiDaux")
OdphiDauxc = flavor_tensor_op("OdphiDauxc")
OuphiDaux = flavor_tensor_op("OuphiDaux")
OuphiDauxc = flavor_tensor_op("OuphiDauxc")

O1phil = flavor_tensor_op("O1phil")
O1philc = flavor_tensor_op("O1philc")
O1phiq = flavor_tensor_op("O1phiq")
O1phiqc = flavor_tensor_op("O1phiqc")
O3phil = flavor_tensor_op("O3phil")
O3philc = flavor_tensor_op("O3philc")
O3phiq = flavor_tensor_op("O3phiq")
O3phiqc = flavor_tensor_op("O3phiqc")
O1phie = flavor_tensor_op("O1phie")
O1phiec = flavor_tensor_op("O1phiec")
O1phid = flavor_tensor_op("O1phid")
O1phidc = flavor_tensor_op("O1phidc")
O1phiu = flavor_tensor_op("O1phiu")
O1phiuc = flavor_tensor_op("O1phiuc")
Ophiud = flavor_tensor_op("Ophiud")
Ophiudc = flavor_tensor_op("Ophiudc")

O1ll = flavor_tensor_op("O1ll")
Oee = flavor_tensor_op("Oee")
Ole = flavor_tensor_op("Ole")
O1qd = flavor_tensor_op("O1qd")
O8qd = flavor_tensor_op("O8qd")
O1qu = flavor_tensor_op("O1qu")
O8qu = flavor_tensor_op("O8qu")
Oledq = flavor_tensor_op("Oledq")
Olequ = flavor_tensor_op("Olequ")
O1qud = flavor_tensor_op("O1qud")
Oledqc = flavor_tensor_op("Oledqc")
Olequc = flavor_tensor_op("Olequc")
O1qudc = flavor_tensor_op("O1qudc")

rules = [
    # -- Higgs and derivatives --
    
    # (Dphic phi) (Dphic phi) -> - (OphiD + Q1 + (DDphic phi) (phic phi))
    (Op(D(0, phic(1)), phi(1), D(0, phic(2)), phi(2)),
     -OpSum(OphiD,
            Op(D(0, phic(1)), D(0, phi(1)), phic(2), phi(2)),
            Op(D(1, D(1, phic(0))), phi(0), phic(2), phi(2)))),

    # (phic Dphi) (phic Dphi) -> - (OphiD + Q1 + (phic DDphi) (phic phi))
    (Op(phic(1), D(0, phi(1)), phic(2), D(0, phi(2))),
     -OpSum(OphiD,
            Op(D(0, phic(1)), D(0, phi(1)), phic(2), phi(2)),
            Op(phic(1), D(0, D(0, phi(1))), phic(2), phi(2)))),
    
    # Q1 -> 1/2 Ophisq - 1/2 OD2 - 1/2 OD2c
    (Op(D(0, phic(1)), D(0, phi(1)), phic(2), phi(2)),
     OpSum(half * Ophisq,
           -half * Op(phic(0), D(1, D(1, phi(0))), phic(2), phi(2)),
           -half * Op(D(0, D(0, phic(1))), phi(1), phic(2), phi(2)))),

    # -- SU(2) --

    # sigmaSU2(0, -1, -2) sigmaSU2(0, -3, -4) ->
    # 2 kdelta(-1, -4) kdelta(-3, -2) - kdelta(-1, -2) kdelta(-3, -4)
    (Op(sigmaSU2(0, -1, -2), sigmaSU2(0, -3, -4)),
     OpSum(number_op(2) * Op(kdelta(-1, -4), kdelta(-3, -2)),
           -Op(kdelta(-1, -2), kdelta(-3, -4)))),
    
    # epsSU2(-1, -2) epsSU2(-3, -4) ->
    # kdelta(-1, -3) kdelta(-2, -4) - kdelta(-1, -4) kdelta(-2, -3)
    (Op(epsSU2(-1, -2), epsSU2(-3, -4)),
     OpSum(Op(kdelta(-1, -3), kdelta(-2, -4)),
           -Op(kdelta(-1, -4), kdelta(-2, -3)))),
    
    # epsSU2(-1, 0) epsSU2(0, -2) -> -kdelta(-1, -2)
    (Op(epsSU2(-1, 0), epsSU2(0, -2)),
     -OpSum(Op(kdelta(-1, -2)))),

    # epsSU2(0, -1) epsSU2(0, -2) -> kdelta(-1, -2)
    (Op(epsSU2(0, -1), epsSU2(0, -2)),
     OpSum(Op(kdelta(-1, -2)))),
    
    # epsSU2(-1, 0) epsSU2(-2, 0) -> kdelta(-1, -2)
    (Op(epsSU2(-1, 0), epsSU2(-2, 0)),
     OpSum(Op(kdelta(-1, -2)))),

    # epsSU2(0, 0) -> 0
    (Op(epsSU2(0, 0)), OpSum()),

    # phi epsSU2 phi -> 0
    (Op(phi(0), epsSU2(0, 1), phi(1)), OpSum()),

    # phic epsSU2 phic -> 0
    (Op(phic(0), epsSU2(0, 1), phic(1)), OpSum()),
    
    # -- Lorentz --
    
    # epsUp(-1, -2) epsUpDot(-3, -4) ->
    # -1/2 sigma4bar(0, -3, -1) sigma4bar(0, -4, -2)
    (Op(epsUp(-1, -2), epsUpDot(-3, -4)),
     OpSum(-half * Op(sigma4bar(0, -3, -1), sigma4bar(0, -4, -2)))),

    # epsDown(-1, -2) epsDownDot(-3, -4) ->
    # -1/2 sigma4(0, -1, -3) sigma4(0, -2, -4)
    (Op(epsDown(-1, -2), epsDownDot(-3, -4)),
     OpSum(-half * Op(sigma4(0, -1, -3), sigma4(0, -2, -4)))),

    # epsUp(0, -1) epsDown(0, -2) -> -kdelta(-1, -2)
    (Op(epsUp(0, -1), epsDown(0, -2)),
     OpSum(Op(kdelta(-1, -2)))),

    # epsDown(-1, 0) epsUp(0, -2) -> kdelta(-1, -2)
    (Op(epsDown(-1, 0), epsUp(0, -2)),
     OpSum(-Op(kdelta(-1, -2)))),

    # epsUp(0, -1) epsDown(0, -2) -> -kdelta(-1, -2)
    (Op(epsUp(-1, 0), epsDown(-2, 0)),
     OpSum(Op(kdelta(-1, -2)))),

    # epsUpDot(-1, 0) epsDownDot(0, -2) -> kdelta(-1, -2)
    (Op(epsUpDot(-1, 0), epsDownDot(0, -2)),
     OpSum(-Op(kdelta(-1, -2)))),

    # epsUpDot(-1, 0) epsDownDot(-2, 0) -> kdelta(-1, -2)
    (Op(epsUpDot(-1, 0), epsDownDot(-2, 0)),
     OpSum(Op(kdelta(-1, -2)))),

    # epsDownDot(-1, 0) epsUpDot(0, -2) -> kdelta(-1, -2)
    (Op(epsDownDot(-1, 0), epsUpDot(0, -2)),
     OpSum(-Op(kdelta(-1, -2)))),

    # -- Higgs and fermion doublets --

    # (lLc Dphi) gamma (phic lL) -> -i/2 (O3phil + O1phil)
    (Op(lLc(0, 1, -1), sigma4bar(2, 0, 3), lL(3, 4, -2), phic(4), D(2, phi(1))),
     OpSum(number_op(-0.5j) * O3phil(-1, -2),
           number_op(-0.5j) * O1phil(-1, -2))),

    # (lLc phi) gamma (Dphic lL) -> i/2 (O3philc + O1philc)
    (Op(lLc(0, 1, -2), sigma4bar(2, 0, 3), lL(3, 4, -1), D(2, phic(4)), phi(1)),
     OpSum(number_op(0.5j) * O3philc(-1, -2),
           number_op(0.5j) * O1philc(-1, -2))),

    # (qLc Dphi) gamma (phic qL) -> -i/2 (O3phiq + O1phiq)
    (Op(qLc(0, 5, 1, -1), sigma4bar(2, 0, 3), qL(3, 5, 4, -2), phic(4), D(2, phi(1))),
     OpSum(number_op(-0.5j) * O3phiq(-1, -2),
           number_op(-0.5j) * O1phiq(-1, -2))),

    # (qLc phi) gamma (Dphic qL) -> i/2 (O3phiqc + O1phiqc)
    (Op(qLc(0, 5, 1, -2), sigma4bar(2, 0, 3), qL(3, 5, 4, -1), D(2, phic(4)), phi(1)),
     OpSum(number_op(0.5j) * O3phiqc(-1, -2),
           number_op(0.5j) * O1phiqc(-1, -2))),

    # -- Four-fermion Fierz reorderings

    (Op(lLc(0, 1, -1), eR(0, -2), eRc(2, -3), lL(2, 1, -4)),
     OpSum(-half * Ole(-1, -4, -3, -2))),

    (Op(qLc(0, 1, 2, -1), dR(0, 1, -2), dRc(3, 4, -3), qL(3, 4, 2, -4)),
     OpSum(-number_op(1./6) * O1qd(-1, -4, -3, -2),
           -O8qd(-1, -4, -3, -2))),

    (Op(qLc(0, 1, 2, -1), uR(0, 1, -2), uRc(3, 4, -3), qL(3, 4, 2, -4)),
     OpSum(-number_op(1./6) * O1qu(-1, -4, -3, -2),
           -O8qu(-1, -4, -3, -2))),

    # -- Involving field strengths and Higgs --

    # (Dphic Dphi) bFS
    (Op(D(0, phic(1)), D(2, phi(1)), bFS(0, 2)),
     OpSum(number_op(-0.5) * Op(phic(1), D(0, D(2, phi(1))), bFS(0, 2)),
           number_op(0.5) * Op(D(0, D(2, phic(1))), phi(1), bFS(0, 2)),
           number_op(-0.5) * Op(phic(1), D(2, phi(1)), D(0, bFS(0, 2))),
           number_op(0.5) * Op(D(2, phic(1)), phi(1), D(0, bFS(0, 2))))),

    # (Dphic sigmaSu2 Dphi) wFS
    (Op(D(0, phic(1)), sigmaSU2(3, 1, 4), D(2, phi(4)), wFS(0, 2, 3)),
     OpSum(number_op(-0.5) * Op(phic(1), sigmaSU2(3, 1, 4),
                                D(0, D(2, phi(4))), wFS(0, 2, 3)),
           number_op(0.5) * Op(D(0, D(2, phic(1))), sigmaSU2(3, 1, 4),
                               phi(4), wFS(0, 2, 3)),
           number_op(-0.5) * Op(phic(1), sigmaSU2(3, 1, 4),
                                D(2, phi(4)), D(0, wFS(0, 2, 3))),
           number_op(0.5) * Op(D(2, phic(1)), sigmaSU2(3, 1, 4),
                               phi(4), D(0, wFS(0, 2, 3))))),

    # (Dphic sigmaSU2 Dphi) esp4 bFS
    (Op(D(0, phic(1)), D(2, phi(1)), eps4(0, 2, 3, 4), bFS(3, 4)),
     OpSum(number_op(-0.5) * Op(phic(1), D(0, D(2, phi(1))),
                                eps4(0, 2, 3, 4), bFS(3, 4)),
           number_op(0.5) * Op(D(0, D(2, phic(1))), phi(1),
                               eps4(0, 2, 3, 4), bFS(3, 4)))),

    # (Dphic sigmaSU2 Dphi) eps4 wFS
    (Op(D(0, phic(1)), sigmaSU2(3, 1, 4), D(2, phi(4)),
        eps4(0, 2, 5, 6), wFS(5, 6, 3)),
     OpSum(number_op(-0.5) * Op(phic(1), sigmaSU2(3, 1, 4), D(0, D(2, phi(4))),
                                eps4(0, 2, 5, 6), wFS(5, 6, 3)),
           number_op(0.5) * Op(D(0, D(2, phic(1))), sigmaSU2(3, 1, 4), phi(4),
                               eps4(0, 2, 5, 6), wFS(5, 6, 3)))),

    # (phic DDphi) bFs
    (Op(phic(2), D(0, D(1, phi(2))), bFS(0, 1)),
     OpSum(number_op(-0.25j) * Op(gb()) * OphiB,
           number_op(-0.25j) * Op(gw()) * OWB)),
    (Op(D(0, D(1, phic(2))), phi(2), bFS(0, 1)),
     OpSum(number_op(0.25j) * Op(gb()) * OphiB,
           number_op(0.25j) * Op(gw()) * OWB)),

    # (phic sigmaSU2 DDphi) wFS
    (Op(phic(2), sigmaSU2(3, 2, 4), D(0, D(1, phi(4))), wFS(0, 1, 3)),
     OpSum(number_op(-0.25j) * Op(gb()) * OWB,
           number_op(-0.25j) * Op(gw()) * OphiW)),
    (Op(D(0, D(1, phic(2))), sigmaSU2(3, 2, 4), phi(4), wFS(0, 1, 3)),
     OpSum(number_op(0.25j) * Op(gb()) * OWB,
           number_op(0.25j) * Op(gw()) * OphiW)),

    # (phic DDphi) eps4 bFS
    (Op(phic(2), D(0, D(1, phi(2))), eps4(0, 1, 3, 4), bFS(3, 4)),
     OpSum(number_op(-0.25j) * Op(gb()) * OphiBTilde,
           number_op(-0.25j) * Op(gw()) * OWBTilde)),
    (Op(D(0, D(1, phic(2))), phi(2), eps4(0, 1, 3, 4), bFS(3, 4)),
     OpSum(number_op(0.25j) * Op(gb()) * OphiBTilde,
           number_op(0.25j) * Op(gw()) * OWBTilde)),

    # (phic) sigmaSU2 DDPhi) eps4 wFS
    (Op(phic(2), sigmaSU2(3, 2, 4), D(0, D(1, phi(4))),
        eps4(0, 1, 5, 6), wFS(5, 6, 3)),
     OpSum(number_op(-0.25j) * Op(gb()) * OWBTilde,
           number_op(-0.25j) * Op(gw()) * OphiWTilde)),
    (Op(D(0, D(1, phic(2))), sigmaSU2(3, 2, 4), phi(4),
        eps4(0, 1, 5, 6), wFS(5, 6, 3)),
     OpSum(number_op(0.25j) * Op(gb()) * OWBTilde,
           number_op(0.25j) * Op(gw()) * OphiWTilde)),

    # (Dmu Dn phic) (Dmu Dnu phi)
    (Op(D(0, D(1, phic(2))), D(0, D(1, phi(2)))),
     OpSum(-number_op(0.5) * Op(D(0, D(0, D(1, phic(2)))), D(1, phi(2))),
           -number_op(0.5) * Op(D(1, phic(2)), D(0, D(0, D(1, phi(2))))))),

    # (Dmu Dn phic) (Dnu Dmu phi)
    (Op(D(0, D(1, phic(2))), D(1, D(0, phi(2)))),
     OpSum(-number_op(0.5) * Op(D(1, D(0, D(1, phic(2)))), D(0, phi(2))),
           -number_op(0.5) * Op(D(1, phic(2)), D(0, D(1, D(0, phi(2))))))),

    # (Dmu phic) (Dnu Dmu Dnu phi)
    (Op(D(0, phic(1)), D(2, D(0, D(2, phi(1))))),
     OpSum(-Op(D(0, D(0, phic(1))), D(2, D(2, phi(1)))),
           number_op(0.5j) * Op(gb(), bFS(0, 1), D(0, phic(2)), D(1, phi(2))),
           number_op(0.5j) * Op(gw(), wFS(0, 1, 2), D(0, phic(3)),
                                sigmaSU2(2, 3, 4), D(1, phi(4))))),

    # (Dmu Dnu Dmu phic) (Dnu phi)
    (Op(D(0, D(2, D(0, phic(1)))), D(2, phi(1))),
     OpSum(-Op(D(0, D(0, phic(1))), D(2, D(2, phi(1)))),
           number_op(0.5j) * Op(gb(), bFS(0, 1), D(0, phic(2)), D(1, phi(2))),
           number_op(0.5j) * Op(gw(), wFS(0, 1, 2), D(0, phic(3)),
                                 sigmaSU2(2, 3, 4), D(1, phi(4))))),

    # (Dmu phic) (Dnu Dnu Dmu phi)
    (Op(D(0, phic(1)), D(2, D(2, D(0, phi(1))))),
     OpSum(-Op(D(0, D(0, phic(1))), D(2, D(2, phi(1)))),
           number_op(1j) * Op(gb(), D(0, phic(1)), D(2, phi(1)), bFS(0, 2)),
           number_op(1j) * Op(gw(), D(0, phic(1)), sigmaSU2(2, 1, 3),
                              D(4, phi(3)), wFS(0, 4, 2)),
           number_op(-0.5j) * Op(gb(), D(0, phic(1)), phi(1), D(2, bFS(2, 0))),
           number_op(-0.5j) * Op(gw(), D(0, phic(1)), sigmaSU2(2, 1, 3), phi(3),
                                 D(4, wFS(4, 0, 2))))),

    # (Dmu Dmu Dnu phic) (Dnu phi)
    (Op(D(0, D(0, D(1, phic(2)))), D(1, phi(2))),
     OpSum(-Op(D(0, D(0, phic(1))), D(2, D(2, phi(1)))),
           number_op(1j) * Op(gb(), D(0, phic(1)), D(2, phi(1)), bFS(0, 2)),
           number_op(1j) * Op(gw(), D(0, phic(1)), sigmaSU2(2, 1, 3),
                              D(4, phi(3)), wFS(0, 4, 2)),
           number_op(0.5j) * Op(gb(), phic(1), D(0, phi(1)), D(2, bFS(2, 0))),
           number_op(0.5j) * Op(gw(), phic(1), sigmaSU2(2, 1, 3), D(0, phi(3)),
                                D(4, wFS(4, 0, 2))))),

    # (Dmu phic) (Dmu Dnu Dnu phi) -> - (Dmu Dmu phic) (Dnu Dnu phi)
    (Op(D(0, phic(1)), D(0, D(2, D(2, phi(1))))),
     OpSum(-Op(D(0, D(0, phic(1))), D(2, D(2, phi(1)))))),

    # (Dmu Dnu Dnu phic) (Dmu phi) -> - (Dmu Dmu phic) (Dnu Dnu phi)
    (Op(D(0, phi(1)), D(0, D(2, D(2, phic(1))))),
     OpSum(-Op(D(0, D(0, phic(1))), D(2, D(2, phi(1)))))),

    # -- Pass derivatives from fermions to Higgs --

    # Dphic DeRc lL -> -OephiDaux - DDphic eRc lL
    (Op(D(0, phic(1)), D(0, eRc(2, -1)), lL(2, 1, -2)),
     OpSum(-OephiDaux(-1, -2),
           -Op(D(0, D(0, phic(1))), eRc(2, -1), lL(2, 1, -2)))),

    # Dphi lLc DeR -> -OephiDauxc - DDphi lLc eR
    (Op(D(0, phi(1)), lLc(2, 1, -2), D(0, eR(2, -1))),
     OpSum(-OephiDauxc(-1, -2),
           -Op(D(0, D(0, phi(1))), lLc(2, 1, -2), eR(2, -1)))),

    # Dphic DdRc qL -> -OdphiDaux - DDphic dRc qL
    (Op(D(0, phic(1)), D(0, dRc(2, 3, -1)), qL(2, 3, 1, -2)),
     OpSum(-OdphiDaux(-1, -2),
           -Op(D(0, D(0, phic(1))), dRc(2, 3, -1), qL(2, 3, 1, -2)))),

    # Dphi qLc DdR -> -OdphiDaux - DDphi qLc DdR
    (Op(D(0, phi(1)), qLc(2, 3, 1, -2), D(0, dR(2, 3, -1))),
     OpSum(-OdphiDauxc(-1, -2),
           -Op(D(0, D(0, phi(1))), qLc(2, 3, 1, -2), dR(2, 3, -1)))),

    # Dphi DuRc qL -> -OuphiDaux - DDphi uRc qL
    (Op(epsSU2(4, 1), D(0, phi(1)), D(0, uRc(2, 3, -1)), qL(2, 3, 4, -2)),
     OpSum(-OuphiDaux(-1, -2),
           -Op(epsSU2(4, 1), D(0, D(0, phi(1))), uRc(2, 3, -1), qL(2, 3, 4, -2)))),

    # Dphic qLc DuR -> -OuphiDaux - DDphic qLc DuR
    (Op(epsSU2(4, 1), D(0, phic(1)), qLc(2, 3, 4, -2), D(0, uR(2, 3, -1))),
     OpSum(-OuphiDauxc(-1, -2),
           -Op(epsSU2(4, 1), D(0, D(0, phic(1))), qLc(2, 3, 4, -2), uR(2, 3, -1))))
]

Ofphi2Ophif = [
    # deltaB qLc sigma4bar qL phic D phi ->
    # deltaB(-O1phiq - DqLc sigma4bar qL phic phi - qLc sigma4bar DqL phic phi)
    (Op(deltaB(-1, -2), phic(0), D(1, phi(0)),
        qLc(2, 3, 4, -3), sigma4bar(1, 2, 5), qL(5, 3, 4, -4)),
     OpSum(-Op(deltaB(-1, -2), D(1, phic(0)), phi(0),
               qLc(2, 3, 4, -3), sigma4bar(1, 2, 5), qL(5, 3, 4, -4)),
           -Op(deltaB(-1, -2), phic(0), phi(0),
               D(1, qLc(2, 3, 4, -3)), sigma4bar(1, 2, 5), qL(5, 3, 4, -4)),
           -Op(deltaB(-1, -2), phic(0), phi(0),
               qLc(2, 3, 4, -3), sigma4bar(1, 2, 5), D(1, qL(5, 3, 4, -4))))),

    # deltaB lLc sigma4bar lL phic D phi ->
    # deltaB(-O1phil - DlLc sigma4bar lL phic phi - lLc sigma4bar DlL phic phi)
    (Op(deltaB(-1, -2), phic(0), D(1, phi(0)),
        lLc(2, 3, -3), sigma4bar(1, 2, 5), lL(5, 3, -4)),
     OpSum(-Op(deltaB(-1, -2), D(1, phic(0)), phi(0),
               lLc(2, 3, -3), sigma4bar(1, 2, 5), lL(5, 3, -4)),
           -Op(deltaB(-1, -2), phic(0), phi(0),
               D(1, lLc(2, 3, -3)), sigma4bar(1, 2, 5), lL(5, 3, -4)),
           -Op(deltaB(-1, -2), phic(0), phi(0),
               lLc(2, 3, -3), sigma4bar(1, 2, 5), D(1, lL(5, 3, -4))))),

    # deltaW qLc sigma4bar sigmaSU2 qL phic sigmaSU2 D phi ->
    # deltaW(-iO3phiq - DqLc sigma4bar sigmaSU2 qL phic sigmaSU2 phi
    # - qLc sigma4bar sigmaSU2 DqL phic sigmaSU2 phi)
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

    # deltaW lLc sigma4bar sigmaSU2 lL phic sigmaSU2 D phi ->
    # deltaW(-iO3phil - DlLc sigmaSU2 sigma4bar lL phic sigmaSU2 phi -
    # lLc sigma4bar sigmaSU2 DlL phic sigmaSU2 phi)
    (Op(deltaW(-1, -2), phic(0), sigmaSU2(6, 0, 7), D(1, phi(7)),
        lLc(2, 3, -3), sigma4bar(1, 2, 5), sigmaSU2(6, 3, 8), lL(5, 8, -4)),
     OpSum(-Op(deltaW(-1, -2), D(1, phic(0)), sigmaSU2(6, 0, 7), phi(7),
               lLc(2, 3, -3), sigma4bar(1, 2, 5),
               sigmaSU2(6, 3, 8), lL(5, 8, -4)),
           -Op(deltaW(-1, -2), phic(0), sigmaSU2(6, 0, 7), phi(7),
               D(1, lLc(2, 3, -3)), sigma4bar(1, 2, 5),
               sigmaSU2(6, 3, 8), lL(5, 8, -4)),
           -Op(deltaW(-1, -2), phic(0), sigmaSU2(6, 0, 7), phi(7),
               lLc(2, 3, -3), sigma4bar(1, 2, 5),
               sigmaSU2(6, 3, 8), D(1, lL(5, 8, -4))))),

    # deltaB dRc sigma4bar dR phic D phi ->
    # deltaB(-O1phid - DdRc sigma4bar dR phic phi - dRc sigma4bar DdR phic phi)
    (Op(deltaB(-1, -2), phic(0), D(1, phi(0)),
        dRc(2, 3, -3), sigma4(1, 2, 5), dR(5, 3, -4)),
     OpSum(-Op(deltaB(-1, -2), D(1, phic(0)), phi(0),
               dRc(2, 3, -3), sigma4(1, 2, 5), dR(5, 3, -4)),
           -Op(deltaB(-1, -2), phic(0), phi(0),
               D(1, dRc(2, 3, -3)), sigma4(1, 2, 5), dR(5, 3, -4)),
           -Op(deltaB(-1, -2), phic(0), phi(0),
               dRc(2, 3, -3), sigma4(1, 2, 5), D(1, dR(5, 3, -4))))),

    # deltaB uRc sigma4bar uR phic D phi ->
    # deltaB(-O1phiu - DuRc sigma4bar uR phic phi - uRc sigma4bar DuR phic phi)
    (Op(deltaB(-1, -2), phic(0), D(1, phi(0)),
        uRc(2, 3, -3), sigma4(1, 2, 5), uR(5, 3, -4)),
     OpSum(-Op(deltaB(-1, -2), D(1, phic(0)), phi(0),
               uRc(2, 3, -3), sigma4(1, 2, 5), uR(5, 3, -4)),
           -Op(deltaB(-1, -2), phic(0), phi(0),
               D(1, uRc(2, 3, -3)), sigma4(1, 2, 5), uR(5, 3, -4)),
           -Op(deltaB(-1, -2), phic(0), phi(0),
               uRc(2, 3, -3), sigma4(1, 2, 5), D(1, uR(5, 3, -4))))),

    # deltaB eRc sigma4bar eR phic D phi ->
    # deltaB(-O1phie - DuRc sigma4bar eR phic phi - eRc sigma4bar DeR phic phi)
    (Op(deltaB(-1, -2), phic(0), D(1, phi(0)),
        eRc(2, -3), sigma4(1, 2, 5), eR(5, -4)),
     OpSum(-Op(deltaB(-1, -2), D(1, phic(0)), phi(0),
               eRc(2, -3), sigma4(1, 2, 5), eR(5, -4)),
           -Op(deltaB(-1, -2), phic(0), phi(0),
               D(1, eRc(2, -3)), sigma4(1, 2, 5), eR(5, -4)),
           -Op(deltaB(-1, -2), phic(0), phi(0),
               eRc(2, -3), sigma4(1, 2, 5), D(1, eR(5, -4))))),
]

SM_eoms = [
    (Op(D(0, D(0, phic(-1)))),
     OpSum(
         Op(mu2phi(), phic(-1)),
         -number_op(2) * Op(lambdaphi(), phic(0), phi(0), phic(-1)),
         -Op(ye(0, 1), lLc(2, -1, 0), eR(2, 1)),
         -Op(yd(0, 1), qLc(2, 3, -1, 0), dR(2, 3, 1)),
         -Op(V(0, 1), yuc(0, 2), uRc(3, 4, 2), qL(3, 4, 5, 1), epsSU2(5, -1)))),
     
    (Op(D(0, D(0, phi(-1)))),
     OpSum(
         Op(mu2phi(), phi(-1)),
         -number_op(2) * Op(lambdaphi(), phic(0), phi(0), phi(-1)),
         -Op(yec(0, 1), eRc(2, 1), lL(2, -1, 0)),
         -Op(ydc(0, 1), dRc(2, 3, 1), qL(2, 3, -1, 0)),
         -Op(Vc(0, 1), yu(0, 2), qLc(3, 4, 5, 1), uR(3, 4, 2), epsSU2(5, -1)))),

    (Op(D(0, bFS(0, -1))),
     -OpSum(
         number_op(-1./2) * Op(gb(), deltaFlavor(2, 4),
                               lLc(0, 1, 2), sigma4bar(-1, 0, 3), lL(3, 1, 4)),
         number_op(1./6) * Op(gb(), deltaFlavor(3, 5),
                              qLc(0, 1, 2, 3), sigma4bar(-1, 0, 4), qL(4, 1, 2, 5)),
         number_op(-1.) * Op(gb(), deltaFlavor(1, 3),
                             eRc(0, 1), sigma4(-1, 0, 2), eR(2, 3)),
         number_op(-1./3) * Op(gb(), deltaFlavor(2, 4),
                               dRc(0, 1, 2), sigma4(-1, 0, 3), dR(3, 1, 4)),
         number_op(2./3) * Op(gb(), deltaFlavor(2, 4),
                               uRc(0, 1, 2), sigma4(-1, 0, 3), uR(3, 1, 4)),
         number_op(1j/2.) * Op(gb(), phic(0), D(-1, phi(0))),
         number_op(-1j/2.) * Op(gb(), D(-1, phic(0)), phi(0)))),

    (Op(D(0, wFS(0, -1, -2))),
     -OpSum(
         number_op(-0.5) * Op(gw(), deltaFlavor(0, 1), lLc(3, 4, 0),
                              sigma4bar(-1, 3, 5), sigmaSU2(-2, 4, 6),
                              lL(5, 6, 1)),
         number_op(-0.5) * Op(gw(), deltaFlavor(0, 1), qLc(3, 4, 5, 0),
                              sigma4bar(-1, 3, 6), sigmaSU2(-2, 5, 7),
                              qL(6, 4, 7, 1)),
         number_op(-0.5j) * Op(gw(), phic(0), sigmaSU2(-2, 0, 1), D(-1, phi(1))),
         number_op(0.5j) * Op(gw(), D(-1, phic(0)), sigmaSU2(-2, 0, 1), phi(1)))),

    (Op(sigma4bar(0, -1, 1), D(0, lL(1, -2, -3))),
     OpSum(number_op(-1j) * Op(ye(-3, 0), phi(-2), eR(-1, 0)))),
    (Op(sigma4bar(0, 1, -1), D(0, lLc(1, -2, -3))),
     OpSum(number_op(1j) * Op(yec(-3, 0), phic(-2), eRc(-1, 0)))),

    (Op(sigma4bar(0, -1, 1), D(0, qL(1, -2, -3, -4))),
     OpSum(number_op(-1j) * Op(yd(-4, 0), phi(-3), dR(-1, -2, 0)),
           number_op(-1j) * Op(Vc(0, -4), yu(0, 1), epsSU2(-3, 2), phic(2), uR(-1, -2, 1)))),
    (Op(sigma4bar(0, 1, -1), D(0, qLc(1, -2, -3, -4))),
     OpSum(number_op(1j) * Op(ydc(-4, 0), phic(-3), dRc(-1, -2, 0)),
           number_op(1j) * Op(V(0, -4), yuc(0, 1), epsSU2(-3, 2), phi(2), uRc(-1, -2, 1)))),

    (Op(sigma4(0, -1, 1), D(0, eR(1, -2))),
     OpSum(number_op(-1j) * Op(yec(0, -2), phic(1), lL(-1, 1, 0)))),
    (Op(sigma4(0, 1, -1), D(0, eRc(1, -2))),
     OpSum(number_op(1j) * Op(ye(0, -2), phi(1), lLc(-1, 1, 0)))),

    (Op(sigma4(0, -1, 1), D(0, dR(1, -2, -3))),
     OpSum(number_op(-1j) * Op(ydc(0, -3), phic(1), qL(-1, -2, 1, 0)))),
    (Op(sigma4(0, 1, -1), D(0, dRc(1, -2, -3))),
     OpSum(number_op(1j) * Op(yd(0, -3), phi(1), qLc(-1, -2, 1, 0)))),

    (Op(sigma4(0, -1, 1), D(0, uR(1, -2, -3))),
     OpSum(number_op(-1j) * Op(V(0, 1), yuc(0, -3), epsSU2(2, 3), phi(3), qL(-1, -2, 2, 1)))),
    (Op(sigma4(0, 1, -1), D(0, uRc(1, -2, -3))),
     OpSum(number_op(1j) * Op(Vc(0, 1), yu(0, -3), epsSU2(2, 3), phic(3), qLc(-1, -2, 2, 1))))
]

definitions = [
    # (phi Dphic) (Dphic phi) -> OphiD
    (Op(phic(0), D(1, phi(0)), D(1, phic(2)), phi(2)),
     OpSum(OphiD)),
    
    # (phic phi)^3 -> 3 * Ophi
    (Op(phic(0), phi(0), phic(1), phi(1), phic(2), phi(2)),
     OpSum(number_op(3) * Ophi)),

    # (phic phi) B B -> OphiB
    (Op(phic(0), phi(0), B(1, 2), B(1, 2)),
     OpSum(OphiB)),

    # (phic phi) W B -> OWB
    (Op(phic(0), sigmaSU2(1, 0, 2), phi(2), W(3, 4, 2), B(3, 4)),
     OpSum(OWB)),

    # (phic phi) W W -> OphiW
    (Op(phic(0), phi(0), W(1, 2, 3), W(1, 2, 3)),
     OpSum(OphiW)),
    
    # (phic phi) (lLc phi eR) -> Oephi
    (Op(phic(0), phi(0), lLc(1, 2, -1), phi(2), eR(1, -2)),
     OpSum(Oephi(-1, -2))),

    # (phic phi) (eRc phic lL) -> Oephic
    (Op(phic(0), phi(0), eRc(1, -2), phic(2), lL(1, 2, -1)),
     OpSum(Oephic(-1, -2))),

    # (phic phi) (qLc phi dR) -> Odphi
    (Op(phic(0), phi(0), qLc(1, 2, 3, -1), phi(3), dR(1, 2, -2)),
     OpSum(Odphi(-1, -2))),

    # (phic phi) (dRc phic qL) -> Odphic
    (Op(phic(0), phi(0), dRc(1, 2, -2), phic(3), qL(1, 2, 3, -1)),
     OpSum(Odphic(-1, -2))),

    # (phic phi) (qLc epsSU2 phic uR) -> Ouphi
    (Op(phic(0), phi(0), qLc(1, 2, 3, -1), epsSU2(3, 4), phic(4), uR(1, 2, -2)),
     OpSum(Ouphi(-1, -2))),

    # (phic phi) (uRc qL epsSU2 phi) -> Ouphic
    (Op(phic(0), phi(0), uRc(1, 2, -2), qL(1, 2, 3, -1), epsSU2(3, 4), phi(4)),
     OpSum(Ouphic(-1, -2))),

    # Dphic eRc DlL -> OephiDaux
    (Op(D(0, phic(1)), eRc(2, -1), D(0, lL(2, 1, -2))),
     OpSum(OephiDaux(-1, -2))),

    # Dphi DlLc eR -> OephiDauxc
    (Op(D(0, phi(1)), D(0, lLc(2, 1, -2)), eR(2, -1)),
     OpSum(OephiDauxc(-1, -2))),

    # Dphic dRc DqL -> OdphiDaux
    (Op(D(0, phic(1)), dRc(2, 3, -1), D(0, qL(2, 3, 1, -2))),
     OpSum(OdphiDaux(-1, -2))),

    # Dphi DqLc dR -> OdphiDauxc
    (Op(D(0, phi(1)), D(0, qLc(2, 3, 1, -2)), dR(2, 3, -1)),
     OpSum(OdphiDauxc(-1, -2))),

    # epsSU2 Dphi uRc DqL -> OuphiDaux
    (Op(epsSU2(0, 1), D(2, phi(1)), uRc(3, 4, -1), D(2, qL(3, 4, 0, -2))),
     OpSum(OuphiDaux(-1, -2))),

    # epsSU2 Dphic DqLc uR -> OuphiDauxc
    (Op(epsSU2(0, 1), D(2, phic(1)), D(2, qLc(3, 4, 0, -2)), uR(3, 4, -1)),
     OpSum(OuphiDauxc(-1, -2))),

    # lL(0, 1, -1) epsSU2(1, 2) phi(2) epsUp(0, 3)
    # phi(4) epsSU2(5, 4) lL(3, 5, -2) -> O5
    (Op(lL(0, 1, -1), epsSU2(1, 2), phi(2), epsUp(0, 3),
        phi(4), epsSU2(5, 4), lL(3, 5, -2)),
     OpSum(O5(-1, -2))),

    # lLc(0, 1, -2) epsSU2(1, 2) phic(2) epsUp(3, 0)
    # phic(4) epsSU2(5, 4) lLc(3, 5, -1) -> O5c
    (Op(lLc(0, 1, -1), epsSU2(1, 2), phic(2), epsUpDot(0, 3),
        phic(4), epsSU2(5, 4), lLc(3, 5, -2)),
     -OpSum(O5c(-1, -2))),

    # lL(0, 1, -1) epsUp(0, 2) lL(2, 1, -2) phi(3) phi(3) -> O5aux
     (Op(lL(0, 1, -1), epsUp(0, 2), lL(2, 1, -2), phi(3), phi(3)),
      OpSum(O5aux(-1, -2))),

    # lLc(0, 1, -2) epsUpDot(0, 2) lLc(2, 1, -1) phic(3) phic(3) -> O5auxc
     (Op(lLc(0, 1, -2), epsUpDot(0, 2), lLc(2, 1, -1), phic(3), phic(3)),
      OpSum(O5auxc(-1, -2))),

    # (phic Dphi) (lLc gamma lL) -> -i O1phil
    (Op(phic(0), D(1, phi(0)), lLc(2, 3, -1), sigma4bar(1, 2, 4), lL(4, 3, -2)),
     OpSum(number_op(-1j) * O1phil(-1, -2))),

    # (Dphic phi) (lLc gamma lL) -> i O1philc
    (Op(D(1, phic(0)), phi(0), lLc(2, 3, -2), sigma4bar(1, 2, 4), lL(4, 3, -1)),
     OpSum(number_op(1j) * O1philc(-1, -2))),

    # (phic sigma Dphi) (lLc sigma gamma lL) -> -i O3phil
    (Op(phic(0), sigmaSU2(1, 0, 2), D(3, phi(2)),
        lLc(4, 5, -1), sigma4bar(3, 4, 6), sigmaSU2(1, 5, 7), lL(6, 7, -2)),
     OpSum(number_op(-1j) * O3phil(-1, -2))),

    # (Dphic sigma phi) (lLc sigma gamma lL) -> i O3philc
    (Op(D(3, phic(0)), sigmaSU2(1, 0, 2), phi(2),
        lLc(4, 5, -2), sigma4bar(3, 4, 6), sigmaSU2(1, 5, 7), lL(6, 7, -1)),
     OpSum(number_op(1j) * O3philc(-1, -2))),
    
    # (phic Dphi) (qLc gamma qL) -> -i O1phiq
    (Op(phic(0), D(1, phi(0)), qLc(2, 3, 4, -1), sigma4bar(1, 2, 5), qL(5, 3, 4, -2)),
     OpSum(number_op(-1j) * O1phiq(-1, -2))),

    # (Dphic phi) (qLc gamma qL) -> i O1phiqc
    (Op(D(1, phic(0)), phi(0), qLc(2, 3, 4, -2), sigma4bar(1, 2, 5), qL(5, 3, 4, -1)),
     OpSum(number_op(1j) * O1phiqc(-1, -2))),

    # (phic sigma Dphi) (qLc sigma gamma qL) -> -i O3phiq
    (Op(phic(0), sigmaSU2(1, 0, 2), D(3, phi(2)), qLc(4, 5, 6, -1),
        sigma4bar(3, 4, 7), sigmaSU2(1, 6, 8), qL(7, 5, 8, -2)),
     OpSum(number_op(-1j) * O3phiq(-1, -2))),

    # (phic sigma Dphi) (qLc sigma gamma qL) -> i O3phiqc
    (Op(D(3, phic(0)), sigmaSU2(1, 0, 2), phi(2), qLc(4, 5, 6, -2),
        sigma4bar(3, 4, 7), sigmaSU2(1, 6, 8), qL(7, 5, 8, -1)),
     OpSum(number_op(1j) * O3phiqc(-1, -2))),

    # (phic Dphi) (eRc gamma eR) -> -i O1phie
    (Op(phic(0), D(1, phi(0)), eRc(2, -1), sigma4(1, 2, 3), eR(3, -2)),
     OpSum(number_op(-1j) * O1phie(-1, -2))),

    # (Dphic phi) (eRc gamma eR) -> i O1phiec
    (Op(D(1, phic(0)), phi(0), eRc(2, -2), sigma4(1, 2, 3), eR(3, -1)),
     OpSum(number_op(1j) * O1phiec(-1, -2))),

    # (phic Dphi) (dRc gamma dR) -> -i O1phid
    (Op(phic(0), D(1, phi(0)), dRc(2, 3, -1), sigma4(1, 2, 4), dR(4, 3, -2)),
     OpSum(number_op(-1j) * O1phid(-1, -2))),

    # (Dphic phi) (dRc gamma dR) -> i O1phidc
    (Op(D(1, phic(0)), phi(0), dRc(2, 3, -2), sigma4(1, 2, 4), dR(4, 3, -1)),
     OpSum(number_op(1j) * O1phidc(-1, -2))),

    # (phic Dphi) (uRc gamma uR) -> -i O1phiu
    (Op(phic(0), D(1, phi(0)), uRc(2, 3, -1), sigma4(1, 2, 4), uR(4, 3, -2)),
     OpSum(number_op(-1j) * O1phiu(-1, -2))),

    # (Dphic phi) (uRc gamma uR) -> i O1phiuc
    (Op(D(1, phic(0)), phi(0), uRc(2, 3, -2), sigma4(1, 2, 4), uR(4, 3, -1)),
     OpSum(number_op(1j) * O1phiuc(-1, -2))),
    
    # (phi epsSU2 Dphi) (uRc gamma dR) -> -i Ophiud
    (Op(phi(0), epsSU2(0, 1), D(2, phi(1)),
        uRc(3, 4, -1), sigma4(2, 3, 5), dR(5, 4, -2)),
     OpSum(number_op(-1j) * Ophiud(-1, -2))),

    # (phic epsSU2 Dphic) (dRc gamma uR) -> i Ophiud
    (Op(phic(0), epsSU2(0, 1), D(2, phic(1)),
        dRc(3, 4, -2), sigma4(2, 3, 5), uR(5, 4, -1)),
     OpSum(number_op(1j) * Ophiudc(-1, -2))),

    # -- Four-fermion operators --

    # (lLc sigma4bar lL) (lLc simgaUpBar lL) -> 2 * O1ll
    (Op(lLc(0, 1, -1), sigma4bar(2, 0, 3), lL(3, 1, -2),
        lLc(4, 5, -3), sigma4bar(2, 4, 6), lL(6, 5, -4)),
     OpSum(number_op(2) * O1ll(-1, -2, -3, -4))),
    
    # (eRc sigma4 eR) (eRc sigma4 eR) -> 2 * Oee
    (Op(eRc(0, -1), sigma4(1, 0, 2), eR(2, -2),
        eRc(3, -3), sigma4(1, 3, 4), eR(4, -4)),
     OpSum(number_op(2) * Oee(-1, -2, -3, -4))),
    
    # (lLc sigma4bar lL) (eRc sigma4 eR) -> Ole
    (Op(lLc(0, 1, -1), sigma4bar(2, 0, 3), lL(3, 1, -2),
        eRc(4, -3), sigma4(2, 4, 5), eR(5, -4)),
     OpSum(Ole(-1, -2, -3, -4))),
    
    # (qLc sigma4bar qL) (dRc sigma4 dR) -> O1qd
    (Op(qLc(0, 1, 2, -1), sigma4bar(3, 0, 4), qL(4, 1, 2, -2),
        dRc(5, 6, -3), sigma4(3, 5, 7), dR(7, 6, -4)),
     OpSum(O1qd(-1, -2, -3, -4))),
    
    # (qLc sigma4bar qL) (uRc sigma4 uR) -> O1qu
    (Op(qLc(0, 1, 2, -1), sigma4bar(3, 0, 4), qL(4, 1, 2, -2),
               uRc(5, 6, -3), sigma4(3, 5, 7), uR(7, 6, -4)),
     OpSum(O1qu(-1, -2, -3, -4))),
    
    # (qLc sigma4bar lambdaColor qL) (dRc sigma4 lambdaColor dR) -> O1qd
    (Op(qLc(0, 1, 2, -1), sigma4bar(3, 0, 4),
        lambdaColor(5, 1, 6), qL(4, 6, 2, -2),
        dRc(6, 7, -3), sigma4(3, 6, 8),
        lambdaColor(5, 7, 9), dR(8, 9, -4)),
     OpSum(O8qd(-1, -2, -3, -4))),
    
    # (qLc sigma4bar lambdaColor qL) (uRc sigma4 lambdaColor uR) -> O1qu
    (Op(qLc(0, 1, 2, -1), sigma4bar(3, 0, 4),
        lambdaColor(5, 1, 6), qL(4, 6, 2, -2),
        uRc(6, 7, -3), sigma4(3, 6, 8),
        lambdaColor(5, 7, 9), uR(8, 9, -4)),
     OpSum(O8qu(-1, -2, -3, -4))),

    # (lLc eR) (dRc qL) -> Oledq
    (Op(lLc(0, 1, -1), eR(0, -2), dRc(2, 3, -3), qL(2, 3, 1, -4)),
     OpSum(Oledq(-1, -2, -3, -4))),

    # (eRc lL) (qLc dR) -> Oledqc
    (Op(eRc(0, -2), lL(0, 1, -1), qLc(2, 3, 1, -4), dR(2, 3, -3)),
     OpSum(Oledqc(-1, -2, -3, -4))),

    # (lLc eR) epsSU2 (qLc uR) -> Olequ
    (Op(lLc(0, 1, -1), eR(0, -2), epsSU2(1, 2), qLc(3, 4, 2, -3), uR(3, 4, -4)),
     OpSum(Olequ(-1, -2, -3, -4))),

    # (eRc lL) epsSU2 (uRc qL) -> Olequc
    (Op(eRc(0, -2), lL(0, 1, -1), epsSU2(1, 2), uRc(3, 4, -4), qL(3, 4, 2, -3)),
     OpSum(Olequc(-1, -2, -3, -4))),

    # (qLc uR) epsSU2 (qLc dR) -> O1qud
    (Op(qLc(0, 1, 2, -1), uR(0, 1, -2), epsSU2(2, 3),
        qLc(4, 5, 3, -3), dR(4, 5, -4)),
     OpSum(O1qud(-1, -2, -3, -4))),

    # (uRc qL) epsSU2 (dRc qL) -> O1qudc
    (Op(uRc(0, 1, -2), qL(0, 1, 2, -1), epsSU2(2, 3),
               dRc(4, 5, -4), qL(4, 5, 3, -3)),
     OpSum(O1qudc(-1, -2, -3, -4))),

    # -- SM operators --
    
    # Dphic Dphi -> Okinphi
    (Op(D(0, phic(1)), D(0, phi(1))),
     OpSum(Okinphi)),

    # (phic phi)^2 -> Ophi4
    (Op(phic(0), phi(0), phic(1), phi(1)),
     OpSum(Ophi4)),

    # phic phi -> Ophi2
    (Op(phic(0), phi(0)),
     OpSum(Ophi2)),
    
    # lLc phi eR -> Oye
    (Op(lLc(0, 1, -1), phi(1), eR(0, -2)),
     OpSum(Oye(-1, -2))),

    # eR phic lLc -> Oyec
    (Op(eRc(0, -2), phic(1), lL(0, 1, -1)),
     OpSum(Oyec(-1, -2))),

    # qLc phi dR -> Oyd
    (Op(qLc(0, 2, 1, -1), phi(1), dR(0, 2, -2)),
     OpSum(Oyd(-1, -2))),

    # dR phic qLc -> Oydc
    (Op(dRc(0, 2, -2), phic(1), qL(0, 2, 1, -1)),
     OpSum(Oydc(-1, -2))),

    # qLc phi uR -> Oyu
    (Op(qLc(0, 2, 1, -1), epsSU2(1, 3), phic(3), uR(0, 2, -2)),
     OpSum(Oyu(-1, -2))),

    # uR phic qLc -> Oyuc
    (Op(uRc(0, 2, -2), epsSU2(1, 3), phi(3), qL(0, 2, 1, -1)),
     OpSum(Oyuc(-1, -2)))
]

transpose_epsSU2 = [(Op(epsSU2(-1, -2)), OpSum(-Op(epsSU2(-2, -1))))]

all_rules = rules + Ofphi2Ophif + SM_eoms + definitions + transpose_epsSU2

op_names = [
    "Okinphi", "Ophi4", "Ophi2", "Oye", "Oyec", "Oyd", "Oydc", "Oyu", "Oyuc",

    "O5", "O5c", "O5aux", "O5auxc",
    
    "Ophi", "OphiD", "Ophisq",
    
    "OphiB", "OWB", "OphiW", "OphiBTilde", "OWBTilde", "OphiWTilde",
    
    "Oephi", "Oephic", "Odphi", "Odphic", "Ouphi", "Ouphic",

    "OephiDaux", "OephiDauxc", "OdphiDaux", "OdphiDauxc", "OuphiDaux", "OuphiDauxc",
    
    "O1phil", "O1philc", "O1phiq", "O1phiqc", "O3phil", "O3philc",
    "O3phiq", "O3phiqc", "O1phie", "O1phiec", "O1phid", "O1phidc",
    "O1phiu", "O1phiuc", "Ophiud", "Ophiudc",
    
    "O1ll", "Oee", "Ole", "O1qd", "O1qu", "O8qd", "O8qu",
    "Oledq", "Olequ", "O1qud", "Oledqc", "Olequc", "O1qudc",
]

print "Applying rules...",
sys.stdout.flush()
final_lag = apply_rules_until(eff_lag, all_rules, op_names, 12)
print "done."

structures = {
    "deltaFlavor": "\\delta_{{{}{}}}",

    # Standard Model
    "gb": "g'",
    "gw": "g",
    "mu2phi": "\\mu^2_\\phi",
    "lambdaphi": "\\lambda_\\phi",
    "ye": "\\delta_{{{0}{1}}}y^e_{{{0}{0}}}",
    "yec": "\\delta_{{{0}{1}}}y^{{e*}}_{{{0}{0}}}",
    "yd": "\\delta_{{{0}{1}}}y^d_{{{0}{0}}}",
    "ydc": "\\delta_{{{0}{1}}}y^{{d*}}_{{{0}{0}}}",
    "yu": "y^u_{{{}{}}}",
    "yuc": "y^{{u*}}_{{{}{}}}",
    "V": "V_{{{}{}}}",
    "Vc": "V^\\dagger_{{{1}{0}}}",

    # Quarks
    "lambdaP1": "\\lambda'^{{(1)}}_{{{}{}}}",
    "lambdaP1c": "\\lambda'^{{(1)*}}_{{{}{}}}",
    "lambdaP2": "\\lambda'^{{(2)}}_{{{}{}}}",
    "lambdaP2c": "\\lambda'^{{(2)*}}_{{{}{}}}",
    "lambdaP3u": "\\lambda'^{{(3u)}}_{{{}{}}}",
    "lambdaP3uc": "\\lambda'^{{(3u)*}}_{{{}{}}}",
    "lambdaP3d": "\\lambda'^{{(3d)}}_{{{}{}}}",
    "lambdaP3dc": "\\lambda'^{{(3d)*}}_{{{}{}}}",
    "lambdaP4": "\\lambda'^{{(4)}}_{{{}{}}}",
    "lambdaP4c": "\\lambda'^{{(4)*}}_{{{}{}}}",
    "lambdaP5": "\\lambda'^{{(5)}}_{{{}{}}}",
    "lambdaP5c": "\\lambda'^{{(5)*}}_{{{}{}}}",
    "lambdaP6": "\\lambda'^{{(6)}}_{{{}{}}}",
    "lambdaP6c": "\\lambda'^{{(6)*}}_{{{}{}}}",
    "lambdaP7": "\\lambda'^{{(7)}}_{{{}{}}}",
    "lambdaP7c": "\\lambda'^{{(7)*}}_{{{}{}}}",

    # Leptons
    "lambdaDelta1e": "\\left(\\lambda^{{\\Delta_1}}_{{e}}\\right)_{{{}{}}}",
    "lambdaDelta1ec": "\\left(\\lambda^{{\\Delta_1}}_{{e}}\\right)^*_{{{}{}}}",
    "lambdaDelta3e": "\\left(\\lambda^{{\\Delta_3}}_{{e}}\\right)_{{{}{}}}",
    "lambdaDelta3ec": "\\left(\\lambda^{{\\Delta_3}}_{{e}}\\right)^*_{{{}{}}}",
    "lambdaNRl": "\\left(\\lambda^{{N_R}}_{{l}}\\right)_{{{}{}}}",
    "lambdaNRlc": "\\left(\\lambda^{{N_R}}_{{l}}\\right)^*_{{{}{}}}",
    "lambdaNLl": "\\left(\\lambda^{{N_L}}_{{l}}\\right)_{{{}{}}}",
    "lambdaNLlc": "\\left(\\lambda^{{N_L}}_{{l}}\\right)^*_{{{}{}}}",
    "lambdaNmajl": "\\left(\\lambda^{{N^{{(maj)}}}}_{{l}}\\right)_{{{}{}}}",
    "lambdaNmajlc": "\\left(\\lambda^{{N^{{(maj)}}}}_{{l}}\\right)^*_{{{}{}}}",
    "lambdaEl": "\\left(\\lambda^{{E}}_{{l}}\\right)_{{{}{}}}",
    "lambdaElc": "\\left(\\lambda^{{E}}_{{l}}\\right)^*_{{{}{}}}",
    "lambdaSigma0Rl": "\\left(\\lambda^{{\\Sigma_{{0R}}}}_{{l}}\\right)_{{{}{}}}",
    "lambdaSigma0Rlc": "\\left(\\lambda^{{\\Sigma_{{0R}}}}_{{l}}\\right)^*_{{{}{}}}",
    "lambdaSigma0Ll": "\\left(\\lambda^{{\\Sigma_{{0L}}}}_{{l}}\\right)_{{{}{}}}",
    "lambdaSigma0Llc": "\\left(\\lambda^{{\\Sigma_{{0L}}}}_{{l}}\\right)^*_{{{}{}}}",
    "lambdaSigma0majl":
    "\\left(\\lambda^{{\\Sigma^{{(maj)}}_{{0}}}}_{{l}}\\right)_{{{}{}}}",
    "lambdaSigma0majlc":
    "\\left(\\lambda^{{\\Sigma^{{(maj)}}_{{0}}}}_{{l}}\\right)^*_{{{}{}}}",
    "lambdaSigma1l": "\\left(\\lambda^{{\\Sigma_1}}_{{l}}\\right)_{{{}{}}}",
    "lambdaSigma1lc": "\\left(\\lambda^{{\\Sigma_1}}_{{l}}\\right)^*_{{{}{}}}",

    # Scalars
    "kappaS": "\\kappa_{{\\mathcal{{S}}_{}}}",
    "lambdaS": "\\lambda^{{{}{}}}_{{\\mathcal{{S}}}}",
    "kappaS3": "\\kappa^{{{}{}{}}}_{{\\mathcal{{S}}^3}}",
    "kappaXi0": "\\kappa_{{\\Xi_{{0{}}}}}",
    "lambdaXi0": "\\lambda^{{{}{}}}_{{\\Xi_0}}",
    "ylS1": "\\left(y^l_{{\\mathcal{{S}}_{{1{}}}}}\\right)_{{{}{}}}",
    "ylS1c": "\\left(y^l_{{\mathcal{{S}}_{{1{}}}}}\\right)^*_{{{}{}}}",
    "yeS2": "\\left(y^e_{{\\mathcal{{S}}_{{2{}}}}}\\right)_{{{}{}}}",
    "yeS2c": "\\left(y^e_{{\mathcal{{S}}_{{2{}}}}}\\right)^*_{{{}{}}}",
    "yevarphi": "\\left(y^e_{{\\varphi_{{{}}}}}\\right)_{{{}{}}}",
    "yevarphic": "\\left(y^e_{{\\varphi_{{{}}}}}\\right)^*_{{{}{}}}",
    "ydvarphi": "\\left(y^d_{{\\varphi_{{{}}}}}\\right)_{{{}{}}}",
    "ydvarphic": "\\left(y^d_{{\\varphi_{{{}}}}}\\right)^*_{{{}{}}}",
    "yuvarphi": "\\left(y^u_{{\\varphi_{{{}}}}}\\right)_{{{}{}}}",
    "yuvarphic": "\\left(y^u_{{\\varphi_{{{}}}}}\\right)^*_{{{}{}}}",
    "lambdavarphi": "\\lambda_{{\\varphi_{}}}",
    "lambdavarphic": "\\lambda^*_{{\\varphi_{}}}",
    "ylXi1": "\\left(y^l_{{\\Xi_{{1{}}}}}\\right)_{{{}{}}}",
    "ylXi1c": "\\left(y^l_{{\\Xi_{{1{}}}}}\\right)^*_{{{}{}}}",
    "kappaXi1": "\\kappa_{{\\Xi_{{1{}}}}}",
    "kappaXi1c": "\\kappa^*_{{\\Xi_{{1{}}}}}",
    "lambdaXi1": "\\lambda^{{{}{}}}_{{\\Xi_1}}",
    "lambdaXi1c": "\\left(\\lambda^{{{}{}}}_{{\\Xi_1}}\right)^*",
    "lambdaTildeXi1": "\\tilde{{\\lambda}}^{{{}{}}}_{{\\Xi_1}}",
    "lambdaTildeXi1c": "\\left(\\tilde{{\\lambda}}^{{{}{}}}_{{\\Xi_1}}\\right)^*",
    "kappaSXi0": "\\kappa^{{{}{}{}}}_{{\\mathcal{{S}}\\Xi_0}}",
    "kappaSXi1": "\\kappa^{{{}{}{}}}_{{\\mathcal{{S}}\\Xi_1}}",
    "kappaXi0Xi1": "\\kappa^{{{}{}{}}}_{{\\Xi_0\\Xi_1}}",
    "lambdaSXi0": "\\lambda^{{{}{}}}_{{\\mathcal{{S}}\\Xi_0}}",
    "kappaSvarphi": "\\kappa^{{{}{}}}_{{\\mathcal{{S}}\\varphi}}",
    "kappaSvarphic": "\\left(\\kappa^{{{}{}}}_{{\mathcal{{S}}\\varphi}}\\right)^*",
    "kappaXi0varphi": "\\kappa^{{{}{}}}_{{\\Xi_0\\varphi}}",
    "kappaXi0varphic": "\\left(\\kappa^{{{}{}}}_{{\\Xi_0\\varphi}}\\right)^*",
    "kappaXi1varphi": "\\kappa^{{{}{}}}_{{\\Xi_1\\varphi}}",
    "kappaXi1varphic": "\\left(\\kappa^{{{}{}}}_{{\\Xi_1\\varphi}}\\right)^*",
    "lambdaSXi1": "\\lambda^{{{}{}}}_{{\\mathcal{{S}}\\Xi_1}}",
    "lambdaSXi1c": "\\left(\\lambda^{{{}{}}}_{{\\mathcal{{S}}\\Xi_1}}\\right)^*",
    "lambdaXi1Xi0": "\\lambda^{{{}{}}}_{{\\Xi_1\\Xi_0}}",
    "lambdaXi1Xi0c": "\\left(\\lambda^{{{}{}}}_{{\\Xi_1\\Xi_0}}\right)^*",

    # Vectors 1
    "glB": "\\left(g^l_{{\mathcal{{B}}_{{{}}}}}\\right)_{{{}{}}}",
    "gqB": "\\left(g^{{q}}_{{\\mathcal{{B}}}}\\right)_{{{}{}}}",
    "geB": "\\left(g^{{e}}_{{\\mathcal{{B}}}}\\right)_{{{}{}}}",
    "gdB": "\\left(g^{{d}}_{{\\mathcal{{B}}}}\\right)_{{{}{}}}",
    "guB": "\\left(g^{{u}}_{{\\mathcal{{B}}}}\\right)_{{{}{}}}",
    "gphiB": "g^{{\\phi}}_{{\\mathcal{{B}}_{{{}}}}}",
    "gphiBc": "g^{{\\phi *}}_{{\\mathcal{{B}}_{{{}}}}}",
    "glW": "\\left(g^{{l}}_{{\\mathcal{{W}}}}\\right)_{{{}{}}}",
    "gqW": "\\left(g^{{q}}_{{\\mathcal{{W}}}}\\right)_{{{}{}}}",
    "gphiW": "g^{{\\phi}}_{{\\mathcal{{W}}_{{{}}}}}",
    "gphiWc": "g^{{\\phi *}}_{{\\mathcal{{W}}_{{{}}}}}",
    "gduB1": "\\left(g^{{du}}_{{\\mathcal{{B}}^1}}\\right)_{{{}{}}}",
    "gduB1c": "\\left(g^{{du}}_{{\\mathcal{{B}}^1}}\\right)^*_{{{}{}}}",
    "gphiB1": "g^{{\\phi}}_{{\\mathcal{{B}}^1_{{{}}}}}",
    "gphiB1c": "g^{{\\phi *}}_{{\\mathcal{{B}}^1_{{{}}}}}",
    "gphiW1": "g^{{\\phi}}_{{\\mathcal{{W}}^1_{{{}}}}}",
    "gphiW1c": "g^{{\\phi *}}_{{\\mathcal{{W}}^1_{{{}}}}}",

    # Vectors 2
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
    "gB": "g^{{{}{}}}_{{\\mathcal{{B}}}}",
    "gW": "g^{{{}{}}}_{{\\mathcal{{W}}}}",
    "gTildeB": "\\tilde{{g}}^{{{}{}}}_{{\\mathcal{{B}}}}",
    "gTildeW": "\\tilde{{g}}^{{{}{}}}_{{\\mathcal{{W}}}}",
    "h1": "h^{{(1)}}_{{{}{}}}",
    "h2": "h^{{(2)}}_{{{}{}}}",
    "h3": "h^{{(3)}}_{{{}{}}}",
    "h3c": "h^{{(3)*}}_{{{}{}}}",

    # VDS
    "deltaB": "\\delta^{{{}{}}}_{{\\mathcal{{B}}}}",
    "deltaW": "\\delta^{{{}{}}}_{{\\mathcal{{W}}}}",
    "deltaL1": "\\delta^{{{}{}}}_{{\\mathcal{{L}}^1}}",
    "deltaL1c": "\\delta^{{{}{}*}}_{{\\mathcal{{L}}^1}}",
    "deltaW1": "\\delta^{{{}{}}}_{{\\mathcal{{W}}^1}}",
    "deltaW1c": "\\delta^{{{}{}*}}_{{\\mathcal{{W}}^1}}",

    # VVS
    "epsilonS": "\\varepsilon^{{{}{}{}}}_{{\\mathcal{{S}}}}",
    "epsilonXi0": "\\varepsilon^{{{}{}{}}}_{{\\Xi_0}}",
    "epsilonXi1": "\\varepsilon^{{{}{}{}}}_{{\\Xi_1}}",
    "epsilonXi1c": "\\varepsilon^{{{}{}{}*}}_{{\\Xi_1}}",

    # VS(SM)
    "g1Xi1L1": "g^{{(1)}}_{{\\Xi_{{1{}}}\\mathcal{{L}}^1_{{{}}}}}",
    "g1Xi1L1c": "g^{{(1)*}}_{{\\Xi_{{1{}}}\\mathcal{{L}}^1_{{{}}}}}",
    "g1Xi0L1": "g^{{(1)}}_{{\\Xi_{{0{}}}\\mathcal{{L}}^1_{{{}}}}}",
    "g1Xi0L1c": "g^{{(1)*}}_{{\\Xi_{{0{}}}\\mathcal{{L}}^1_{{{}}}}}",
    "g1SL1": "g^{{(1)}}_{{\\mathcal{{S}}_{{{}}}\\mathcal{{L}}^1_{{{}}}}}",
    "g1SL1c": "g^{{(1)*}}_{{\\mathcal{{S}}_{{{}}}\\mathcal{{L}}^1_{{{}}}}}",
    "g2Xi1L1": "g^{{(2)}}_{{\\Xi_{{1{}}}\\mathcal{{L}}^1_{{{}}}}}",
    "g2Xi1L1c": "g^{{(2)*}}_{{\\Xi_{{1{}}}\\mathcal{{L}}^1_{{{}}}}}",
    "g2Xi0L1": "g^{{(2)}}_{{\\Xi_{{0{}}}\\mathcal{{L}}^1_{{{}}}}}",
    "g2Xi0L1c": "g^{{(2)*}}_{{\\Xi_{{0{}}}\\mathcal{{L}}^1_{{{}}}}}",
    "g2SL1": "g^{{(2)}}_{{\\mathcal{{S}}_{{{}}}\\mathcal{{L}}^1_{{{}}}}}",
    "g2SL1c": "g^{{(2)*}}_{{\\mathcal{{S}}_{{{}}}\\mathcal{{L}}^1_{{{}}}}}",

    # VF(SM)
    "zSigma0L": "z^{{\\Sigma_{{0L}}}}_{{{}{}{}}}",
    "zSigma0Lc": "z^{{\\Sigma_{{0L}}*}}_{{{}{}{}}}",
    "zSigma0R": "z^{{\\Sigma_{{0R}}}}_{{{}{}{}}}",
    "zSigma0Rc": "z^{{\\Sigma_{{0R}}*}}_{{{}{}{}}}",
    "zSigma0maj": "z^{{\\Sigma^{{(maj)}}_{{0}}}}_{{{}{}{}}}",
    "zSigma0majc": "z^{{\\Sigma^{{(maj)}}_{{0}}*}}_{{{}{}{}}}",
    "zSigma1": "z^{{\\Sigma_1}}_{{{}{}{}}}",
    "zSigma1c": "z^{{\\Sigma_1*}}_{{{}{}{}}}",
    "zDelta1": "z^{{\\Delta_1}}_{{{}{}{}}}",
    "zDelta1c": "z^{{\\Delta_1*}}_{{{}{}{}}}",
    "zDelta3": "z^{{\\Delta_3}}_{{{}{}{}}}",
    "zDelta3c": "z^{{\\Delta_3*}}_{{{}{}{}}}",
    "zNL": "z^{{N_L}}_{{{}{}{}}}",
    "zNLc": "z^{{N_L*}}_{{{}{}{}}}",
    "zNR": "z^{{N_R}}_{{{}{}{}}}",
    "zNRc": "z^{{N_L*}}_{{{}{}{}}}",
    "zNmaj": "z^{{N^{{maj}}}}_{{{}{}{}}}",
    "zNmajc": "z^{{N^{{(maj)}}*}}_{{{}{}{}}}",
    "zE": "z^E_{{{}{}{}}}",
    "zEc": "z^{{E*}}_{{{}{}{}}}",
    "zXUD": "z^{{XUD}}_{{{}{}{}}}",
    "zXUDc": "z^{{XUD*}}_{{{}{}{}}}",
    "zUDY": "z^{{UDY}}_{{{}{}{}}}",
    "zUDYc": "z^{{UDY*}}_{{{}{}{}}}",
    "zUDu": "z^{{UD}}_{{u,{}{}{}}}",
    "zUDuc": "z^{{UD*}}_{{u,{}{}{}}}",
    "zUDd": "z^{{UD}}_{{d,{}{}{}}}",
    "zUDdc": "z^{{UD}}_{{d,{}{}{}}}",
    "zXU": "z^{{XU}}_{{{}{}{}}}",
    "zXUc": "z^{{XU*}}_{{{}{}{}}}",
    "zDY": "z^{{DY}}_{{{}{}{}}}",
    "zDYc": "z^{{DY*}}_{{{}{}{}}}",
    "zU": "z^{{U}}_{{{}{}{}}}",
    "zUc": "z^{{U*}}_{{{}{}{}}}",
    "zD": "z^{{D}}_{{{}{}{}}}",
    "zDc": "z^{{D*}}_{{{}{}{}}}",

    # SF(SM)
    "wuS": "\\left(w^u_{{\\mathcal{{S}}_{{{}}}}}\\right)_{{{}{}}}",
    "wuSc": "\\left(w^u_{{\\mathcal{{S}}_{{{}}}}}\\right)^*_{{{}{}}}",
    "wdS": "\\left(w^d_{{\\mathcal{{S}}_{{{}}}}}\\right)_{{{}{}}}",
    "wdSc": "\\left(w^d_{{\\mathcal{{S}}_{{{}}}}}\\right)^*_{{{}{}}}",
    "weS": "\\left(w^e_{{\\mathcal{{S}}_{{{}}}}}\\right)_{{{}{}}}",
    "weSc": "\\left(w^e_{{\\mathcal{{S}}_{{{}}}}}\\right)^*_{{{}{}}}",
    "wqS": "\\left(w^q_{{\\mathcal{{S}}_{{{}}}}}\\right)_{{{}{}}}",
    "wqSc": "\\left(w^q_{{\\mathcal{{S}}_{{{}}}}}\\right)^*_{{{}{}}}",
    "wlS": "\\left(w^l_{{\\mathcal{{S}}_{{{}}}}}\\right)_{{{}{}}}",
    "wlSc": "\\left(w^l_{{\\mathcal{{S}}_{{{}}}}}\\right)^*_{{{}{}}}",
    "wuXi0": "\\left(w^u_{{\\Xi_{{0{}}}}}\\right)_{{{}{}}}",
    "wuXi0c": "\\left(w^u_{{\\Xi_{{0{}}}}}\\right)^*_{{{}{}}}",
    "wdXi0": "\\left(w^d_{{\\Xi_{{0{}}}}}\\right)_{{{}{}}}",
    "wdXi0c": "\\left(w^d_{{\\Xi_{{0{}}}}}\\right)^*_{{{}{}}}",
    "weXi0": "\\left(w^e_{{\\Xi_{{0{}}}}}\\right)_{{{}{}}}",
    "weXi0c": "\\left(w^e_{{\\Xi_{{0{}}}}}\\right)^*_{{{}{}}}",
    "wqXi0": "\\left(w^q_{{\\Xi_{{0{}}}}}\\right)_{{{}{}}}",
    "wqXi0c": "\\left(w^q_{{\\Xi_{{0{}}}}}\\right)^*_{{{}{}}}",
    "wlXi0": "\\left(w^l_{{\\Xi_{{0{}}}}}\\right)_{{{}{}}}",
    "wlXi0c": "\\left(w^l_{{\\Xi_{{0{}}}}}\\right)^*_{{{}{}}}",
    "wuXi1": "\\left(w^u_{{\\Xi_{{1{}}}}}\\right)_{{{}{}}}",
    "wuXi1c": "\\left(w^u_{{\\Xi_{{1{}}}}}\\right)^*_{{{}{}}}",
    "wdXi1": "\\left(w^d_{{\\Xi_{{1{}}}}}\\right)_{{{}{}}}",
    "wdXi1c": "\\left(w^d_{{\\Xi_{{1{}}}}}\\right)^*_{{{}{}}}",
    "weSigma0LXi1":
    "\\left(w^{{e\\Sigma_{{0L}}}}_{{\\Xi_{{1{}}}}}\\right)_{{{}{}}}",
    "weSigma0LXi1c":
    "\\left(w^{{e\\Sigma_{{0L}}}}_{{\\Xi_{{1{}}}}}\\right)^*_{{{}{}}}",
    "weSigma0RXi1": "\\left(w^{{e\\Sigma_{{0R}}}}_{{\\Xi_{{1{}}}}}\\right)_{{{}{}}}",
    "weSigma0RXi1c":
    "\\left(w^{{e\\Sigma_{{0R}}}}_{{\\Xi_{{1{}}}}}\\right)^*_{{{}{}}}",
    "weSigma0majXi1":
    "\\left(w^{{e\\Sigma^{{(maj)}}_{{0}}}}_{{\\Xi_{{1{}}}}}\\right)_{{{}{}}}",
    "weSigma0majXi1c":
    "\\left(w^{{e\\Sigma^{{(maj)}}_{{0}}}}_{{\\Xi_{{1{}}}}}\\right)^*_{{{}{}}}",
    "wl3Xi1": "\\left(w^{{l(3)}}_{{\\Xi_{{1{}}}}}\\right)_{{{}{}}}",
    "wl3Xi1c": "\\left(w^{{l(3)}}_{{\\Xi_{{1{}}}}}\\right)^*_{{{}{}}}",
    "wq7Xi1": "\\left(w^{{l(7)}}_{{\\Xi_{{1{}}}}}\\right)_{{{}{}}}",
    "wq7Xi1c": "\\left(w^{{l(7)}}_{{\\Xi_{{1{}}}}}\\right)^*_{{{}{}}}",
    "wq5Xi1": "\\left(w^{{l(5)}}_{{\\Xi_{{1{}}}}}\\right)_{{{}{}}}",
    "wq5Xi1c": "\\left(w^{{l(5)}}_{{\\Xi_{{1{}}}}}\\right)^*_{{{}{}}}",

    "MS": "M_{{\\mathcal{{S}}_{}}}",
    "MXi0": "M_{{\\Xi_{{0{}}}}}",
    "MXi1": "M_{{\\Xi_{{1{}}}}}",
    "Mvarphi": "M_{{\\varphi_{}}}",
    
    "MB": "M_{{\\mathcal{{B}}_{}}}",
    "MW": "M_{{\\mathcal{{W}}_{}}}",
    "MB1": "M_{{\\mathcal{{B}}^1_{}}}",
    "MW1": "M_{{\\mathcal{{W}}^1_{}}}",
    "ML1": "M_{{\\mathcal{{L}}^1_{}}}",

    "MN": "M_{{N_{}}}",
    "ME": "M_{{E_{}}}",
    "MDelta1": "M_{{\\Delta_{{1{}}}}}",
    "MDelta3": "M_{{\\Delta_{{3{}}}}}",
    "MSigma0": "M_{{\\Sigma_{{0{}}}}}",
    "MSigma1": "M_{{\\Sigma_{{1{}}}}}",
    "MNmaj": "M_{{N^{{(maj)}}_{}}}",
    "MSigma0maj": "M_{{\\Sigma^{{(maj)}}_{{0{}}}}}",

    "MU": "M_{{U_{}}}",
    "MD": "M_{{D_{}}}",
    "MXU": "M_{{XU_{}}}",
    "MUD": "M_{{UD_{}}}",
    "MDY": "M_{{DY_{}}}",
    "MXUD": "M_{{XUD_{}}}",
    "MUDY": "M_{{UDY_{}}}",

    "phi": "\\phi_{}",
    "phic": "\\phi^*_{}",
    "lL": "l_{{L{}{}{}}}",
    "lLc": "l^c_{{L\\dot{{{}}}{}{}}}",
    "qL": "q_{{L{}{}{}{}}}",
    "qLc": "q^c_{{L\\dot{{{}}}{}{}{}}}",
    "eR": "e^{{\\dot{{{}}}}}_{{R{}}}",
    "eRc": "e^{{c{}}}_{{R{}}}",
    "dR": "d^{{\\dot{{{}}}}}_{{R{}{}}}",
    "dRc": "d^{{c{}}}_{{R{}{}}}",
    "uR": "u^{{\\dot{{{}}}}}_{{R{}{}}}",
    "uRc": "u^{{c{}}}_{{R{}{}}}",
    "bFS": "B_{{{}{}}}",
    "wFS": "W_{{{}{}}}",

    "epsSU2": "i(\\sigma_2)_{{{}{}}}",
    "sigmaSU2": "\\sigma^{}_{{{}{}}}",
    "fSU2": "f_{{{}{}{}}}",

    "lambdaColor": "\\lambda^{}_{{{}{}}}",

    "eps4": "\\epsilon_{{{}{}{}{}}}",

    "epsUp": "\\epsilon^{{{}{}}}",
    "epsUpDot": "\\epsilon^{{\\dot{{{}}}\\dot{{{}}}}}",
    "epsDown": "\\epsilon_{{{}{}}}",
    "epsDownDot": "\\epsilon_{{\\dot{{{}}}\\dot{{{}}}}}",
    "sigma4bar": "\\bar{{\\sigma}}_{{4{}}}^{{\\dot{{{}}}{}}}",
    "sigma4": "\\sigma^{{4{}}}_{{{}\\dot{{{}}}}}",
    "deltaUpDown": "\\delta^{}_{}",
    "deltaUpDownDot": "\\delta_{{\\dot{{{}}}}}^{{\\dot{{{}}}}}",
}



op_reps = {
    "Okinphi": "\\alpha_{kin,\\phi}", 
    "Ophi4": "\\alpha_{\\phi 4}",
    "Ophi2": "\\alpha_{\\phi 2}",
    "Oye": "\\left(\\alpha_{{y^e}}\\right)_{ij}",
    "Oyec": "\\left(\\alpha_{{y^e}}\\right)^*_{ij}",
    "Oyd": "\\left(\\alpha_{{y^d}}\\right)_{ij}",
    "Oydc": "\\left(\\alpha_{{y^d}}\\right)^*_{ij}",
    "Oyu": "\\left(\\alpha_{{y^u}}\\right)_{ij}",
    "Oyuc": "\\left(\\alpha_{{y^u}}\\right)^*_{ij}",

    "O5": "\\frac{\\left(\\alpha_5\\right)_{}}{\\Lambda}",
    "O5c": "\\frac{\\left(\\alpha_5\right)^*_{ij}}{\\Lambda}",
    "O5aux": "\\frac{\\left(\\alpha^{{(aux)}}_5\right)_{ij}}{\\Lambda}",
    "O5auxc": "\\frac{\\left(\\alpha^{{(aux)}}_5\right)^*_{ij}}{\\Lambda}",
    
    "Ophi": "\\frac{\\alpha_\\phi}{\\Lambda^2}",
    "OphiD": "\\frac{\\alpha_{\\phi D}}{\\Lambda^2}",
    "Ophisq": "\\frac{\\alpha_{\\phi\\square}}{\\Lambda^2}",
    
    "OphiB": "\\frac{\\alpha_{\\phi B}}{\\Lambda^2}",
    "OWB": "\\frac{\\alpha_{WB}}{\\Lambda^2}",
    "OphiW": "\\frac{\\alpha_{\\phi W}}{\\Lambda^2}",
    "OphiBTilde": "\\frac{\\alpha_{\\phi\\tilde{B}}}{\\Lambda^2}",
    "OWBTilde": "\\frac{\\alpha_{W\\tilde{B}}}{\\Lambda^2}",
    "OphiWTilde": "\\frac{\\alpha_{\\phi\\tilde{W}}}{\\Lambda^2}",
    
    "Oephi": "\\frac{\\left(\\alpha_{e\\phi}\\right)_{ij}}{\\Lambda^2}",
    "Oephic": "\\frac{\\left(\\alpha_{e\\phi}\\right)^*_{ij}}{\\Lambda^2}",
    "Odphi": "\\frac{\\left(\\alpha_{d\\phi}\\right)_{ij}}{\\Lambda^2}",
    "Odphic": "\\frac{\\left(\\alpha_{d\\phi}\\right)^*_{ij}}{\\Lambda^2}",
    "Ouphi": "\\frac{\\left(\\alpha_{u\\phi}\\right)_{ij}}{\\Lambda^2}",
    "Ouphic": "\\frac{\\left(\\alpha_{u\\phi}\\right)^*_{ij}}{\\Lambda^2}",

    "O1phil":
    "\\frac{\\left(\\alpha^{(1)}_{\\phi l}\\right)_{ij}}{\\Lambda^2}",
    "O1philc":
    "\\frac{\\left(\\alpha^{(1)}_{\\phi l}\\right)^*_{ij}}{\\Lambda^2}", 
    "O3phil":
    "\\frac{\\left(\\alpha^{(3)}_{\\phi l}\\right)_{ij}}{\\Lambda^2}",
    "O3philc":
    "\\frac{\\left(\\alpha^{(3)}_{\\phi l}\\right)^*_{ij}}{\\Lambda^2}",
    "O1phiq":
    "\\frac{\\left(\\alpha^{(1)}_{\\phi q}\\right)_{ij}}{\\Lambda^2}",
    "O1phiqc":
    "\\frac{\\left(\\alpha^{(1)}_{\\phi q}\\right)^*_{ij}}{\\Lambda^2}", 
    "O3phiq":
    "\\frac{\\left(\\alpha^{(3)}_{\\phi q}\\right)_{ij}}{\\Lambda^2}",
    "O3phiqc":
    "\\frac{\\left(\\alpha^{(3)}_{\\phi q}\\right)^*_{ij}}{\\Lambda^2}",
    "O1phie":
    "\\frac{\\left(\\alpha^{(1)}_{\\phi e}\\right)_{ij}}{\\Lambda^2}",
    "O1phiec":
    "\\frac{\\left(\\alpha^{(1)}_{\\phi e}\\right)^*_{ij}}{\\Lambda^2}", 
    "O1phid":
    "\\frac{\\left(\\alpha^{(1)}_{\\phi d}\\right)_{ij}}{\\Lambda^2}",
    "O1phidc":
    "\\frac{\\left(\\alpha^{(1)}_{\\phi d}\\right)^*_{ij}}{\\Lambda^2}",
    "O1phiu":
    "\\frac{\\left(\\alpha^{(1)}_{\\phi u}\\right)_{ij}}{\\Lambda^2}",
    "O1phiuc":
    "\\frac{\\left(\\alpha^{(1)}_{\\phi u}\\right)^*_{ij}}{\\Lambda^2}",
    "Ophiud":
    "\\frac{\\left(\\alpha_{\\phi ud}\\right)_{ij}}{\\Lambda^2}",
    "Ophiudc":
    "\\frac{\\left(\\alpha_{\\phi ud}\\right)^*_{ij}}{\\Lambda^2}", 
    
    "O1ll": "\\frac{\\left(\\alpha^{(1)}_{ll}\\right)_{ijkl}}{\\Lambda^2}",
    "Oee": "\\frac{\\left(\\alpha_{ee}\\right)_{ijkl}}{\\Lambda^2}",
    "Ole": "\\frac{\\left(\\alpha_{le}\\right)_{ijkl}}{\\Lambda^2}",
    "O1qd": "\\frac{\\left(\\alpha^{(1)}_{qd}\\right)_{ijkl}}{\\Lambda^2}",
    "O1qu": "\\frac{\\left(\\alpha^{(1)}_{qu}\\right)_{ijkl}}{\\Lambda^2}",
    "O8qd": "\\frac{\\left(\\alpha^{(8)}_{qd}\\right)_{ijkl}}{\\Lambda^2}",
    "O8qu": "\\frac{\\left(\\alpha^{(8)}_{qu}\\right)_{ijkl}}{\\Lambda^2}",
    "Oledq": "\\frac{\\left(\\alpha_{ledq}\\right)_{ijkl}}{\\Lambda^2}",
    "Olequ": "\\frac{\\left(\\alpha_{lequ}\\right)_{ijkl}}{\\Lambda^2}",
    "O1qud": "\\frac{\\left(\\alpha^{(1)}_{qud}\\right)_{ijkl}}{\\Lambda^2}",
    "Oledqc": "\\frac{\\left(\\alpha_{ledq}\\right)^*_{ijkl}}{\\Lambda^2}",
    "Olequc": "\\frac{\\left(\\alpha_{lequ}\\right)^*_{ijkl}}{\\Lambda^2}",
    "O1qudc": "\\frac{\\left(\\alpha^{(1)}_{qud}\\right)^*_{ijkl}}{\\Lambda^2}"
}

final_lag_writer = Writer(final_lag, op_reps.keys())

print final_lag_writer

final_lag_writer.show_pdf("mixed", "open", structures, op_reps,
                          ["i", "j", "k", "l", "m", "n", "p", "q",
                           "a", "b", "c", "d", "e", "f", "g", "h"])


