from operators import (
    Tensor, Op, OpSum,
    TensorBuilder, FieldBuilder, D,
    number_op, symbol_op, tensor_op, flavor_tensor_op,
    boson, fermion, kdelta,
    sigma4, sigma4bar, epsUp, epsUpDot, epsDown, epsDownDot)

from transformations import (
    collect_numbers_and_symbols, collect_by_tensors,
    apply_rules_until)

from integration import (
    integrate, RealScalar, ComplexScalar, RealVector, ComplexVector, VectorLikeFermion)

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

# -- Integration --

print "Integrating...",
vectors_eff_lag = integrate(heavy_vectors, L_1vectors + L_2vectors, max_dim=6)
print "done."

# -- Remove operators without ML1 --

L1_eff_lag = OpSum(*[op for op in vectors_eff_lag.operators
                     if op.contains_symbol("ML1")])

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
Q1 = tensor_op("Q1")

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
     -OpSum(OphiD, Q1, OD2c)),

    # (phic Dphi) (phic Dphi) -> - (OphiD + Q1 + (phic DDphi) (phic phi))
    (Op(phic(1), D(0, phi(1)), phic(2), D(0, phi(2))),
     -OpSum(OphiD, Q1, OD2)),
    
    # Q1 -> 1/2 Ophisq - 1/2 OD2 - 1/2 OD2c
    (Q1, OpSum(half * Ophisq, -half * OD2, -half * OD2c)),

    # OD2 -> (phic DDphi) (phic phi)
    (OD2, OpSum(Op(phic(0), D(1, D(1, phi(0))), phic(2), phi(2)))),

    # OD2c -> (DDphic phi) (phic phi)
    (OD2c, OpSum(Op(D(0, D(0, phic(1))), phi(1), phic(2), phi(2)))),

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
    
    # -- Lorentz --
    
    # epsUp(-1, -2) epsUpDot(-3, -4) ->
    # -1/2 sigma4bar(0, -3, -1) sigma4bar(0, -4, -2)
    (Op(epsUp(-1, -2), epsUpDot(-3, -4)),
     OpSum(-half * Op(sigma4bar(0, -3, -1), sigma4bar(0, -4, -2)))),

    # epsDown(-1, -2) epsDownDot(-3, -4) ->
    # -1/2 sigma4(0, -1, -3) sigma4(0, -2, -4)
    (Op(epsDown(-1, -2), epsDownDot(-3, -4)),
     OpSum(-half * Op(sigma4(0, -1, -3), sigma4(0, -2, -4)))),

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
     OpSum(-number_op(1./6) * O1qu(-1, -4, -3, -2), -O8qu(-1, -4, -3, -2))),

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
                                D(4, wFS(4, 0, 2)))))
]

SM_eoms = [
    (Op(D(0, D(0, phic(-1)))),
     OpSum(
         Op(mu2phi(), phic(-1)),
         -Op(lambdaphi(), phic(0), phi(0), phic(-1)),
         -Op(ye(0, 1), lLc(2, -1, 0), eR(2, 1)),
         -Op(yd(0, 1), qLc(2, 3, -1, 0), dR(2, 3, 1)),
         -Op(V(0, 1), yuc(0, 2), uRc(3, 4, 2), qL(3, 4, 5, 1), epsSU2(5, -1)))),
     
    (Op(D(0, D(0, phi(-1)))),
     OpSum(
         Op(mu2phi(), phi(-1)),
         -Op(lambdaphi(), phic(0), phi(0), phi(-1)),
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
         number_op(0.5j) * Op(gw(), D(-1, phic(0)), sigmaSU2(-2, 0, 1), phi(1))))
]

definitions = [
    # (Dphic Dphi) (phic phi) -> Q1
    (Op(D(0, phic(1)), D(0, phi(1)), phic(2), phi(2)),
     OpSum(Q1)),
    
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

    # (Dphic phi) (qpLc gamma qL) -> i O1phiqc
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
    (Op(lLc(0, 1, -1), eR(0, -2), epsSU2(1, 2), qLc(3, 4, 2, -4), uR(3, 4, -3)),
     OpSum(Olequ(-1, -2, -3, -4))),

    # (eRc lL) epsSU2 (uRc qL) -> Olequc
    (Op(eRc(0, -2), lL(0, 1, -1), epsSU2(1, 2), uRc(3, 4, -3), qL(3, 4, 2, -4)),
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
     OpSum(Oye(-1, -2))),

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

transpose_epsSU2 = [(Op(epsSU2(-1, -2)), OpSum(Op(epsSU2(-2, -1))))]


all_rules = rules + SM_eoms + definitions + transpose_epsSU2

op_names = [
    "Okinphi", "Ophi4", "Ophi2", "Oye", "Oyec", "Oyd", "Oydc", "Oyu", "Oyuc",

    "O5", "O5c", "O5aux", "O5auxc",
    
    "Ophi", "OphiD", "Ophisq",
    
    "OphiB", "OWB", "OphiW", "OphiBTilde", "OWBTilde", "OphiWTilde",
    
    "Oephi", "Oephic", "Odphi", "Odphic", "Ouphi", "Ouphic",
    
    "O1phil", "O1philc", "O1phiq", "O1phiqc", "O3phil", "O3philc",
    "O3phiq", "O3phiqc", "O1phie", "O1phiec", "O1phid", "O1phidc",
    "O1phiu", "O1phiuc", "Ophiud", "Ophiudc",
    
    "O1ll", "Oee", "Ole", "O1qd", "O1qu", "O8qd", "O8qu",
    "Oledq", "Olequ", "O1qud", "Oledqc", "Olequc", "O1qudc",
]

print "Applying rules...",
L1_final_lag = apply_rules_until(L1_eff_lag, all_rules, op_names, 11)
print "done."

print "Collecting...",
L1_final_lag = collect_numbers_and_symbols(L1_final_lag)
L1_final_lag, L1_rest = collect_by_tensors(L1_final_lag, op_names)
print "done."

print "-- L1_final_lag --"
for op_name, coef_lst in L1_final_lag:
    print str(op_name) + ":"
    for op_coef, num in coef_lst:
        print "  " + str(num) + " " + str(op_coef)

print "-- rest --"
for op in L1_rest:
    print op
