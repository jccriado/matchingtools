from operators import (
    Tensor, Op, OpSum,
    TensorBuilder, FieldBuilder, D,
    number_op, symbol_op, tensor_op,
    boson, fermion, kdelta,
    sigma4, sigma4bar, epsUp, epsUpDot, epsDown, epsDownDot)

from transformations import (
    collect_numbers_and_symbols, collect_by_tensors,
    apply_rules_until)

from integration import (
    integrate, RealScalar, ComplexScalar)

# -- Flavor tensors --

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

# -- Group tensors --

epsSU2 = TensorBuilder("epsSU2")
sigmaSU2 = TensorBuilder("sigmaSU2")
fSU2 = TensorBuilder("fSU2")

lambdaColor = TensorBuilder("lambdaColor")

# -- Fields --

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

S = FieldBuilder("S", 1, boson)
S1 = FieldBuilder("S1", 1, boson)
S1c = FieldBuilder("S1c", 1, boson)
S2 = FieldBuilder("S2", 1, boson)
S2c = FieldBuilder("S2c", 1, boson)
varphi = FieldBuilder("varphi", 1, boson)
varphic = FieldBuilder("varphic", 1, boson)
Xi0 = FieldBuilder("Xi0", 1, boson)
Xi1 = FieldBuilder("Xi1", 1, boson)
Xi1c = FieldBuilder("Xi1c", 1, boson)

# -- Lagrangian --

interaction_lagrangian = -OpSum(
    # -- S --
    Op(kappaS(0), phic(1), phi(1), S(0)),
    Op(lambdaS(0, 1), S(0), S(1), phic(2), phi(2)),
    Op(kappaS3(0, 1, 2), S(0), S(1), S(2)),
    
    # -- S1 --
    Op(ylS1(0, 1, 2), S1c(0), lLc(3, 4, 1), epsUpDot(3, 5),
              epsSU2(4, 6), lLc(5, 6, 2)),
    Op(ylS1c(0, 1, 2), S1(0), lL(3, 4, 1), epsUp(3, 5),
              epsSU2(4, 6), lL(5, 6, 2)),

    # -- S2 --
    Op(yeS2(0, 1, 2), S2c(0), epsDown(3, 4), eRc(3, 1), eRc(4, 2)),
    Op(yeS2c(0, 1, 2), S2(0), epsDownDot(3, 4), eR(3, 1), eR(4, 2)),

    # -- varphi --
    Op(yevarphi(0, 1, 2), varphic(3, 0), eRc(4, 1), lL(4, 3, 2)),
    Op(yevarphic(0, 1, 2), varphi(3, 0), lLc(4, 3, 2), eR(4, 1)),
    Op(ydvarphi(0, 1, 2), varphic(3, 0), dRc(4, 5, 1), qL(4, 5, 3, 2)),
    Op(ydvarphic(0, 1, 2), varphi(3, 0), qLc(4, 5, 3, 2), dR(4, 5, 1)),
    Op(yuvarphi(0, 1, 2), varphic(3, 0), epsSU2(3, 5), qLc(4, 6, 5, 1), uR(4, 6, 2)),
    Op(yuvarphic(0, 1, 2), varphi(3, 0), epsSU2(3, 5), uRc(4, 6, 2), qL(4, 6, 5, 1)),
    Op(lambdavarphi(0), varphic(1, 0), phi(1), phic(2), phi(2)),
    Op(lambdavarphic(0), phic(1), varphi(1, 0), phic(2), phi(2)),

    # -- Xi0 --
    Op(kappaXi0(0), Xi0(1, 0), phic(2), sigmaSU2(1, 2, 3), phi(3)),
    Op(lambdaXi0(0, 1), Xi0(2, 0), Xi0(2, 1), phic(3), phi(3)),

    # -- Xi1 --
    Op(ylXi1(0, 1, 2), Xi1c(3, 0), lLc(4, 5, 1), lLc(6, 7, 2),
              epsUpDot(4, 6), sigmaSU2(3, 5, 8), epsSU2(8, 7)),
    Op(ylXi1c(0, 1, 2), Xi1(3, 0), lL(4, 5, 1), lL(6, 7, 2),
              epsUp(4, 6), sigmaSU2(3, 8, 5), epsSU2(8, 7)),
    Op(kappaXi1(0), Xi1(1, 0), phic(2), sigmaSU2(1, 2, 3),
              epsSU2(3, 4), phic(4)),
    Op(kappaXi1c(0), Xi1c(1, 0), phi(2), epsSU2(3, 2),
              sigmaSU2(1, 3, 4), phi(4)),
    Op(lambdaXi1(0, 1), Xi1c(2, 0), Xi1(2, 1), phic(3), phi(3)),
    Op(lambdaTildeXi1(0, 1), fSU2(2, 3, 4), Xi1c(2, 0), Xi1(3, 1),
              phic(5), sigmaSU2(4, 5, 6), phi(6))
)
                               

# -- Heavy fields --

heavy_S = RealScalar("S", 1)
heavy_S1 = ComplexScalar("S1", "S1c", 1)
heavy_S2 = ComplexScalar("S2", "S2c", 1)
heavy_varphi = ComplexScalar("varphi", "varphic", 2)
heavy_Xi0 = RealScalar("Xi0", 2)
heavy_Xi1 = ComplexScalar("Xi1", "Xi1c", 2)
heavy_fields = [
    heavy_S, heavy_S1, heavy_S2, heavy_varphi, heavy_Xi0, heavy_Xi1]

# -- Integration --

print "Integrating...",
eff_lag = integrate(heavy_fields, interaction_lagrangian, max_dim=6)
print "done."

# -- Transformations --

Ophi = tensor_op("Ophi")
Ophi4 = tensor_op("Ophi4")
OphiD = tensor_op("OphiD")
Ophisq = tensor_op("Ophisq")
OD2 = tensor_op("OD2")
OD2c = tensor_op("OD2c")
Q1 = tensor_op("Q1")
half = number_op(0.5)

def flavor_tensor_op(name):
    def f(*indices):
        return Op(Tensor(name, list(indices)))
    return f

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
Oephi = flavor_tensor_op("Oephi")
Odphi = flavor_tensor_op("Odphi")
Ouphi = flavor_tensor_op("Ouphi")
Oephic = flavor_tensor_op("Oephic")
Odphic = flavor_tensor_op("Odphic")
Ouphic = flavor_tensor_op("Ouphic")
O5 = flavor_tensor_op("O5")
O5c = flavor_tensor_op("O5c")
O5aux = flavor_tensor_op("O5aux")
O5auxc = flavor_tensor_op("O5auxc")

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

    # fSU2(0, 1, 2) (phic sigmaSU2(0) phi) (phi eps sigmaSU2(1) phi)
    # (phic sigmaSU2(2) eps phic) -> 3 * sqrt(2) * Ophi
    (Op(fSU2(0, 1, 2), phic(3), sigmaSU2(0, 3, 4), phi(4),
               phi(5), epsSU2(6, 5), sigmaSU2(1, 6, 7), phi(7),
               phic(8), sigmaSU2(2, 8, 9), epsSU2(9, 10), phic(10)),
     OpSum(number_op(3) * symbol_op("sqrt(2)", 1) * Ophi)),
    (Op(fSU2(0, 1, 2), phic(3), sigmaSU2(0, 3, 4), phi(4),
               phi(5), epsSU2(6, 5), sigmaSU2(2, 6, 7), phi(7),
               phic(8), sigmaSU2(1, 8, 9), epsSU2(9, 10), phic(10)),
     OpSum(number_op(-3) * symbol_op("sqrt(2)", 1) * Ophi)),
    (Op(fSU2(0, 1, 2), phic(3), sigmaSU2(1, 3, 4), phi(4),
               phi(5), epsSU2(6, 5), sigmaSU2(0, 6, 7), phi(7),
               phic(8), sigmaSU2(2, 8, 9), epsSU2(9, 10), phic(10)),
     OpSum(number_op(-3) * symbol_op("sqrt(2)", 1) * Ophi)),
    (Op(fSU2(0, 1, 2), phic(3), sigmaSU2(1, 3, 4), phi(4),
               phi(5), epsSU2(6, 5), sigmaSU2(2, 6, 7), phi(7),
               phic(8), sigmaSU2(0, 8, 9), epsSU2(9, 10), phic(10)),
     OpSum(number_op(3) * symbol_op("sqrt(2)", 1) * Ophi)),
    (Op(fSU2(0, 1, 2), phic(3), sigmaSU2(2, 3, 4), phi(4),
               phi(5), epsSU2(6, 5), sigmaSU2(0, 6, 7), phi(7),
               phic(8), sigmaSU2(1, 8, 9), epsSU2(9, 10), phic(10)),
     OpSum(number_op(3) * symbol_op("sqrt(2)", 1) * Ophi)),
    (Op(fSU2(0, 1, 2), phic(3), sigmaSU2(2, 3, 4), phi(4),
               phi(5), epsSU2(6, 5), sigmaSU2(1, 6, 7), phi(7),
               phic(8), sigmaSU2(0, 8, 9), epsSU2(9, 10), phic(10)),
     OpSum(number_op(-3) * symbol_op("sqrt(2)", 1) * Ophi)),

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

    # -- Four-fermion Fierz reorderings --

    (Op(lLc(0, 1, -1), sigma4bar(2, 0, 3), lL(3, 4, -2),
        lLc(5, 4, -3), sigma4bar(2, 5, 6), lL(6, 1, -4)),
     OpSum(Op(lLc(0, 1, -1), sigma4bar(2, 0, 3), lL(3, 1, -4),
              lLc(5, 4, -3), sigma4bar(2, 5, 6), lL(6, 4, -2)))),
    
    (Op(lLc(0, 1, -1), eR(0, -2), eRc(2, -3), lL(2, 1, -4)),
     OpSum(-half * Ole(-1, -4, -3, -2))),

    (Op(qLc(0, 1, 2, -1), dR(0, 1, -2), dRc(3, 4, -3), qL(3, 4, 2, -4)),
     OpSum(-number_op(1./6) * O1qd(-1, -4, -3, -2),
            -O8qd(-1, -4, -3, -2))),

    (Op(qLc(0, 1, 2, -1), uR(0, 1, -2), uRc(3, 4, -3), qL(3, 4, 2, -4)),
     OpSum(-number_op(1./6) * O1qu(-1, -4, -3, -2), -O8qu(-1, -4, -3, -2))),

    # -- ?? --
    (Op(lL(0, 1, -1), phi(1), epsUp(0, 2), phi(3), lL(2, 3, -2)),
     OpSum(-O5(-1, -2), O5aux(-1, -2))),
    (Op(lLc(0, 1, -1), phic(1), epsUpDot(0, 2), phic(3), lLc(2, 3, -2)),
     OpSum(-O5c(-1, -2), O5auxc(-1, -2))),

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
         -Op(Vc(0, 1), yu(0, 2), qLc(3, 4, 5, 1), uR(3, 4, 2), epsSU2(5, -1))))
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
    
    # (phic phi)^2 -> Ophi4
    (Op(phic(0), phi(0), phic(1), phi(1)),
     OpSum(Ophi4)),
    
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
      OpSum(O5auxc(-1, -2)))
]

change_epsSU2 = [
    (Op(epsSU2(-1, -2)), OpSum(Op(epsSU2(-2, -1))))]

all_rules = rules + SM_eoms + definitions + change_epsSU2

op_names = [
    "Ophi4", "Ophi", "OphiD", "Ophisq", "O1ll", "Oee", "Ole",
    "O1qd", "O1qu", "O8qd", "O8qu",
    "Oledq", "Olequ", "O1qud", "Oledqc", "Olequc", "O1qudc",
    "Oephi", "Oephic", "Odphi", "Odphic", "Ouphi", "Ouphic",
    "O5", "O5c", "O5aux", "O5auxc"]

print "Appling rules...",
final_lag = apply_rules_until(eff_lag, all_rules, op_names, 10)
print "done."

print "Collecting...",
final_lag = collect_numbers_and_symbols(final_lag)
final_lag, rest = collect_by_tensors(final_lag, op_names)
print "done."

# -- Printing --

# print "-- eff_lag --"
# for op in eff_lag.operators:
#     print op

print "-- final_lag --"
for op_name, coef_lst in final_lag:
    print str(op_name) + ":"
    for op_coef, num in coef_lst:
        print "  " + str(num) + " " + str(op_coef)

print "-- rest --"
for op in rest:
    print op
