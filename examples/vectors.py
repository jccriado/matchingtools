import context

from efttools.algebra import (
    Tensor, Op, OpSum,
    TensorBuilder, FieldBuilder, D,
    number_op, symbol_op, tensor_op,
    boson, fermion, kdelta,
    sigma4, sigma4bar, epsUp, epsUpDot, epsDown, epsDownDot,
    collect_numbers_and_symbols, collect_by_tensors,
    apply_rules_until)

from efttools.integration import (
    integrate, RealVector, ComplexVector)

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

glB = TensorBuilder("glB")
gqB = TensorBuilder("gqB")
geB = TensorBuilder("geB")
gdB = TensorBuilder("gdB")
guB = TensorBuilder("guB")
gphiB = TensorBuilder("gphiB")
gphiBc = TensorBuilder("gphiBc")

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

B = FieldBuilder("B", 1, boson)

# -- Lagrangian --

interaction_lagrangian = -OpSum(
    Op(glB(0, 1, 2), B(3, 0), lLc(4, 5, 1), sigma4bar(3, 4, 6), lL(6, 5, 2)),
    Op(gqB(0, 1, 2), B(3, 0), qLc(4, 5, 6, 1), sigma4bar(3, 4, 7), qL(7, 5, 6, 2)),
    Op(geB(0, 1, 2), B(3, 0), eRc(4, 1), sigma4(3, 4, 5), eR(5, 2)),
    Op(gdB(0, 1, 2), B(3, 0), dRc(4, 5, 1), sigma4(3, 4, 6), dR(6, 5, 2)),
    Op(guB(0, 1, 2), B(3, 0), uRc(4, 5, 1), sigma4(3, 4, 6), uR(6, 5, 2)),
    number_op(1j) * Op(gphiB(0), B(1, 0), phic(2), D(1, phi(2))),
    number_op(-1j) * Op(gphiBc(0), B(1, 0), phi(2), D(1, phic(2)))
)

# -- Heavy fields --

heavy_B = RealVector("B", 2)
heavy_fields = [heavy_B]

print heavy_B.equations_of_motion(interaction_lagrangian)

# -- Integration --

print "Integrating...",
eff_lag = integrate(heavy_fields, interaction_lagrangian, max_dim=6)
print "done."

# for op in eff_lag.operators:
#     print op

# -- Transformations --

O1phi = tensor_op("O1phi")
O3phi = tensor_op("O3phi")
Ophi4 = tensor_op("Ophi4")
Ophi6 = tensor_op("Ophi6")

def flavor_tensor_op(name):
    def f(*indices):
        return Op(Tensor(name, list(indices)))
    return f

O1ll = flavor_tensor_op("O1ll")
O11qq = flavor_tensor_op("O11qq")
O1lq = flavor_tensor_op("O1lq")
Ole = flavor_tensor_op("Ole")
Old = flavor_tensor_op("Old")
Olu = flavor_tensor_op("Olu")
Oqe = flavor_tensor_op("Oqe")
O1qd = flavor_tensor_op("O1qd")
O1qu = flavor_tensor_op("O1qu")
O8qd = flavor_tensor_op("O8qd")
O8qu = flavor_tensor_op("O8qu")
Oee = flavor_tensor_op("Oee")
O1dd = flavor_tensor_op("O1dd")
O1uu = flavor_tensor_op("O1uu")
Oeu = flavor_tensor_op("Oeu")
Oed = flavor_tensor_op("Oed")
O1ud = flavor_tensor_op("O1ud")
O1phil = flavor_tensor_op("O1phil")
O1phiq = flavor_tensor_op("O1phiq")
O1phie = flavor_tensor_op("O1phie")
O1phid = flavor_tensor_op("O1phid")
O1phiu = flavor_tensor_op("O1phiu")
O1philc = flavor_tensor_op("O1philc")
O1phiqc = flavor_tensor_op("O1phiqc")
O1phiec = flavor_tensor_op("O1phiec")
O1phidc = flavor_tensor_op("O1phidc")
O1phiuc = flavor_tensor_op("O1phiuc")

rules = [
    # -- Higgs and derivatives --
    (Op(phic(0), D(1, phi(0)), phic(2), D(1, phi(2))),
     -OpSum(O1phi, O3phi, Op(phic(0), phi(0), phic(1), D(2, D(2, phi(1)))))),
    (Op(D(1, phic(0)), phi(0), D(1, phic(2)), phi(2)),
     -OpSum(O1phi, O3phi, Op(phic(0), phi(0), D(2, D(2, phic(1))), phi(1)))),

    # -- Lorentz --
    (Op(sigma4(0, -1, -2), sigma4bar(0, -3, -4)),
     OpSum(number_op(2) * Op(kdelta(-1, -4), kdelta(-3, -2)))),

    # -- Four-quark Fierz reorderings --
    (Op(qLc(0, 1, 2, -1), dR(0, 3, -2), dRc(4, 3, -3), qL(4, 1, 2, -4)),
     OpSum(number_op(1./3) * O1qd(-1, -2, -3, -4),
           number_op(1./2) * O8qd(-1, -2, -3, -4))),
    (Op(qLc(0, 1, 2, -1), uR(0, 3, -2), uRc(4, 3, -3), qL(4, 1, 2, -4)),
     OpSum(number_op(1./3) * O1qu(-1, -2, -3, -4),
           number_op(1./2) * O8qu(-1, -2, -3, -4))),
]

definitions = [
    # -- O1phi --
    (Op(phic(0), phi(0), D(1, phic(2)), D(1, phi(2))), OpSum(O1phi)),

    # -- O3phi --
    (Op(phic(0), D(1, phi(0)), D(1, phic(2)), phi(2)), OpSum(O3phi)),

    # -- Ophi6 --
    (Op(phic(0), phi(0), phic(1), phi(1), phic(2), phi(2)),
        OpSum(number_op(3) * Ophi6)),

    # -- Ophi4 --
    (Op(phic(0), phi(0), phic(1), phi(1)), OpSum(Ophi4)),
    
    # -- O1ll --
    (Op(lLc(0, 1, -1), sigma4bar(2, 0, 3), lL(3, 1, -2),
        lLc(4, 5, -3), sigma4bar(2, 4, 6), lL(6, 5, -4)),
     OpSum(number_op(2) * O1ll(-1, -2, -3, -4))),

    # -- O11qq --
    (Op(qLc(0, 1, 2, -1), sigma4bar(3, 0, 4), qL(4, 1, 2, -2),
        qLc(5, 6, 7, -3), sigma4bar(3, 5, 8), qL(8, 6, 7, -4)),
     OpSum(number_op(2) * O11qq(-1, -2, -3, -4))),

    # -- O1lq --
    (Op(lLc(0, 1, -1), sigma4bar(2, 0, 3), lL(3, 1, -2),
        qLc(4, 5, 6, -3), sigma4bar(2, 4, 7), qL(7, 5, 6, -4)),
     OpSum(O1lq(-1, -2, -3, -4))),

    # -- Ole --
    (Op(lLc(0, 1, -1), eR(0, -2), eRc(2, -3), lL(2, 1, -4)),
     OpSum(Ole(-1, -2, -3, -4))),

    # -- Old --
    (Op(lLc(0, 1, -1), dR(0, 2, -2), dRc(3, 2, -3), lL(3, 1, -4)),
     OpSum(Old(-1, -2, -3, -4))),

    # -- Olu --
    (Op(lLc(0, 1, -1), uR(0, 2, -2), uRc(3, 2, -3), lL(3, 1, -4)),
     OpSum(Olu(-1, -2, -3, -4))),

    # -- Oqe --
    (Op(qLc(0, 1, 2, -1), eR(0, -2), eRc(3, -3), qL(3, 1, 2, -4)),
     OpSum(Ole(-1, -2, -3, -4))),

    # -- O1qd --
    (Op(qLc(0, 1, 2, -1), dR(0, 1, -2), dRc(3, 4, -3), qL(3, 4, 2, -4)),
     OpSum(O1qd(-1, -2, -3, -4))),

    # -- O1qu --
    (Op(qLc(0, 1, 2, -1), uR(0, 1, -2), uRc(3, 4, -3), qL(3, 4, 2, -4)),
     OpSum(O1qu(-1, -2, -3, -4))),

    # -- O1qd --
    (Op(qLc(0, 1, 2, -1), lambdaColor(3, 1, 4), dR(0, 4, -2),
        dRc(5, 6, -3), lambdaColor(3, 6, 7), qL(5, 7, 2, -4)),
     OpSum(O8qd(-1, -2, -3, -4))),

    # -- O1qu --
    (Op(qLc(0, 1, 2, -1), lambdaColor(3, 1, 4), uR(0, 4, -2),
        uRc(5, 6, -3), lambdaColor(3, 6, 7), qL(5, 7, 2, -4)),
     OpSum(O8qu(-1, -2, -3, -4))),

    # -- Oee --
    (Op(eRc(0, -1), sigma4(1, 0, 2), eR(2, -2),
        eRc(3, -3), sigma4(1, 3, 4), eR(4, -4)),
     OpSum(number_op(2) * Oee(-1, -2, -3, -4))),

    # -- O1dd --
    (Op(dRc(0, 1, -1), sigma4(2, 0, 3), dR(3, 1, -2),
        dRc(4, 5, -3), sigma4(2, 4, 6), dR(6, 5, -4)),
     OpSum(number_op(2) * O1dd(-1, -2, -3, -4))),

    # -- O1uu --
    (Op(uRc(0, 1, -1), sigma4(2, 0, 3), uR(3, 1, -2),
        uRc(4, 5, -3), sigma4(2, 4, 6), uR(6, 5, -4)),
     OpSum(number_op(2) * O1uu(-1, -2, -3, -4))),

    # -- Oed --
    (Op(eRc(0, -1), sigma4(1, 0, 2), eR(2, -2),
        dRc(3, 4, -3), sigma4(1, 3, 5), dR(5, 4, -4)),
     OpSum(Oed(-1, -2, -3, -4))),
    
    # -- Oeu --
    (Op(eRc(0, -1), sigma4(1, 0, 2), eR(2, -2),
        uRc(3, 4, -3), sigma4(1, 3, 5), uR(5, 4, -4)),
     OpSum(Oeu(-1, -2, -3, -4))),

    # -- O1ud --
    (Op(uRc(0, 1, -1), sigma4(2, 0, 3), uR(3, 1, -2),
        dRc(4, 5, -3), sigma4(2, 4, 6), dR(6, 5, -4)),
     OpSum(O1ud(-1, -2, -3, -4))),

    # -- O1phil --
    (Op(phic(0), D(1, phi(0)), lLc(2, 3, -1), sigma4bar(1, 2, 4), lL(4, 3, -2)),
     OpSum(number_op(-1j) * O1phil(-1, -2))),

    # -- O1phiq --
    (Op(phic(0), D(1, phi(0)), qLc(2, 3, 4, -1), sigma4bar(1, 2, 5), qL(5, 3, 4, -2)),
     OpSum(number_op(-1j) * O1phiq(-1, -2))),

    # -- O1phie --
    (Op(phic(0), D(1, phi(0)), eRc(2, -1), sigma4(1, 2, 3), eR(3, -2)),
     OpSum(number_op(-1j) * O1phie(-1, -2))),

    # -- O1phid --
    (Op(phic(0), D(1, phi(0)), dRc(2, 3, -1), sigma4(1, 2, 4), dR(4, 3, -2)),
     OpSum(number_op(-1j) * O1phid(-1, -2))),

    # -- O1phiu --
    (Op(phic(0), D(1, phi(0)), uRc(2, 3, -1), sigma4(1, 2, 4), uR(4, 3, -2)),
     OpSum(number_op(-1j) * O1phiu(-1, -2))),

    # -- O1philc --
    (Op(phi(0), D(1, phic(0)), lLc(2, 3, -1), sigma4bar(1, 2, 4), lL(4, 3, -2)),
     OpSum(number_op(1j) * O1philc(-1, -2))),

    # -- O1phiqc --
    (Op(phi(0), D(1, phic(0)), qLc(2, 3, 4, -1), sigma4bar(1, 2, 5), qL(5, 3, 4, -2)),
     OpSum(number_op(1j) * O1phiqc(-1, -2))),

    # -- O1phiec --
    (Op(phi(0), D(1, phic(0)), eRc(2, -1), sigma4(1, 2, 3), eR(3, -2)),
     OpSum(number_op(1j) * O1phiec(-1, -2))),

    # -- O1phidc --
    (Op(phi(0), D(1, phic(0)), dRc(2, 3, -1), sigma4(1, 2, 4), dR(4, 3, -2)),
     OpSum(number_op(1j) * O1phidc(-1, -2))),

    # -- O1phiuc --
    (Op(phi(0), D(1, phic(0)), uRc(2, 3, -1), sigma4(1, 2, 4), uR(4, 3, -2)),
     OpSum(number_op(1j) * O1phiuc(-1, -2))),
]

all_rules = rules + definitions

op_names = ["O1phi", "O3phi", "Ophi4", "Ophi6",
            "O1ll", "O11qq", "O1lq", "Ole", "Old", "Olu",
            "Oqe", "O1qd", "O1qu", "O8qd", "O8qu",
            "Oee", "O1dd", "O1uu", "Oeu", "Oed", "O1ud",
            "O1phil", "O1phiq", "O1phie", "O1phid", "O1phiu",
            "O1philc", "O1phiqc", "O1phiec", "O1phidc", "O1phiuc"
]

print "Appling rules...",
final_lag = apply_rules_until(eff_lag, all_rules, op_names, 11)
print "done."

print "Collecting...",
final_lag = collect_numbers_and_symbols(final_lag)
final_lag, rest = collect_by_tensors(final_lag, op_names)
print "done."

# -- Printing --

print "-- final_lag --"
for op_name, coef_lst in final_lag:
    print str(op_name) + ":"
    for op_coef, num in coef_lst:
        print "  " + str(num) + " " + str(op_coef)

print "-- rest --"
for op in rest:
    print op
