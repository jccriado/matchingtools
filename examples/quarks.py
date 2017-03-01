import context

from efttools.algebra import (
    Tensor, Op, OpSum,
    TensorBuilder, FieldBuilder, D,
    number_op, symbol_op, tensor_op, flavor_tensor_op,
    boson, fermion, kdelta,
    sigma4, sigma4bar, epsUp, epsUpDot, epsDown, epsDownDot,
    collect_numbers_and_symbols, collect_by_tensors,
    apply_rules_until)

from efttools.integration import (
    integrate, VectorLikeFermion)

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

lambdaP2 = TensorBuilder("lambdaP2")
lambdaP2c = TensorBuilder("lambdaP2c")

# -- Group tensors --

epsSU2 = TensorBuilder("epsSU2")
sigmaSU2 = TensorBuilder("sigmaSU2")

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

DL = FieldBuilder("DL", 1, boson)
DR = FieldBuilder("DR", 1, boson)
DLc = FieldBuilder("DLc", 1, boson)
DRc = FieldBuilder("DRc", 1, boson)

# -- Lagrangian --

interaction_lagrangian = -OpSum(
    Op(lambdaP2(0, 1), DRc(2, 3, 0), phic(4), qL(2, 3, 4, 1)),
    Op(lambdaP2c(0, 1), qLc(2, 3, 4, 1), phi(4), DR(2, 3, 0))
)

heavy_D = VectorLikeFermion("D", "DL", "DR", "DLc", "DRc", 3)
heavy_fields = [heavy_D]

# -- Integration --

print "Integrating...",
eff_lag = integrate(heavy_fields, interaction_lagrangian, max_dim=6)
print "done."

for op in eff_lag.operators:
    print op

# -- Rules --

O1phiq = flavor_tensor_op("O1phiq")
O1phiqc = flavor_tensor_op("O1phiqc")
O3phiq = flavor_tensor_op("O3phiq")
O3phiqc = flavor_tensor_op("O3phiqc")
Oephi = flavor_tensor_op("Oephi")
Odphi = flavor_tensor_op("Odphi")
Ouphi = flavor_tensor_op("Ouphi")
Oephic = flavor_tensor_op("Oephic")
Odphic = flavor_tensor_op("Odphic")
Ouphic = flavor_tensor_op("Ouphic")

rules = [
    (Op(phic(0), D(1, phi(4)), qLc(2, 3, 4, -1), sigma4bar(1, 2, 5), qL(5, 3, 0, -2)),
     OpSum(number_op(-1j/2) * O1phiq(-1, -2),
           number_op(-1j/2) * O3phiq(-1, -2))),

    (Op(D(1, phic(0)), phi(4), qLc(2, 3, 4, -2), sigma4bar(1, 2, 5), qL(5, 3, 0, -1)),
     OpSum(number_op(1j/2) * O1phiqc(-1, -2),
           number_op(1j/2) * O3phiqc(-1, -2))),

    (Op(phi(0), epsSU2(0, 1), phi(1)), OpSum()),
    (Op(phic(0), epsSU2(0, 1), phic(1)), OpSum())]

SM_eoms = [
    (Op(sigma4bar(0, -1, 1), D(0, qL(1, -2, -3, -4))),
     OpSum(number_op(-1j) * Op(yd(-4, 0), phi(-3), dR(-1, -2, 0)),
           number_op(-1j) * Op(Vc(0, -4), yu(0, 1), epsSU2(-3, 2),
                               phic(2), uR(-1, -2, 1)))),

    (Op(sigma4bar(0, 1, -1), D(0, qLc(1, -2, -3, -4))),
     OpSum(number_op(1j) * Op(ydc(-4, 0), phic(-3), dRc(-1, -2, 0)),
           number_op(1j) * Op(V(0, -4), yuc(0, 1), epsSU2(-3, 2),
                               phi(2), uRc(-1, -2, 1))))]
    
definitions = [
    # -- O1phiq --
    (Op(phic(0), D(1, phi(0)), qLc(2, 3, 4, -1), sigma4bar(1, 2, 5), qL(5, 3, 4, -2)),
     OpSum(number_op(-1j) * O1phiq(-1, -2))),

    # -- O1phiqc --
    (Op(phi(0), D(1, phic(0)), qLc(2, 3, 4, -2), sigma4bar(1, 2, 5), qL(5, 3, 4, -1)),
     OpSum(number_op(1j) * O1phiqc(-1, -2))),

    # -- O3phiq --
    (Op(phic(0), sigmaSU2(1, 0, 3), D(2, phi(3)), qLc(4, 5, 6, -1),
        sigmaSU2(1, 6, 7), sigma4bar(2, 4, 8), qL(8, 5, 7, -2)),
     OpSum(number_op(-1j) * O3phiq(-1, -2))),

    # -- O3phiqc --
    (Op(D(2, phic(0)), sigmaSU2(1, 0, 3), phi(3), qLc(4, 5, 6, -2),
        sigmaSU2(1, 6, 7), sigma4bar(2, 4, 8), qL(8, 5, 7, -1)),
     OpSum(number_op(1j) * O3phiqc(-1, -2))),

    # (phic phi) (lLc phi eR) -> Oephi
    (Op(phic(0), phi(0), lLc(1, 2, -1), phi(2), eR(1, -2)),
     OpSum(Oephi(-1, -2))),

    # (phic phi) (eRc phic lL) -> Oephic
    (Op(phic(0), phi(0), eRc(1, -2), phic(2), lL(1, 2, -1)),     OpSum(Oephic(-1, -2))),

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
]

all_rules = rules + SM_eoms + definitions

op_names = [
    "O1phiq", "O1phiqc", "O3phiq", "O3phiqc",
    "Oephi", "Oephic", "Odphi", "Odphic", "Ouphi", "Ouphic",]

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

