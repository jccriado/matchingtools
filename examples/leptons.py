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
    integrate, VectorLikeFermion, MajoranaFermion)

from efttools.output import write_latex

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

lambdaNmajl = TensorBuilder("lambdaNmajl")
lambdaNmajlc = TensorBuilder("lambdaNmajlc")

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

Nmaj = FieldBuilder("Nmaj", 1.5, fermion)
Nmajc = FieldBuilder("Nmajc", 1.5, fermion)

# -- Lagrangian --

interaction_lagrangian = -OpSum(
    Op(lambdaNmajl(0, 1), Nmaj(2, 0), epsSU2(3, 4), phi(4),
       epsUp(2, 5), lL(5, 3, 1)),
    Op(lambdaNmajlc(0, 1), lLc(2, 3, 1), epsSU2(3, 4), phic(4),
       epsUpDot(2, 5), Nmajc(5, 0)))

# -- Integration --

heavy_Nmaj = MajoranaFermion("Nmaj", "Nmajc", 2)

heavy_fields = [heavy_Nmaj]

print "Integrating...",
eff_lag = integrate(heavy_fields, interaction_lagrangian, max_dim=6)
print "done."

# -- Rules --

O1phil = flavor_tensor_op("O1phil")
O1philc = flavor_tensor_op("O1philc")
O3phil = flavor_tensor_op("O3phil")
O3philc = flavor_tensor_op("O3philc")
O1phie = flavor_tensor_op("O1phie")
O1phiec = flavor_tensor_op("O1phiec")
Oephi = flavor_tensor_op("Oephi")
Oephic = flavor_tensor_op("Oephic")

O5 = flavor_tensor_op("O5")
O5c = flavor_tensor_op("O5c")

rules = [
    (Op(epsUp(-1, 0), epsDown(0, -2)), -OpSum(Op(kdelta(-1, -2)))),
    (Op(epsUp(-1, 0), epsDown(-2, 0)), OpSum(Op(kdelta(-1, -2)))),
    (Op(epsUp(0, -1), epsDown(0, -2)), OpSum(Op(kdelta(-1, -2)))),
    (Op(epsUp(0, -1), epsDown(-2, 0)), -OpSum(Op(kdelta(-1, -2)))),

    (Op(epsUpDot(-1, 0), epsDownDot(0, -2)), -OpSum(Op(kdelta(-1, -2)))),
    (Op(epsUpDot(-1, 0), epsDownDot(-2, 0)), OpSum(Op(kdelta(-1, -2)))),
    (Op(epsUpDot(0, -1), epsDownDot(0, -2)), OpSum(Op(kdelta(-1, -2)))),
    (Op(epsUpDot(0, -1), epsDownDot(-2, 0)), -OpSum(Op(kdelta(-1, -2)))),

    # epsSU2(-1, -2) epsSU2(-3, -4) ->
    # kdelta(-1, -3) kdelta(-2, -4) - kdelta(-1, -4) kdelta(-2, -3)
    # (Op(epsSU2(-1, -2), epsSU2(-3, -4)),
    #  OpSum(Op(kdelta(-1, -3), kdelta(-2, -4)),
    #        -Op(kdelta(-1, -4), kdelta(-2, -3)))),

    (Op(phic(0), D(1, phi(4)), lLc(2, 4, -1), sigma4bar(1, 2, 5), lL(5, 0, -2)),
     OpSum(number_op(-0.5j) * O1phil(-1, -2),
           number_op(-0.5j) * O3phil(-1, -2))),

    (Op(D(1, phic(0)), phi(4), lLc(2, 4, -2), sigma4bar(1, 2, 5), lL(5, 0, -1)),
     OpSum(number_op(0.5j) * O1philc(-1, -2),
           number_op(0.5j) * O3philc(-1, -2))),

    (Op(epsSU2(0, 1), epsSU2(3, 2), phic(2), 
        D(4, phi(1)), lLc(5, 3, -1), sigma4bar(4, 5, 6), lL(6, 0, -2)),
     OpSum(number_op(0.5j) * O1phil(-1, -2),
           number_op(-0.5j) * O3phil(-1, -2))),

    (Op(epsSU2(0, 1), epsSU2(3, 2), phi(2), 
        D(4, phic(1)), lLc(5, 0, -2), sigma4bar(4, 5, 6), lL(6, 3, -1)),
     OpSum(number_op(-0.5j) * O1philc(-1, -2),
           number_op(0.5j) * O3philc(-1, -2))),

    (Op(phi(0), epsSU2(0, 1), phi(1)), OpSum()),
    (Op(phic(0), epsSU2(0, 1), phic(1)), OpSum())
]

SM_eoms = [
    (Op(sigma4bar(0, -1, 1), D(0, lL(1, -2, -3))),
     OpSum(number_op(-1j) * Op(ye(-3, 0), phi(-2), eR(-1, 0)))),

    (Op(sigma4bar(0, 1, -1), D(0, lLc(1, -2, -3))),
     OpSum(number_op(1j) * Op(yec(-3, 0), phic(-2), eRc(-1, 0))))
]

definitions = [
    # -- O1phil --
    (Op(phic(0), D(1, phi(0)), lLc(2, 4, -1), sigma4bar(1, 2, 5), lL(5, 4, -2)),
     OpSum(number_op(-1j) * O1phil(-1, -2))),

    # -- O1philc --
    (Op(phi(0), D(1, phic(0)), lLc(2, 4, -2), sigma4bar(1, 2, 5), lL(5, 4, -1)),
     OpSum(number_op(1j) * O1philc(-1, -2))),

    # -- O3phil --
    (Op(phic(0), sigmaSU2(1, 0, 3), D(2, phi(3)), lLc(4, 6, -1),
        sigmaSU2(1, 6, 7), sigma4bar(2, 4, 8), lL(8, 7, -2)),
     OpSum(number_op(-1j) * O3phil(-1, -2))),

    # -- O3philc --
    (Op(D(2, phic(0)), sigmaSU2(1, 0, 3), phi(3), lLc(4, 6, -2),
        sigmaSU2(1, 6, 7), sigma4bar(2, 4, 8), lL(8, 7, -1)),
     OpSum(number_op(1j) * O3philc(-1, -2))),

    # -- O1phie --
    (Op(phic(0), D(1, phi(0)), eRc(2, -1), sigma4(1, 2, 5), eR(5, -2)),
     OpSum(number_op(-1j) * O1phie(-1, -2))),

    # -- O1phiec --
    (Op(phi(0), D(1, phic(0)), eRc(2, -2), sigma4(1, 2, 5), eR(5, -1)),
     OpSum(number_op(1j) * O1phiec(-1, -2))),

    # (phic phi) (lLc phi eR) -> Oephi
    (Op(phic(0), phi(0), lLc(1, 2, -1), phi(2), eR(1, -2)),
     OpSum(Oephi(-1, -2))),

    # (phic phi) (eRc phic lL) -> Oephic
    (Op(phic(0), phi(0), eRc(1, -2), phic(2), lL(1, 2, -1)),
     OpSum(Oephic(-1, -2))),

    # lL(0, 1, -1) epsSU2(1, 2) phi(2) epsUp(0, 3)
    # phi(4) epsSU2(5, 4) lL(3, 5, -2) -> O5
    (Op(lL(0, 1, -1), epsSU2(1, 2), phi(2), epsUp(0, 3),
        phi(4), epsSU2(5, 4), lL(3, 5, -2)),
     OpSum(O5(-1, -2))),

    # lLc(0, 1, -2) epsSU2(1, 2) phic(2) epsUp(3, 0)
    # phic(4) epsSU2(5, 4) lLc(3, 5, -1) -> O5c
    (Op(lLc(0, 1, -1), epsSU2(1, 2), phic(2), epsUpDot(0, 3),
        phic(4), epsSU2(5, 4), lLc(3, 5, -2)),
     OpSum(O5c(-1, -2))),
]

all_rules = rules + SM_eoms + definitions

op_names = [
    "O1phil", "O1philc", "O3phil", "O3philc", "O1phie", "O1phiec",
    "Oephi", "Oephic", "O5", "O5c"]

print "Appling rules...",
final_lag = apply_rules_until(eff_lag, all_rules, op_names, 11)
print "done."

final_lag_1 = final_lag

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


print "-- latex --"
structures = {
    "lambdaNmajl": "(\\lambda_N^{{l(\\text{{maj}})}})_{{{}{}}}",
    "lambdaNmajlc": "(\\lambda_N^{{l(\\text{{maj}})}})^*_{{{}{}}}",
    "MNmaj": "M_{{N^{{\\text{{maj}}}}_{{{}}}}}"
}

op_reps = {
    "O1phil": "\\frac{\\left(\\alpha^{\\left(1\\right)}_{\\phi l}\\right)_{ij}}{\Lambda^2}",
    "O1philc": "\\frac{\\left(\\alpha^{\\left(1\\right)}_{\\phi l}\\right)^*_{ij}}{\Lambda^2}", 
    "O3phil": "\\frac{\\left(\\alpha^{\\left(3\\right)}_{\\phi l}\\right)_{ij}}{\Lambda^2}",
    "O3philc": "\\frac{\\left(\\alpha^{\\left(3\\right)}_{\\phi l}\\right)^*_{ij}}{\Lambda^2}",
    "O1phie": "\\frac{\\left(\\alpha^{\\left(1\\right)}_{\\phi e}\\right)_{ij}}{\Lambda^2}",
    "O1phiec": "\\frac{\\left(\\alpha^{\\left(1\\right)}_{\\phi e}\\right)^*_{ij}}{\Lambda^2}",
    "Oephi": "\\frac{\\left(\\alpha_{e\\phi}\\right)_{ij}}{\Lambda^2}",
    "Oephic": "\\frac{\\left(\\alpha_{e\\phi}\\right)^*_{ij}}{\Lambda^2}",
    "O5": "\\frac{\\left(\\alpha_5\\right)_{ij}}{\Lambda^2}", 
    "O5c": "\\frac{\\left(\\alpha_5\\right)^*_{ij}}{\Lambda^2}"}
print type(final_lag_1.operators)
write_latex("leptons.tex", final_lag_1, structures, op_reps,
            ["i", "j", "k", "l", "m", "n"])
