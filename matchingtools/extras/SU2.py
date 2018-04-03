"""
This module defines tensors and rules related to the
group :math:`SU(2)`.
"""

from matchingtools.core import (
    Op, OpSum, TensorBuilder, number_op, power_op, kdelta)
from math import sqrt
from fractions import Fraction

epsSU2 = TensorBuilder("epsSU2")
r"""
Totally antisymmetric tensor :math:`\epsilon=i\sigma^2` with two
:math:`SU(2)` doublet indices such that :math:`\epsilon_{12}=1`.
"""

sigmaSU2 = TensorBuilder("sigmaSU2")
r"""
Pauli matrices :math:`(\sigma^a)_{ij}`.
"""

CSU2 = TensorBuilder("CSU2")
r"""
Clebsh-Gordan coefficients :math:`C^I_{a\beta}` with
the first index :math:`I` being a quadruplet index,
the second :math:`a` a triplet index, and the third 
:math:`\beta` a doublet index.
"""

CSU2c = TensorBuilder("CSU2c")
r"""
Conjugate of the Clebsh-Gordan coefficients :math:`C^I_{a\beta}`.
"""

epsSU2triplets = TensorBuilder("epsSU2triplets")
r"""
Totally antisymmetric tensor :math:`\epsilon_{abc}` with three
:math:`SU(2)` triplet indices such that :math:`\epsilon_{123}=1`.
"""

epsSU2quadruplets = TensorBuilder("epsSU2quadruplets")
r"""
Two-index that gives a singlet when contracted with two
:math:`SU(2)` quadruplets.
"""

fSU2 = TensorBuilder("fSU2")
r"""
Totally antisymmetric tensor with three :math:`SU(2)` triplet indices
given by :math:`f_{abc}=\frac{i}{\sqrt{2}}\epsilon_{abc}` with
:math:`\epsilon_{123}=1`.
"""

rule_SU2_fierz = (
    (Op(sigmaSU2(0, -1, -2), sigmaSU2(0, -3, -4)),
     OpSum(number_op(2) * Op(kdelta(-1, -4), kdelta(-3, -2)),
           -Op(kdelta(-1, -2), kdelta(-3, -4)))))
r"""
Subtitute :math:`\sigma^a_{ij} \sigma^a_{kl}` by
:math:`2\delta_{il}\delta_{kj} - \delta_{ij}\delta{kl}`.
"""

rule_SU2_product_sigmas = (
    (Op(sigmaSU2(0, -1, 1), sigmaSU2(0, 1, -2)),
     OpSum(number_op(3) * Op(kdelta(-1, -2)))))
r"""
Subtitute :math:`\sigma^a_{ij}\sigma^a_{jk}` by
:math:`3\delta_{ik}`.
"""

rule_SU2_free_eps = (
    (Op(epsSU2(-1, -2), epsSU2(-3, -4)),
     OpSum(Op(kdelta(-1, -3), kdelta(-2, -4)),
           -Op(kdelta(-1, -4), kdelta(-2, -3)))))
r"""
Subtitute :math:`\epsilon_{ij}\epsilon_{kl}` by
:math:`\delta_{ik}\delta_{jl}-\delta_{il}\delta_{jk}`.
"""

rules_SU2_eps_cancel = [
    (Op(epsSU2(-1, 0), epsSU2(0, -2)), -OpSum(Op(kdelta(-1, -2)))),

    (Op(epsSU2(0, -1), epsSU2(0, -2)), OpSum(Op(kdelta(-1, -2)))),

    (Op(epsSU2(-1, 0), epsSU2(-2, 0)), OpSum(Op(kdelta(-1, -2))))]
r"""
Substitute contracted :math:`\epsilon` tensors with the corresponding
Kronecker delta.
"""

rule_SU2_eps_zero = (Op(epsSU2(0, 0)), OpSum())
r"""
Substitute :math:`\epsilon_{ii}` by zero because :math:`\epsilon` 
is antisymmetric.
"""

rules_SU2_epsquadruplets_cancel = [
    (Op(epsSU2quadruplets(-1, 0), epsSU2quadruplets(0, -2)),
     -OpSum(number_op(Fraction(1, 4)) * Op(kdelta(-1, -2)))),

    (Op(epsSU2quadruplets(0, -1), epsSU2quadruplets(0, -2)),
     OpSum(number_op(Fraction(1, 4)) * Op(kdelta(-1, -2)))),

    (Op(epsSU2quadruplets(-1, 0), epsSU2quadruplets(-2, 0)),
     OpSum(number_op(Fraction(1, 4)) * Op(kdelta(-1, -2))))]
r"""
Substitute contracted :math:`\epsilon` tensors with the corresponding
Kronecker delta.
"""

rules_SU2_C_sigma = [
    (Op(CSU2c(0, 1, -1), sigmaSU2(1, -3, -4),
        CSU2(0, 2, -2), sigmaSU2(2, -5, -6)),
     OpSum(
         number_op(-Fraction(2, 3)) * Op(
             kdelta(-1, -2), kdelta(-3, -4), kdelta(-5, -6)),
         number_op(Fraction(4, 3)) * Op(
             kdelta(-1, -2), kdelta(-3, -6), kdelta(-5, -4)),
         number_op(-Fraction(2, 3)) * Op(
             kdelta(-1, -4), kdelta(-3, -6), kdelta(-5, -2)),
         number_op(Fraction(2, 3)) * Op(
             kdelta(-1, -6), kdelta(-3, -2), kdelta(-5, -4)))),

    (Op(CSU2c(0, 1, 2), sigmaSU2(1, -1, -2),
        CSU2(0, 3, 2), sigmaSU2(3, -3, -4)),
     OpSum(
         number_op(-Fraction(4, 3)) * Op(kdelta(-1, -2), kdelta(-3, -4)),
         number_op(Fraction(8, 3)) * Op(kdelta(-1, -4), kdelta(-3, -2)))),

    (Op(CSU2c(0, 1, 2), sigmaSU2(1, 2, -1),
        CSU2(0, 3, 4), sigmaSU2(3, -2, 4)),
     OpSum(number_op(Fraction(8, 3)) * Op(kdelta(-1, -2)))),

    (Op(CSU2c(0, 1, 2), sigmaSU2(1, 2, -2),
        CSU2(0, 3, -1), sigmaSU2(3, -3, -4)),
      OpSum(
          number_op(-Fraction(2, 3)) * Op(kdelta(-1, -2), kdelta(-3, -4)),
          number_op(2) * Op(kdelta(-1, -4), kdelta(-3, -2)),
          number_op(-Fraction(2, 3)) * Op(kdelta(-1, -3), kdelta(-2, -4)))),

    (Op(CSU2c(0, 1, 2), sigmaSU2(1, -2, -3),
        CSU2(0, 3, -1), sigmaSU2(3, -4, 2)),
     OpSum(number_op(Fraction(8, 3)) * Op(kdelta(-2, -1), kdelta(-4, -3)),
           -number_op(Fraction(4, 3)) * Op(kdelta(-2, -3), kdelta(-4, -1)))),

    (Op(CSU2c(0, 1, -1), sigmaSU2(1, -2, -3),
        CSU2(0, 3, 2), sigmaSU2(3, -4, 2)),
     OpSum(-number_op(Fraction(2, 3)) * Op(kdelta(-1, -4), kdelta(-2, -3)),
           -number_op(Fraction(2, 3)) * Op(kdelta(-1, -3), kdelta(-2, -4)),
           number_op(2) * Op(kdelta(-1, -2), kdelta(-4, -3)))),

    (Op(CSU2c(0, 1, 2), sigmaSU2(1, 2, -2),
        CSU2(0, 3, -1), sigmaSU2(3, -3, -4)),
     OpSum(-number_op(Fraction(2, 3)) * Op(kdelta(-1, -2), kdelta(-3, -4)),
           -number_op(Fraction(2, 3)) * Op(kdelta(-4, -2), kdelta(-3, -1)),
           number_op(2) * Op(kdelta(-1, -4), kdelta(-3, -2)))),

    (Op(CSU2c(0, 1, -1), sigmaSU2(1, 2, -2),
        CSU2(0, 2, 3), sigmaSU2(2, -3, -4)),
     OpSum(-number_op(Fraction(4, 3)) * Op(kdelta(-1, -2), kdelta(-3, -4)),
           number_op(Fraction(8, 3)) * Op(kdelta(-1, -4), kdelta(-3, -2))))]
r"""
Substitute 
:math:`C^I_{ap}\epsilon_{pm}\sigma^a_{ij}
C^{I*}_{bq}\epsilon_{qn}\sigma^b_{kl}` by the equivalent
:math:`-\frac{2}{3}\delta_{mn}\delta_{ij}\delta_{kl}
+\frac{4}{3}\delta_{mn}\delta_{il}\delta_{kj}
+\frac{2}{3}\delta_{ml}\delta_{in}\delta_{kj}
-\frac{2}{3}\delta_{mj}\delta_{il}\delta_{kn}`.
"""

rules_f_sigmas = [
    (Op(fSU2(0, 1, 2), sigmaSU2(0, -1, -2),
        sigmaSU2(1, -3, -4), sigmaSU2(2, -5, -6)),
     OpSum(power_op("sqrt(2)", 1)) *
     OpSum(Op(kdelta(-1, -2), kdelta(-3, -4), kdelta(-5, -6)),
           - Op(kdelta(-1, -2), kdelta(-3, -6), kdelta(-5, -4)),
           - Op(kdelta(-1, -4), kdelta(-3, -2), kdelta(-5, -6)),
           number_op(2) * Op(kdelta(-1, -4), kdelta(-3, -6), kdelta(-5, -2)),
           - Op(kdelta(-1, -6), kdelta(-3, -4) ,kdelta(-5, -2)))),
    (Op(fSU2(-1, 0, 1), sigmaSU2(0, -2, 2), sigmaSU2(1, 2, -3)),
     OpSum(-power_op("sqrt(2)", 1) * Op(sigmaSU2(-1, -2, -3)))),
    (Op(fSU2(1, -1, 0), sigmaSU2(0, -2, 2), sigmaSU2(1, 2, -3)),
     OpSum(-power_op("sqrt(2)", 1) * Op(sigmaSU2(-1, -2, -3)))),
    (Op(fSU2(0, 1, -1), sigmaSU2(0, -2, 2), sigmaSU2(1, 2, -3)),
     OpSum(-power_op("sqrt(2)", 1) * Op(sigmaSU2(-1, -2, -3)))),
    (Op(fSU2(0, -1, 1), sigmaSU2(0, -2, 2), sigmaSU2(1, 2, -3)),
     OpSum(power_op("sqrt(2)", 1) * Op(sigmaSU2(-1, -2, -3))))]

                  

rules_SU2 = ([rule_SU2_fierz, rule_SU2_product_sigmas, rule_SU2_eps_zero] +
             rules_f_sigmas +
             rules_SU2_epsquadruplets_cancel +
             rules_SU2_C_sigma +
             rules_SU2_eps_cancel +
             [rule_SU2_free_eps])
"""All the rules defined in :mod:`matchingtools.extras.SU2` together"""

latex_SU2 = {
    "epsSU2": r"i(\sigma_2)_{{{}{}}}",
    "sigmaSU2": r"\sigma^{}_{{{}{}}}",
    "fSU2": r"f_{{{}{}{}}}",
    "CSU2": r"C^{}_{{{}{}}}",
    "CSU2c": r"C^{{{}*}}_{{{}{}}}",
    "epsSU2triplets": r"\varepsilon_{{{}{}{}}}",
    "epsSU2quadruplets": r"\epsilon_{{{}{}}}",
    "sqrt(2)": r"\sqrt{{2}}"}
r"""
LaTeX code representation of the :math:`SU(2)` tensors.
"""
