import unittest

from matchingtools.indices import Index
from matchingtools.core import (
    RealConstant, RealField, ComplexField, Statistics
)
from matchingtools.matches import Match
from matchingtools.rules import Rule
from matchingtools.shortcuts import D


mu, nu, rho, i, j = Index.make('mu', 'nu', 'rho', 'i', 'j')
y, = RealConstant.make('y')
S, = RealField.make('S', statistics=Statistics.BOSON, dimension=1)
V, = ComplexField.make('V', statistics=Statistics.BOSON, dimension=1)
psi, = ComplexField.make('psi', statistics=Statistics.BOSON, dimension=1.5)

operator_1 = V(mu) * psi.c(i, j) * D(mu, psi(i, j))
operator_2 = y(i) * S(j) * psi(i, j)


class TestDerivativesProperties(unittest.TestCase):
    def test_constant(self):
        self.assertEqual(D(mu, y(i, j)), 0)

    def test_fields_change(self):
        self.assertNotEqual(D(mu, S(i)), S(i))
        self.assertNotEqual(D(mu, V(nu)), V(nu))
        self.assertNotEqual(D(mu, V(mu)), V(mu))
        self.assertNotEqual(D(mu, psi(i, j)), psi(i, j))

    def test_different_indices(self):
        self.assertNotEqual(D(mu, S(i)), D(nu, S(i)))
        self.assertNotEqual(D(mu, V(nu)), D(rho, V(nu)))
        self.assertNotEqual(D(mu, V(mu)), D(rho, V(mu)))
        self.assertNotEqual(D(mu, psi(i, j)), psi(i, j))

    def test_fields_dimensions(self):
        self.assertEqual(D(mu, S(i)).dimension, 2)
        self.assertEqual(D(mu, D(nu, V(nu))).dimension, 3)
        self.assertEqual(D(mu, D(mu, V(nu))).dimension, 3)
        self.assertEqual(D(mu, D(mu, psi(i, j))).dimension, 3.5)

    def test_complex_fields(self):
        self.assertEqual(
            D(nu, D(mu, V.c(mu))),
            D(nu, D(mu, V(mu))).conjugate()
        )
        self.assertEqual(D(mu, psi.c(i, j)), D(mu, psi(i, j)).conjugate())

    def test_fields_statistics(self):
        self.assertEqual(D(mu, S(i)).statistics, S(i).statistics)
        self.assertEqual(D(mu, V(nu)).statistics, V(nu).statistics)
        self.assertEqual(D(mu, V(mu)).statistics, V(mu).statistics)
        self.assertEqual(D(mu, psi(i, j)).statistics, psi(i, j).statistics)

    def test_linearity(self):
        self.assertEqual(
            D(mu, operator_1 + operator_2),
            D(mu, operator_1) + D(mu, operator_2)
        )

    def test_leibniz(self):
        self.assertEqual(
            D(nu, operator_1),
            D(nu, V(mu)) * psi.c(i, j) * D(mu, psi(i, j))
            + V(mu) * D(nu, psi.c(i, j)) * D(mu, psi(i, j))
            + V(mu) * psi.c(i, j) * D(nu, D(mu, psi(i, j))),
        )

        self.assertEqual(
            D(mu, operator_2),
            y(i) * D(mu, S(j)) * psi(i, j)
            + y(i) * S(j) * D(mu, psi(i, j))
        )

    def test_double(self):
        self.assertNotEqual(D(mu, D(mu, S(i))), D(mu, D(nu, S(i))))
        self.assertNotEqual(D(nu, D(mu, V(mu))), D(mu, D(nu, V(mu))))


class TestDerivativesRules(unittest.TestCase):
    def setUp(self):
        self.eom_rule = Rule(
            D(mu, D(mu, S(i))),
            2 * S(i) + 3 * S(i) * S(j) * S(j)
        )

    def test_eom_simple(self):
        self.assertEqual(
            self.eom_rule.apply(2 * D(mu, D(mu, S(i)))),
            4 * S(i) + 6 * S(i) * S(j) * S(j)
        )

    def test_eom_times_other(self):
        self.assertEqual(
            self.eom_rule.apply(psi.c(i, j) * psi(j, i) * D(nu, D(nu, S(i)))),
            psi.c(i, j) * psi(j, i) * (2 * S(i) + 3 * S(i) * S(j) * S(j))
        )

    def test_eom_square(self):
        self.assertEqual(
            self.eom_rule.apply(D(mu, D(mu, S(i))) * D(nu, D(nu, S(i)))),
            D(mu, D(mu, S(i))) * (2 * S(i) + 3 * S(i) * S(j) * S(j))
        )

    def test_not_eom(self):
        self.assertIsNone(
            Match.match_operators(
                self.eom_rule.pattern._to_operator(),
                S(i)._to_operator()
            )
        )

        self.assertIsNone(
            Match.match_operators(
                self.eom_rule.pattern._to_operator(),
                D(mu, S(mu))._to_operator()
            )
        )

        self.assertIsNone(
            Match.match_operators(
                self.eom_rule.pattern,
                D(mu, D(nu, S(i)))._to_operator()
            )
        )

    def test_not_eom_field(self):
        self.assertIsNone(
            Match.match_operators(
                self.eom_rule.pattern._to_operator(),
                D(mu, D(mu, V(rho)))._to_operator()
            )
        )


class TestReplaceByDerivative(unittest.TestCase):
    def setUp(self):
        self.rule = Rule(V(mu), psi.c(i, j) * D(mu, psi(i, j)))

    def test_with_vector_inside(self):
        self.assertEqual(
            self.rule.apply(y(i) * S(i) * D(mu, V(mu))),
            y(i) * S(i) * D(mu, V(mu))
        )

    def test_with_vector_outside(self):
        self.assertEqual(
            self.rule.apply(y(i) * D(mu, S(i)) * V(mu)),
            y(i) * D(mu, S(i)) * psi.c(i, j) * D(mu, psi(i, j))
        )


if __name__ == "__main__":
    unittest.main()
