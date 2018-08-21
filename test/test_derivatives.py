import unittest

from matchingtools.indices import Index
from matchingtools.core import RealConstant, RealField, ComplexField
from matchingtools.statistics import Statistics
from matchingtools.matches import Match
from matchingtools.rules import Rule
from matchingtools.shortcuts import D


class TestDerivativesProperties(unittest.TestCase):
    def setUp(self):
        self.mu, self.nu, self.rho = Index.make('mu', 'nu', 'rho')
        self.i, self.j = Index.make('i', 'j')

        self.y, = RealConstant.make('y')

        self.S, = RealField.make(
            'S',
            statistics=Statistics.BOSON,
            dimension=1
        )

        self.V, = ComplexField.make(
            'V',
            statistics=Statistics.BOSON,
            dimension=1
        )

        self.psi, = ComplexField.make(
            'psi',
            statistics=Statistics.BOSON,
            dimension=1.5
        )

        self.operator_1 = (
            self.V(self.mu) * self.psi.c(self.i, self.j)
            * D(self.mu, self.psi(self.i, self.j))
        )

        self.operator_2 = (
            self.y(self.i) * self.S(self.j) * self.psi(self.i, self.j)
        )

    def test_constant(self):
        self.assertEqual(
            D(self.mu, self.y(self.i, self.j)),
            0
        )

    def test_match_single(self):
        self.assertIsNotNone(
            Match.match_operators(
                D(self.mu, self.S(self.i))._to_operator(),
                (self.S(self.i) * self.V(self.mu) * D(self.mu, self.S(self.i)))
                ._to_operator()
            )
        )

    def test_match_double(self):
        operator = (
            D(self.mu, self.S(self.i))
            * D(self.mu, self.S(self.i))
            * self.S(self.j)
            * self.S(self.j)
        )._to_operator()

        operator_crossed = (
            D(self.mu, self.S(self.i))
            * D(self.mu, self.S(self.j))
            * self.S(self.i)
            * self.S(self.j)
        )._to_operator()

        self.assertIsNone(
            Match.match_operators(
                operator,
                operator_crossed
            )
        )

    def test_fields_change(self):
        self.assertNotEqual(
            D(self.mu, self.S(self.i)),
            self.S(self.i)
        )

        self.assertNotEqual(
            D(self.mu, self.V(self.nu)),
            self.V(self.nu)
        )

        self.assertNotEqual(
            D(self.mu, self.V(self.mu)),
            self.V(self.mu)
        )

        self.assertNotEqual(
            D(self.mu, self.psi(self.i, self.j)),
            self.psi(self.i, self.j)
        )

    def test_different_indices(self):
        self.assertNotEqual(
            D(self.mu, self.S(self.i)),
            D(self.nu, self.S(self.i))
        )

        self.assertNotEqual(
            D(self.mu, self.V(self.nu)),
            D(self.rho, self.V(self.nu))
        )

        self.assertNotEqual(
            D(self.mu, self.V(self.mu)),
            D(self.rho, self.V(self.mu))
        )

        self.assertNotEqual(
            D(self.mu, self.psi(self.i, self.j)),
            self.psi(self.i, self.j)
        )

    def test_fields_dimensions(self):
        self.assertEqual(
            D(self.mu, self.S(self.i)).dimension,
            2
        )

        self.assertEqual(
            D(self.mu, D(self.nu, self.V(self.nu))).dimension,
            3
        )

        self.assertEqual(
            D(self.mu, D(self.mu, self.V(self.nu))).dimension,
            3
        )

        self.assertEqual(
            D(self.mu, D(self.mu, self.psi(self.i, self.j))).dimension,
            3.5
        )

    def test_complex_fields(self):
        self.assertEqual(
            D(self.nu, D(self.mu, self.V.c(self.mu))),
            D(self.nu, D(self.mu, self.V(self.mu))).conjugate()
        )

        self.assertEqual(
            D(self.mu, self.psi.c(self.i, self.j)),
            D(self.mu, self.psi(self.i, self.j)).conjugate()
        )

    def test_fields_statistics(self):
        self.assertEqual(
            D(self.mu, self.S(self.i)).statistics,
            self.S(self.i).statistics
        )

        self.assertEqual(
            D(self.mu, self.V(self.nu)).statistics,
            self.V(self.nu).statistics
        )

        self.assertEqual(
            D(self.mu, self.V(self.mu)).statistics,
            self.V(self.mu).statistics
        )

        self.assertEqual(
            D(self.mu, self.psi(self.i, self.j)).statistics,
            self.psi(self.i, self.j).statistics
        )

    def test_linearity(self):
        self.assertEqual(
            D(self.mu, self.operator_1 + self.operator_2),
            D(self.mu, self.operator_1) + D(self.mu, self.operator_2)
        )

    def test_leibniz(self):
        self.assertEqual(
            D(self.nu, self.operator_1),

            D(self.nu, self.V(self.mu))
            * self.psi.c(self.i, self.j)
            * D(self.mu, self.psi(self.i, self.j))

            + self.V(self.mu)
            * D(self.nu, self.psi.c(self.i, self.j))
            * D(self.mu, self.psi(self.i, self.j))

            + self.V(self.mu)
            * self.psi.c(self.i, self.j)
            * D(self.nu, D(self.mu, self.psi(self.i, self.j))),
        )

        self.assertEqual(
            D(self.mu, self.operator_2),

            self.y(self.i)
            * D(self.mu, self.S(self.j))
            * self.psi(self.i, self.j)

            + self.y(self.i)
            * self.S(self.j)
            * D(self.mu, self.psi(self.i, self.j))
        )

    def test_double(self):
        self.assertNotEqual(
            D(self.mu, D(self.mu, self.S(self.i))),
            D(self.mu, D(self.nu, self.S(self.i)))
        )

        self.assertNotEqual(
            D(self.nu, D(self.mu, self.V(self.mu))),
            D(self.mu, D(self.nu, self.V(self.mu)))
        )


class TestReplaceDerivatives(unittest.TestCase):
    def setUp(self):
        self.mu, self.nu, self.rho = Index.make('mu', 'nu', 'rho')
        self.i, self.j, self.k, self.el = Index.make('i', 'j', 'k', 'l')

        self.S, = RealField.make(
            'S',
            statistics=Statistics.BOSON,
            dimension=1
        )

        self.V, = ComplexField.make(
            'V',
            statistics=Statistics.BOSON,
            dimension=1
        )

        self.psi, = ComplexField.make(
            'psi',
            statistics=Statistics.BOSON,
            dimension=1.5
        )

        self.eom_rule = Rule(
            D(self.mu, D(self.mu, self.S(self.i))),
            2 * self.S(self.i)
            + 3 * self.S(self.i) * self.S(self.j) * self.S(self.j)
        )

    def test_eom_simple(self):
        self.assertEqual(
            self.eom_rule.apply(
                2 * D(self.mu, D(self.mu, self.S(self.i)))
            ),
            4 * self.S(self.i)
            + 6 * self.S(self.i) * self.S(self.j) * self.S(self.j)
        )

    def test_eom_times_other(self):
        self.assertEqual(
            self.eom_rule.apply(
                self.psi.c(self.i, self.j) * self.psi(self.j, self.i) *
                self.S(self.k) * D(self.nu, D(self.nu, self.S(self.k)))
            ),
            self.psi.c(self.i, self.j) * self.psi(self.j, self.i)
            * self.S(self.k)
            * (
                2 * self.S(self.k)
                + 3 * self.S(self.k) * self.S(self.el) * self.S(self.el)
            )
        )

    def test_eom_square(self):
        self.assertEqual(
            self.eom_rule.apply(
                D(self.mu, D(self.mu, self.S(self.i)))
                * D(self.nu, D(self.nu, self.S(self.i)))
            ),
            D(self.mu, D(self.mu, self.S(self.i)))
            * (
                2 * self.S(self.i)
                + 3 * self.S(self.i) * self.S(self.j) * self.S(self.j)
            )
        )

    def test_not_eom(self):
        self.assertIsNone(
            Match.match_operators(
                self.eom_rule.pattern._to_operator(),
                self.S(self.i)._to_operator()
            )
        )

        self.assertIsNone(
            Match.match_operators(
                self.eom_rule.pattern._to_operator(),
                D(self.mu, self.S(self.mu))._to_operator()
            )
        )

        self.assertIsNone(
            Match.match_operators(
                self.eom_rule.pattern,
                D(self.mu, D(self.nu, self.S(self.i)))._to_operator()
            )
        )

    def test_not_eom_field(self):
        self.assertIsNone(
            Match.match_operators(
                self.eom_rule.pattern._to_operator(),
                D(self.mu, D(self.mu, self.V(self.rho)))._to_operator()
            )
        )


class TestReplaceByDerivative(unittest.TestCase):
    def setUp(self):
        self.mu, self.i, self.j, self.k = Index.make('mu', 'i', 'j', 'k')

        self.y, = RealConstant.make('y')

        self.S, = RealField.make(
            'S',
            statistics=Statistics.BOSON,
            dimension=1
        )

        self.V, = ComplexField.make(
            'V',
            statistics=Statistics.BOSON,
            dimension=1
        )

        self.psi, = ComplexField.make(
            'psi',
            statistics=Statistics.BOSON,
            dimension=1.5
        )

        self.rule = Rule(
            self.V(self.mu),
            self.psi.c(self.i, self.j) * D(self.mu, self.psi(self.i, self.j))
        )

    def test_with_vector_inside(self):
        self.assertEqual(
            self.rule.apply(
                self.y(self.i)
                * self.S(self.i)
                * D(self.mu, self.V(self.mu))
            ),
            self.y(self.i)
            * self.S(self.i)
            * D(self.mu, self.V(self.mu))
        )

    def test_with_vector_outside(self):
        self.assertEqual(
            self.rule.apply(
                self.y(self.i)
                * D(self.mu, self.S(self.i))
                * self.V(self.mu)
            ),
            self.y(self.k)
            * D(self.mu, self.S(self.k))
            * self.psi.c(self.i, self.j)
            * D(self.mu, self.psi(self.i, self.j))
        )


if __name__ == "__main__":
    unittest.main()
