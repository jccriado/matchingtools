import unittest

from matchingtools.core import RealConstant, RealField, Kdelta
from matchingtools.statistics import Statistics
from matchingtools.indices import Index
from matchingtools.rules import Rule


class TestFierz(unittest.TestCase):
    def setUp(self):
        self.a, self.b, self.c, self.d = Index.make('a', 'b', 'c', 'd')
        self.i, self.j, self.k, self.el = Index.make('i', 'j', 'k', 'l')
        self.I, self.J = Index.make('I', 'J')

        self.sigma, = RealConstant.make('sigma')

        self.x, self.y, self.z, self.w = RealField.make(
            'x', 'y', 'z', 'w',
            dimension=1,
            statistics=Statistics.BOSON
        )

        self.rule = Rule(
            self.sigma(self.I, self.a, self.b)
            * self.sigma(self.I, self.c, self.d),

            2 * Kdelta(self.a, self.d) * Kdelta(self.c, self.b)
            - Kdelta(self.a, self.b) * Kdelta(self.c, self.d)
        )

    def test_tensors_same_indices(self):
        self.assertEqual(
            self.rule.apply(
                self.sigma(self.I, self.a, self.b)
                * self.sigma(self.I, self.c, self.d)
            ),
            2 * Kdelta(self.a, self.d) * Kdelta(self.c, self.b)
            - Kdelta(self.a, self.b) * Kdelta(self.c, self.d)
        )

    def test_tensors_different_indices(self):
        self.assertEqual(
            self.rule.apply(
                self.sigma(self.J, self.i, self.j)
                * self.sigma(self.J, self.k, self.el)
            ),
            2 * Kdelta(self.i, self.el) * Kdelta(self.k, self.j)
            - Kdelta(self.i, self.j) * Kdelta(self.k, self.el)
        )

    def test_tensors_same_indices_reorder(self):
        self.assertEqual(
            self.rule.apply(
                self.sigma(self.I, self.a, self.c)
                * self.sigma(self.I, self.d, self.b)
            ),
            2 * Kdelta(self.a, self.b) * Kdelta(self.d, self.c)
            - Kdelta(self.a, self.c) * Kdelta(self.d, self.b)
        )

    def test_tensors_same_indices_reorder_sum(self):
        self.assertEqual(
            self.rule.apply(
                self.sigma(self.I, self.a, self.c)
                * self.sigma(self.I, self.d, self.b)
            ),
            - Kdelta(self.a, self.c) * Kdelta(self.d, self.b)
            + 2 * Kdelta(self.a, self.b) * Kdelta(self.d, self.c)
        )

    def test_fields(self):
        target = (
            self.x(self.i)
            * self.sigma(self.I, self.i, self.j)
            * self.y(self.j)
            * self.z(self.k)
            * self.sigma(self.I, self.k, self.el)
            * self.w(self.el)
        )

        result = (
            2
            * self.x(self.a)
            * self.w(self.a)
            * self.z(self.b)
            * self.y(self.b)

            - self.x(self.a)
            * self.y(self.a)
            * self.z(self.b)
            * self.w(self.b)
        )

        self.assertEqual(self.rule.apply(target), result)

    def test_fields_same(self):
        target = (
            self.x(self.i)
            * self.sigma(self.I, self.i, self.j)
            * self.x(self.j)
            * self.x(self.k)
            * self.sigma(self.I, self.k, self.el)
            * self.x(self.el)
        )

        result = (
            self.x(self.a)
            * self.x(self.a)
            * self.x(self.b)
            * self.x(self.b)
        )

        self.assertEqual(self.rule.apply(target), result)

    def test_tensors_one_contracted_index(self):
        target = (
            self.sigma(self.a, self.i, self.j)
            * self.sigma(self.a, self.i, self.J)
        )
        result = 2 * Kdelta(self.J, self.j) - Kdelta(self.j, self.J)

        self.assertEqual(self.rule.apply(target), result)

    def test_tensors_two_contracted_indices(self):
        target = (
            self.sigma(self.b, self.i, self.j)
            * self.sigma(self.b, self.i, self.j)
        )
        result = Kdelta(self.J, self.J)

        self.assertEqual(self.rule.apply(target), result._to_operator_sum())


if __name__ == "__main__":
    unittest.main()
