from matchingtools.core import RealConstant, RealField, Kdelta, Statistics
from matchingtools.indices import Index
from matchingtools.rules import Rule

import unittest

a, b, c, d = Index.make('a', 'b', 'c', 'd')
i, j, k, l, I, J = Index.make('i', 'j', 'k', 'l', 'I', 'J')

sigma, = RealConstant.make('sigma')

x, y, z, w = RealField.make(
    'x', 'y', 'z', 'w',
    dimension=1,
    statistics=Statistics.BOSON
)

fierz_rule = Rule(
    sigma(I, a, b) * sigma(I, c, d),
    2 * Kdelta(a, d) * Kdelta(c, b) - Kdelta(a, b) * Kdelta(c, d)
)

class TestFierz(unittest.TestCase):
    def setUp(self):
        self.rule = Rule(
            sigma(I, a, b) * sigma(I, c, d),
            2 * Kdelta(a, d) * Kdelta(c, b) - Kdelta(a, b) * Kdelta(c, d)
        )

    def test_tensors_same_indices(self):
        self.assertEqual(
            self.rule.apply(sigma(I, a, b) * sigma(I, c, d)),
            2 * Kdelta(a, d) * Kdelta(c, b) - Kdelta(a, b) * Kdelta(c, d)
        )

    def test_tensors_different_indices(self):
        self.assertEqual(
            self.rule.apply(sigma(J, i, j) * sigma(J, k, l)),
            2 * Kdelta(i, l) * Kdelta(k, j) - Kdelta(i, j) * Kdelta(k, l)
        )

    def test_tensors_same_indices_reorder(self):
        self.assertEqual(
            self.rule.apply(sigma(I, a, c) * sigma(I, d, b)),
            2 * Kdelta(a, b) * Kdelta(d, c) - Kdelta(a, c) * Kdelta(d, b)
        )

    def test_tensors_same_indices_reorder_sum(self):
        self.assertEqual(
            self.rule.apply(sigma(I, a, c) * sigma(I, d, b)),
            - Kdelta(a, c) * Kdelta(d, b) + 2 * Kdelta(a, b) * Kdelta(d, c) 
        )

    def test_fields(self):
        target = x(i) * sigma(I, i, j) * y(j) * z(k) * sigma(I, k, l) * w(l)
        result = 2 * x(a) * w(a) * z(b) * y(b) - x(a) * y(a) * z(b) * w(b)
                    
        self.assertEqual(self.rule.apply(target), result)

    def test_fields_same(self):
        target = x(i) * sigma(I, i, j) * x(j) * x(k) * sigma(I, k, l) * x(l)
        result = x(a) * x(a) * x(b) * x(b)
                    
        self.assertEqual(self.rule.apply(target), result)

    def test_tensors_one_contracted_index(self):
        target = sigma(a, i, j) * sigma(a, i, J)
        result = 2 * Kdelta(J, j) - Kdelta(j, J)

        self.assertEqual(self.rule.apply(target), result)
    def test_tensors_two_contracted_indices(self):
        target = sigma(b, i, j) * sigma(b, i, j)
        result = Kdelta(J, J)

        self.assertEqual(self.rule.apply(target), result._to_operator_sum())



if __name__ == "__main__":
    unittest.main()
