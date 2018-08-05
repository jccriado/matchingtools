import unittest

from matchingtools.indices import Index
from matchingtools.core import RealField, ComplexField, Statistics


class TestConjugates(unittest.TestCase):
    def setUp(self):
        self.i, self.j = Index.make('i', 'j')

        self.x, self.y = RealField.make(
            'x', 'y',
            statistics=Statistics.BOSON, dimension=1
        )

        self.psi, self.chi = ComplexField.make(
            'psi', 'chi',
            statistics=Statistics.FERMION, dimension=1.5
        )

    def test_conjugate_real(self):
        self.assertEqual(
            self.x(self.i),
            (self.x(self.i)).conjugate()
        )

    def test_conjugate_real_product(self):
        self.assertEqual(
            self.x(self.i) * self.y(self.i),
            (self.x(self.i) * self.y(self.i)).conjugate()
        )

    def test_conjugate_complex(self):
        self.assertNotEqual(
            self.psi(self.i),
            (self.psi(self.i)).conjugate()
        )

    def test_conjugate_complex_product(self):
        self.assertNotEqual(
            self.psi(self.i) * self.chi(self.i),
            (self.psi(self.i) * self.chi(self.i)).conjugate()
        )

    def test_double_conjugate_complex(self):
        self.assertEqual(
            self.psi(self.i),
            (self.psi(self.i)).conjugate().conjugate()
        )

    def test_real_combination(self):
        operator_sum = (
            self.x(self.i) * self.x(self.j)
            * self.psi(self.i) * self.chi(self.j)
            + self.y(self.i) * self.y(self.i)
            * self.chi(self.j) * self.chi(self.j)
        )

        hermitian_combination = operator_sum + operator_sum.conjugate()

        self.assertNotEqual(
            operator_sum,
            operator_sum.conjugate()
        )

        self.assertEqual(
            hermitian_combination,
            hermitian_combination.conjugate()
        )


if __name__ == "__main__":
    unittest.main()
