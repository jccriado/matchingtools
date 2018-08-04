import unittest

from matchingtools.indices import Index
from matchingtools.core import RealField, ComplexField, Statistics

i, j = Index.make('i', 'j')

x, y = RealField.make(
    'x', 'y',
    statistics=Statistics.BOSON, dimension=1
)

psi, chi = ComplexField.make(
    'psi', 'chi',
    statistics=Statistics.FERMION, dimension=1.5
)


class TestConjugates(unittest.TestCase):
    def test_conjugate_real(self):
        self.assertEqual(x(i), (x(i)).conjugate())

    def test_conjugate_real_product(self):
        self.assertEqual(x(i) * y(i), (x(i) * y(i)).conjugate())

    def test_conjugate_complex(self):
        self.assertNotEqual(psi(i), (psi(i)).conjugate())

    def test_conjugate_complex_product(self):
        self.assertNotEqual(psi(i) * chi(i), (psi(i) * chi(i)).conjugate())

    def test_double_conjugate_complex(self):
        self.assertEqual(psi(i), (psi(i)).conjugate().conjugate())

    def test_real_combination(self):
        operators = (
            x(i) * x(j) * psi(i) * chi(j)
            + y(i) * y(i) * chi(j) * chi(j)
        )

        hermitian = operators + operators.conjugate()

        self.assertNotEqual(operators, operators.conjugate())
        self.assertEqual(hermitian, hermitian.conjugate())


if __name__ == "__main__":
    unittest.main()
