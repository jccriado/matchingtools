import unittest

from matchingtools.core import (
    RealField, ComplexField, Statistics, Operator, Kdelta
)
from matchingtools.indices import Index
from matchingtools.shortcuts import D

i, j, a, b, mu, nu = Index.make('i', 'j', 'a', 'b', 'mu', 'nu')
x, y, z = RealField.make(
    'x', 'y', 'z',
    statistics=Statistics.BOSON, dimension=1
)
psi, = ComplexField.make('psi', statistics=Statistics.FERMION, dimension=1.5)


class TestFunctionalDerivative(unittest.TestCase):
    def test_linear_boson(self):
        self.assertEqual(
            (x(i, j) * y(i, j)).functional_derivative(x(a, b)),
            y(a, b)
        )

    def test_square_boson(self):
        self.assertEqual(
            (x(i, j) * x(i, j)).functional_derivative(x(a, b)),
            2 * x(a, b)
        )

    def test_one_derivative_boson(self):
        self.assertEqual(
            (D(mu, x(i, i)) * z(mu)).functional_derivative(D(nu, x(a, b))),
            z(nu) * Kdelta(a, b)
        )

    def test_linear_fermion_positive(self):
        self.assertEqual(
            (psi.c(i, j) * psi(i, j)).functional_derivative(psi.c(a, b)),
            psi(a, b)
        )

    def test_linear_fermion_negative(self):
        self.assertEqual(
            (psi.c(i, j) * psi(i, j)).functional_derivative(psi(a, b)),
            -psi.c(a, b)
        )

    def test_square_fermion(self):
        self.assertEqual(
            (psi(i, j) * psi(i, j)).functional_derivative(psi(a, b)),
            Operator([], 0)
        )

    def test_derivatives_fermion(self):
        self.assertEqual(
            (
                psi(i, j) * D(mu, psi(j, a)) * D(mu, D(mu, psi(a, i)))
            ).functional_derivative(D(mu, D(nu, psi(a, b)))),
            Kdelta(nu, mu) * psi(b, j) * D(mu, psi(j, a))
        )

if __name__ == "__init__":
    unittest.main()
