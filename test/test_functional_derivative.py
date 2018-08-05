import unittest

from matchingtools.core import (
    RealField, ComplexField, Statistics, Operator, Kdelta
)
from matchingtools.indices import Index
from matchingtools.shortcuts import D


class TestFunctionalDerivative(unittest.TestCase):
    def setUp(self):
        self.i, self.j, self.a, self.b = Index.make('i', 'j', 'a', 'b')
        self.mu, self.nu = Index.make('mu', 'nu')

        self.x, self.y, self.z = RealField.make(
            'x', 'y', 'z',
            statistics=Statistics.BOSON,
            dimension=1
        )

        self.psi, = ComplexField.make(
            'psi',
            statistics=Statistics.FERMION,
            dimension=1.5
        )

    def test_linear_boson(self):
        self.assertEqual(
            (
                self.x(self.i, self.j)
                * self.y(self.i, self.j)
            ).functional_derivative(
                self.x(self.a, self.b)
            ),

            self.y(self.a, self.b)
        )

    def test_square_boson(self):
        self.assertEqual(
            (
                self.x(self.i, self.j)
                * self.x(self.i, self.j)
            ).functional_derivative(
                self.x(self.a, self.b)
            ),

            2 * self.x(self.a, self.b)
        )

    def test_one_derivative_boson(self):
        self.assertEqual(
            (
                D(self.mu, self.x(self.i, self.i))
                * self.z(self.mu)
            ).functional_derivative(
                D(self.nu, self.x(self.a, self.b))
            ),

            self.z(self.nu) * Kdelta(self.a, self.b)
        )

    def test_linear_fermion_positive(self):
        self.assertEqual(
            (
                self.psi.c(self.i, self.j)
                * self.psi(self.i, self.j)
            ).functional_derivative(
                self.psi.c(self.a, self.b)
            ),

            self.psi(self.a, self.b)
        )

    def test_linear_fermion_negative(self):
        self.assertEqual(
            (
                self.psi.c(self.i, self.j)
                * self.psi(self.i, self.j)
            ).functional_derivative(
                self.psi(self.a, self.b)
            ),

            -self.psi.c(self.a, self.b)
        )

    def test_square_fermion(self):
        self.assertEqual(
            (
                self.psi(self.i, self.j)
                * self.psi(self.i, self.j)
            ).functional_derivative(
                self.psi(self.a, self.b)
            ),

            Operator([], 0)
        )

    def test_derivatives_fermion(self):
        self.assertEqual(
            (
                self.psi(self.i, self.j)
                * D(self.mu, self.psi(self.j, self.a))
                * D(self.mu, D(self.mu, self.psi(self.a, self.i)))
            ).functional_derivative(
                D(self.mu, D(self.nu, self.psi(self.a, self.b)))
            ),

            Kdelta(self.nu, self.mu)
            * self.psi(self.b, self.j)
            * D(self.mu, self.psi(self.j, self.a))
        )


if __name__ == "__init__":
    unittest.main()
