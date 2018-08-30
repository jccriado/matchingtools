import unittest

from matchingtools.indices import Index
from matchingtools.core import (
    RealConstant, ComplexConstant, ComplexField, Kdelta, Operator
)
from matchingtools.bases import Basis
from matchingtools.shortcuts import D
from matchingtools.identities import Identity


class TestBasis(unittest.TestCase):
    def test_simple_basis_Higgs(self):
        i, j, k, l, m = Index.make('i', 'j', 'k', 'l', 'm')
        I, mu, nu = Index.make('I', 'mu', 'nu')
        a, b, c, d, lambda_ = RealConstant.make('a', 'b', 'c', 'd', 'lambda')
        sigma, = ComplexConstant.make('sigma')
        phi, = ComplexField.make('phi', dimension=1)

        lagrangian = (
            D(mu, phi.c(i)) * D(mu, phi(i))
            - lambda_() * phi.c(i) * phi(i) * phi.c(j) * phi(j)
            + a() * phi.c(i) * phi(i) * phi.c(j) * phi(j) * phi.c(k) * phi(k)
            + b() * phi.c(i) * phi(i) * phi.c(j) * D(mu, D(mu, phi(j)))
            + b() * phi.c(i) * phi(i) * phi(j) * D(mu, D(mu, phi.c(j)))
            + c() * D(mu, phi.c(i)) * phi(i) * D(mu, phi.c(j)) * phi(j)
            + c() * D(mu, phi(i)) * phi.c(i) * D(mu, phi(j)) * phi.c(j)
            + d() * phi.c(i) * sigma(I, i, j) * phi(j)
            * phi.c(k) * sigma(I, k, l) * phi(l)
            * phi.c(m) * phi(m)
        )

        basis = Basis(
            {
                'kin_phi': D(mu, phi.c(i)) * D(mu, phi(i)),
                'phi4': phi.c(i) * phi(i) * phi.c(j) * phi(j),
                'phi6':
                phi.c(i) * phi(i) * phi.c(j) * phi(j) * phi.c(k) * phi(k),
                'phiD': phi.c(i) * D(mu, phi(i)) * D(mu, phi.c(j)) * phi(j),
                'phi3': D(mu, phi.c(i)) * D(mu, phi(i)) * phi.c(j) * phi(j)
            },
            [phi(i)]
        )

        fierz_identity = Identity.equals(
            sigma(I, i, j) * sigma(I, k, l),
            2 * Kdelta(i, l) * Kdelta(k, j) - Kdelta(i, j) * Kdelta(k, l)
        )

        result = basis.compute_wilson_coefficients(
            lagrangian,
            [fierz_identity]
        )

        expected_result = {
            'kin_phi': Operator([], 1),
            'phi4': -lambda_(),
            'phi6': a() + d() + 4 * lambda_() * (b() - c()),
            'phiD': -2 * c(),
            'phi3': -2 * c()
        }

        self.assertEqual(result[0], expected_result)
        self.assertEqual(result[1], 0)


if __name__ == '__main__':
    unittest.main()
