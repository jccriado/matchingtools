import unittest

from matchingtools.core import RealConstant, RealField, ComplexField
from matchingtools.statistics import Statistics
from matchingtools.indices import Index
from matchingtools.invertibles import MassMatrix
from matchingtools.integration import scalar_quadratic_terms, integrate_out


class TestIntegrationRealScalar(unittest.TestCase):
    def setUp(self):
        self.i, self.j, self.k, self.el = Index.make('i', 'j', 'k', 'el')
        self.mu, self.a, self.b, self.c = Index.make('mu', 'a', 'b', 'c')
        self.alpha, self.alpha_dot, self.beta, self.beta_dot = Index.make(
            'alpha', 'alpha_dot', 'beta', 'beta_dot'
        )

        self.kappa, self.sigma, = RealConstant.make('kappa', 'sigma')
        self.phi, = ComplexField.make(
            'phi',
            statistics=Statistics.BOSON,
            dimension=1
        )
        self.Xi, = RealField.make(
            'Xi',
            statistics=Statistics.BOSON,
            dimension=1
        )

        interaction_lagrangian = (
            -self.kappa(self.k)
            * self.phi.c(self.i)
            * self.sigma(self.a, self.i, self.j)
            * self.phi(self.j)
            * self.Xi(self.a, self.k)
        )

        self.lagrangian = (
            interaction_lagrangian
            + scalar_quadratic_terms(
                self.Xi(self.b, self.el),
                flavor_index=self.el
            )
        )

    def test_integration(self):
        effective_lagrangian = integrate_out(
            self.lagrangian,
            [self.Xi(self.a, self.i)],
            4
        )

        self.assertEqual(
            effective_lagrangian,
            1/2 * MassMatrix('Xi', self.b, self.c, exponent=-2)
            * self.kappa(self.b) * self.kappa(self.c)
            * self.sigma(self.a, self.i, self.j)
            * self.sigma(self.a, self.k, self.el)
            * self.phi.c(self.i) * self.phi(self.j)
            * self.phi.c(self.k) * self.phi(self.el)
        )


if __name__ == '__main__':
    unittest.main()
