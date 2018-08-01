import unittest

from matchingtools.indices import Index
from matchingtools.core import (
    RealConstant, RealField, ComplexField, D, Statistics
)
from matchingtools.integration import Scalar, Vector, integrate_out
from matchingtools.eomsolutions import EOMSolutionSystem

mu, i, j, a = Index.make('mu', 'i', 'j', 'a')
m, g, h = RealConstant.make('m', 'g', 'h')
phi, S = RealField.make('phi', 'S', statistics=Statistics.BOSON, dimension=1)
phi1, S1, V = ComplexField.make(
    'phi1', 'S1', 'V',
    statistics=Statistics.BOSON, dimension=1
)
psi, = ComplexField.make('psi', statistics=Statistics.BOSON, dimension=1)
heavy_S = Scalar(S(i, j, a), flavor_index=a)
heavy_S1 = Scalar(S1(i, j, a), flavor_index=a)
heavy_V = Vector(V(mu), mu)

class TestIntegration(unittest.TestCase):
    def test_integration_real_scalar(self):
        lagrangian = (
            - g(a) * phi(i) * phi(j) * S(i, j, a)
        )

        self.assertEqual(
            integrate_out(lagrangian, [heavy_S], 4),
            1/2 * g(a) * g(a) * heavy_S.mass ** (-2)
            * phi(i) * phi(i) * phi(j) * phi(j)
        )

    def test_integration_complex_scalar(self):
        interaction = (
            - g(a) * phi1(i) * phi1(j) * S1(i, j, a)
        )

        lagrangian = interaction + interaction.conjugate()

        self.assertEqual(
            integrate_out(lagrangian, [heavy_S1], 4),
            g(a) * g(a) * heavy_S1.mass ** (-2)
            * phi1(i) * phi1(j) * phi1.c(i) * phi1.c(j)
        )

    def test_integration_complex_vector(self):
        interaction = h() * V(mu) * psi.c(a) * D(mu, psi(a))
        lagrangian = interaction + interaction.conjugate()

        self.assertEqual(
            integrate_out(lagrangian, [heavy_V], 6),
            - heavy_V.mass**(-2) * h() * h()
            * psi.c(a) * D(mu, psi(a)) * D(mu, psi.c(i)) * psi(i)
        )

if __name__ == "__main__":
    unittest.main()
    TestIntegration().test_integration_complex_vector()
