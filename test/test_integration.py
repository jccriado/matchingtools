import unittest

from matchingtools.core import (
    RealConstant, RealField, ComplexField, Statistics, Kdelta,
    epsilon_up, epsilon_down
)
from matchingtools.indices import Index
from matchingtools.integration import (
    Scalar, Vector, DiracFermion, integrate_out
)
from matchingtools.rules import Rule


mu, i, j, k, a = Index.make('mu', 'i', 'j', 'k', 'a')
alpha, alpha_dot, beta, beta_dot = Index.make(
    'alpha', 'alpha_dot', 'beta', 'beta_dot'
)
m, g, h, y = RealConstant.make('m', 'g', 'h', 'y')
phi, S = RealField.make('phi', 'S', statistics=Statistics.BOSON, dimension=1)
phi1, S1, V = ComplexField.make(
    'phi1', 'S1', 'V',
    statistics=Statistics.BOSON, dimension=1
)
psi, FL, FR = ComplexField.make(
    'psi', 'FL', 'FR',
    statistics=Statistics.FERMION, dimension=1.5
)

heavy_S = Scalar(S(i, j, a), flavor_index=a)
heavy_S1 = Scalar(S1(i, j, a), flavor_index=a)
heavy_V = Vector(V(mu), mu)
heavy_F = DiracFermion(
    'F',
    FL(alpha, i), FR(beta_dot, i),
    alpha, beta_dot,
    flavor_index=i
)

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
        pass
        # interaction = h() * V(mu) * phi(a) * D(mu, phi(a))
        # lagrangian = interaction + interaction.conjugate()

        # self.assertEqual(
        #     integrate_out(lagrangian, [heavy_V], 6),
        #     - heavy_V.mass**(-2) * h() * h()
        #     * phi(a) * D(mu, phi(a)) * D(mu, phi(i)) * phi(i)
        # )

    def test_integration_Dirac_fermion(self):
        interaction = (
            y(i, j) * FL.c(alpha_dot, i)
            * phi(a) * psi(alpha_dot, a, j)
        )
        lagrangian = interaction + interaction.conjugate()

        effective_lagrangian = integrate_out(lagrangian, [heavy_F], 6)

        epsilon_rules = [
            Rule(epsilon_down(i, j), epsilon_up(j, i)),
            Rule(epsilon_up(i, j) * epsilon_up(j, k), Kdelta(i, k)),
            Rule(epsilon_up(i, j) * epsilon_up(k, j), -Kdelta(i, k)),
            Rule(epsilon_up(j, i) * epsilon_up(j, k), -Kdelta(i, k)),
            Rule(epsilon_up(j, i) * epsilon_up(k, j), Kdelta(i, k))
        ]

        for _ in range(3):
            for rule in epsilon_rules:
                effective_lagrangian = rule.apply(effective_lagrangian)

        print(effective_lagrangian)



if __name__ == "__main__":
    unittest.main()
    TestIntegration().test_integration_complex_vector()
