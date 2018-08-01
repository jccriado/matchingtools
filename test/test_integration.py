import unittest

from matchingtools.indices import Index
from matchingtools.core import RealConstant, RealField, D, Statistics
from matchingtools.integration import Scalar, integrate_out

mu, i, j, a = Index.make('mu', 'i', 'j', 'a')
m, g = RealConstant.make('m', 'g')
phi, S = RealField.make('phi', 'S', statistics=Statistics.BOSON, dimension=1)
heavy_S = Scalar(S(i, j, a), flavor_index=a)

class TestIntegration(unittest.TestCase):
    def test_integration(self):
        lagrangian = (
            - g(a) * phi(i) * phi(j) * S(i, j, a)
        )

        self.assertEqual(
            integrate_out(lagrangian, [heavy_S], 4),
            1/2 * g(a) * g(a) * heavy_S.mass ** (-2)
            * phi(i) * phi(i) * phi(j) * phi(j)
        )


if __name__ == "__main__":
    unittest.main()
