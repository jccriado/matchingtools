import unittest

from matchingtools.core import RealConstant, RealField
from matchingtools.statistics import Statistics
from matchingtools.indices import Index
from matchingtools.shortcuts import D


mu, i, j, a = Index.make('mu', 'i', 'j', 'a')
m, g = RealConstant.make('m', 'g')
phi, = RealField.make('phi', statistics=Statistics.BOSON, dimension=1)


class TestVariation(unittest.TestCase):
    def test_variation(self):
        lagrangian = (
            1/2 * D(mu, phi(i)) * D(mu, phi(i))
            - 1/2 * m() * phi(i) * phi(i)
            - g() * phi(i) * phi(i) * phi(j) * phi(j)
        )

        variation = (
            - D(mu, D(mu, phi(a)))
            - m() * phi(a)
            - 4 * g() * phi(a) * phi(i) * phi(i)
        )

        self.assertEqual(lagrangian.variation(phi(a)), variation)


if __name__ == "__main__":
    unittest.main()
