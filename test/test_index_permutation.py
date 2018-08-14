import unittest

from matchingtools.invertibles import MassMatrix, EpsilonUp
from matchingtools.indices import Index


class TestIndexPermutation(unittest.TestCase):
    def test_symmetric_mass(self):
        i, j = Index.make('i', 'j')

        self.assertEqual(
            MassMatrix('f', i, j)._to_operator(),
            MassMatrix('f', j, i)._to_operator()
        )

    def test_antisymmetric_epsilon_up(self):
        i, j = Index.make('i', 'j')

        self.assertEqual(
            EpsilonUp(i, j)._to_operator(),
            -EpsilonUp(j, i)._to_operator()
        )


if __name__ == "__main__":
    unittest.main()
