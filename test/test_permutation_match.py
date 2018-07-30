import unittest

from matchingtools.matches import Match
from matchingtools.indices import Index
from matchingtools.core import RealField, Statistics

i, j = Index.make('i', 'j')
x, y = RealField.make('x', 'y', statistics=Statistics.BOSON, dimension=1)

class TestPermutationMatch(unittest.TestCase):
    def test_permutation_with_other_match(self):
        self.assertIsNotNone(
            Match.match_operators(
                (x(i) * x(j) * y(i, j))._to_operator(),
                (x(j) * x(i) * y(i, j))._to_operator()
            )
        )

    def test_permutation_eq(self):
        self.assertEqual(x(i) * x(j), x(j) * x(i))

if __name__ == "__main__":
    unittest.main()
