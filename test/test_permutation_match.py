import unittest

from matchingtools.matches import Match
from matchingtools.indices import Index
from matchingtools.core import RealField, Statistics


class TestPermutationMatch(unittest.TestCase):
    def setUp(self):
        self.i, self.j = Index.make('i', 'j')
        self.x, self.y = RealField.make(
            'x', 'y',
            statistics=Statistics.BOSON,
            dimension=1
        )

    def test_permutation_with_other_match(self):
        pattern = (
            self.x(self.i)
            * self.x(self.j)
            * self.y(self.i, self.j)
        )._to_operator()

        target = (
            self.x(self.j)
            * self.x(self.i)
            * self.y(self.i, self.j)
        )._to_operator()

        self.assertIsNotNone(
            Match.match_operators(pattern, target)
        )

    def test_permutation_eq(self):
        self.assertEqual(
            self.x(self.i) * self.x(self.j),
            self.x(self.j) * self.x(self.i)
        )


if __name__ == "__main__":
    unittest.main()
