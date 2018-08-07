import unittest

from matchingtools.core import ComplexField
from matchingtools.statistics import Statistics
from matchingtools.indices import Index


class TestNotEqual(unittest.TestCase):
    def setUp(self):
        self.i, self.j = Index.make('i', 'j')
        self.x, self.y = ComplexField.make(
            'x', 'y',
            statistics=Statistics.BOSON,
            dimension=1
        )

    def test_different_index(self):
        self.assertNotEqual(
            self.x(self.i),
            self.x(self.j)
        )

    def test_different_name(self):
        self.assertNotEqual(
            self.x(self.i),
            self.y(self.i)
        )

    def test_conjugate(self):
        self.assertNotEqual(
            self.x(self.i),
            self.x.c(self.i)
        )

    def test_different_indices(self):
        self.assertNotEqual(
            self.x(self.i) * self.y(self.j),
            self.x(self.j) * self.y(self.i)
        )

    def test_different_sign(self):
        self.assertNotEqual(
            self.x(self.i) * self.y(self.i),
            -self.y(self.i) * self.x(self.j)
        )

    def test_different_coefficient(self):
        self.assertNotEqual(
            self.x(self.i) * self.y(self.i),
            2 * self.x(self.i) * self.y(self.i)
        )

    def test_different_terms(self):
        self.assertNotEqual(
            self.x(self.i) + self.x(self.i) + self.y(self.i),
            self.x(self.i) + self.y(self.i) + self.y(self.i)
        )


if __name__ == "__main__":
    unittest.main()
