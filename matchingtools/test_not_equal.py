from core import ComplexField, Statistics

import unittest

i, j = Index.make('i', 'j')

x, y = ComplexField.make('x', 'y', statistics=Statistics.BOSON, dimension=1)

class TestNotEqual(unittest.TestCase):
    def test_different_index(self):
        self.assertNotEqual(x(i), x(j))

    def test_different_name(self):
        self.assertNotEqual(x(i), y(i))

    def test_conjugate(self):
        self.assertNotEqual(x(i), x.c(i))

    def test_different_indices(self):
        self.assertNotEqual(x(i) * y(j), x(j) * y(i))

    def test_different_sign(self):
        self.assertNotEqual(x(i) * y(i), -y(i) * x(j))

    def test_different_coefficient(self):
        self.assertNotEqual(x(i) * y(i), 2 * x(i) * y(i))

    def test_different_terms(self):
        self.assertNotEqual(x(i) + x(i) + y(i), x(i) + y(i) + y(i))


if __name__ == "__main__":
    unittest.main()
