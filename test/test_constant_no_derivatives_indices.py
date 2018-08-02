import unittest

from matchingtools.core import RealConstant, ComplexConstant
from matchingtools.indices import Index


class TestConstantNoDerivativesIndices(unittest.TestCase):
    def setUp(self):
        self.i = Index('i')

    def test_exception_raised(self):
        with self.assertRaises(AssertionError):
            RealConstant('x', derivatives_indices=[self.i])

        with self.assertRaises(AssertionError):
            ComplexConstant('y', derivatives_indices=[self.i])


if __name__ == "__main__":
    unittest.main()
