import unittest

from matchingtools.utils import Cycle, Permutation


class TestCycle(unittest.TestCase):
    def test_cycle_eq(self):
        self.assertEqual(Cycle([0, 1, 2]), Cycle([1, 2, 0]))
        self.assertNotEqual(Cycle([0, 1, 2]), Cycle([1, 0, 2]))
        self.assertNotEqual(Cycle([0, 1, 2]), Cycle([0, 1, 2, 3]))


class TestPermutation(unittest.TestCase):
    def test_compare(self):
        pi = Permutation.compare(
            [6, 7, 8, 9, 10],
            [7, 9, 10, 6, 8]
        )

        self.assertEqual(pi, Permutation([3, 0, 4, 1, 2]))

    def test_get_cycles(self):
        cycles = Permutation([2, 4, 3, 0, 5, 1]).get_cycles()

        self.assertEqual(len(cycles), 2)
        self.assertEqual(cycles[0], Cycle([0, 2, 3]))
        self.assertEqual(cycles[1], Cycle([4, 5, 1]))

    def test_parity(self):
        self.assertEqual(Permutation([0, 1, 2]).parity, 1)
        self.assertEqual(Permutation([0, 2, 1]).parity, -1)
        self.assertEqual(Permutation([1, 2, 0]).parity, 1)




if __name__ == "__main__":
    unittest.main()
