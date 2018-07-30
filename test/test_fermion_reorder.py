import unittest

from matchingtools.core import RealField, ComplexField, Statistics
from matchingtools.indices import Index
from matchingtools.rules import Rule


alpha, beta, gamma, delta = Index.make('alpha', 'beta', 'gamma', 'delta')

psi, chi = ComplexField.make(
    'psi', 'psi',
    dimension=1.5,
    statistics=Statistics.FERMION
)

phi, = RealField.make('phi', dimension=1, statistics=Statistics.BOSON)

four_fermion_operator, = RealField.make(
    'four_fermion_operator',
    dimension=6,
    statistics=Statistics.BOSON
)

class TestTwoFermionReorder(unittest.TestCase):
    def setUp(self):
        self.rule = Rule(psi.c(alpha) * psi(beta), chi.c(alpha) * chi(beta))

    def test_no_reorder(self):
        self.assertEqual(
            self.rule.apply(psi.c(gamma) * psi(gamma)),
            chi.c(gamma) * chi(gamma)
        )

    def test_reorder(self):
        self.assertEqual(
            self.rule.apply(psi(gamma) * psi.c(gamma)),
            -chi.c(gamma) * chi(gamma)
        )

    def test_reorder_with_boson(self):
        self.assertEqual(
            self.rule.apply(phi(gamma) * psi(gamma) * phi(delta) * psi.c(delta)),
            -phi(gamma) * chi.c(delta) * chi(gamma) * phi(delta)
        )

class TestFourFermionReorder(object): # unittest.TestCase):
    def setUp(self):
        self.rule = Rule(
            psi.c(alpha) * psi(alpha) * chi.c(beta) * chi(beta),
            four_fermion_operator()
        )

    def test_no_reorder(self):
        self.assertEqual(
            psi.c(alpha) * psi(alpha) * chi.c(beta) * chi(beta),
            four_fermion_operator()
        )

    def test_even_reorder_1(self):
        self.assertEqual(
            psi(alpha) * psi.c(alpha) * chi(beta) * chi.c(beta),
            four_fermion_operator()
        )

    def test_even_reorder_2(self):
        self.assertEqual(
            chi.c(beta) * chi(beta) * psi.c(alpha) * psi(alpha),
            four_fermion_operator()
        )

    def test_odd_reorder_1(self):
        self.assertEqual(
            psi.c(alpha) * psi(alpha) * chi(beta) * chi.c(beta),
            -four_fermion_operator()
        )

    def test_odd_reorder_2(self):
        self.assertEqual(
            chi.c(beta) * chi(beta) * psi(alpha) * psi.c(alpha),
            -four_fermion_operator()
        )

if __name__ == "__main__":
    unittest.main()
