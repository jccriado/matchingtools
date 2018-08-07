import unittest

from matchingtools.core import RealField, ComplexField
from matchingtools.statistics import Statistics
from matchingtools.indices import Index
from matchingtools.rules import Rule


class TestTwoFermionReorder(unittest.TestCase):
    def setUp(self):
        self.alpha, self.beta, self.gamma, self.delta = Index.make(
            'alpha', 'beta', 'gamma', 'delta'
        )

        self.psi, self.chi = ComplexField.make(
            'psi', 'chi',
            dimension=1.5,
            statistics=Statistics.FERMION
        )

        self.phi, = RealField.make(
            'phi',
            dimension=1,
            statistics=Statistics.BOSON
        )

        self.rule = Rule(
            self.psi.c(self.alpha) * self.psi(self.beta),
            self.chi.c(self.alpha) * self.chi(self.beta)
        )

    def test_no_reorder(self):
        self.assertEqual(
            self.rule.apply(self.psi.c(self.gamma) * self.psi(self.gamma)),
            self.chi.c(self.gamma) * self.chi(self.gamma)
        )

    def test_reorder(self):
        self.assertEqual(
            self.rule.apply(self.psi(self.gamma) * self.psi.c(self.gamma)),
            -self.chi.c(self.gamma) * self.chi(self.gamma)
        )

    def test_reorder_with_boson(self):
        self.assertEqual(
            self.rule.apply(
                self.phi(self.gamma)
                * self.psi(self.gamma)
                * self.phi(self.delta)
                * self.psi.c(self.delta)
            ),
            -self.phi(self.gamma) * self.chi.c(self.delta)
            * self.chi(self.gamma) * self.phi(self.delta)
        )


class TestFourFermionReorder(unittest.TestCase):
    def setUp(self):
        self.alpha, self.beta, self.gamma, self.delta = Index.make(
            'alpha', 'beta', 'gamma', 'delta'
        )

        self.psi, self.chi = ComplexField.make(
            'psi', 'chi',
            dimension=1.5,
            statistics=Statistics.FERMION
        )

        self.four_fermion_operator, = RealField.make(
            'four_fermion_operator',
            dimension=6,
            statistics=Statistics.BOSON
        )

        self.rule = Rule(
            self.psi.c(self.alpha)
            * self.psi(self.alpha)
            * self.chi.c(self.beta)
            * self.chi(self.beta),
            self.four_fermion_operator()._to_operator_sum()
        )

    def test_no_reorder(self):
        self.assertEqual(
            self.rule.apply(
                self.psi.c(self.alpha)
                * self.psi(self.alpha)
                * self.chi.c(self.beta)
                * self.chi(self.beta)
            ),
            self.four_fermion_operator()._to_operator_sum()
        )

    def test_even_reorder_1(self):
        self.assertEqual(
            self.rule.apply(
                self.psi(self.alpha)
                * self.psi.c(self.alpha)
                * self.chi(self.beta)
                * self.chi.c(self.beta)
            ),
            self.four_fermion_operator()._to_operator_sum()
        )

    def test_even_reorder_2(self):
        self.assertEqual(
            self.rule.apply(
                self.chi.c(self.beta)
                * self.chi(self.beta)
                * self.psi.c(self.alpha)
                * self.psi(self.alpha)
            ),
            self.four_fermion_operator()._to_operator_sum()
        )

    def test_odd_reorder_1(self):
        self.assertEqual(
            self.rule.apply(
                self.psi.c(self.alpha)
                * self.psi(self.alpha)
                * self.chi(self.beta)
                * self.chi.c(self.beta)
            ),
            -self.four_fermion_operator()._to_operator_sum()
        )

    def test_odd_reorder_2(self):
        self.assertEqual(
            self.rule.apply(
                self.chi.c(self.beta)
                * self.chi(self.beta)
                * self.psi(self.alpha)
                * self.psi.c(self.alpha)
            ),
            -self.four_fermion_operator()._to_operator_sum()
        )


if __name__ == "__main__":
    unittest.main()
