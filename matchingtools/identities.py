from matchingtools.core import OperatorSum
from matchingtools.rules import Rule


class Identity(object):
    def __init__(self, operators):
        self.operators = operators

    @staticmethod
    def equals(lhs, rhs):
        return Identity(lhs.operators + (-rhs).operators)

    def rules(self):
        return (
            [
                Rule(
                    -operator,
                    OperatorSum(
                        self.operators[:pos] + self.operators[pos+1:]
                    )
                )
                for pos, operator in enumerate(self.operators)
            ]
            + [
                Rule(
                    -operator.conjugate(),
                    OperatorSum(
                        self.operators[:pos] + self.operators[pos+1:]
                    ).conjugate()
                )
                for pos, operator in enumerate(self.operators)
            ]
        )
