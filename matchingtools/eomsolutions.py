from matchingtools.rules import Rule

class EOMSolution(object):
    def __init__(self, field_name, replacement):
        self.field_name = field_name
        self.replacement = replacement

    def _rule(self, tensor):
        """ 
        A rule for substituting any tensor by the nth derivative of the
        replacement. The indices of these derivatives are given by the 
        indices of the derivatives of the tensor.
        """
        return Rule(
            tensor,
            self.replacement.nth_derivative(tensor.derivatives_indices)
        )
        
    def _substitute_in_operator(self, operator):
        """
        Replace the first tensor with name self.field_name by the nth
        derivative of the replacement.
        """
        for tensor in operator.tensors:
            if tensor.name == self.field_name:
                return self._rule(tensor).apply(operator)
            
        return operator._to_operator_sum()
        
    def substitute_once(self, target, max_dimension):
        """ Substitute once in each of operator of target. """
        return sum(
            self._substitute_in_operator(operator)
            .filter_by_max_dimension(max_dimension)
            for operator in target._to_operator_sum().operators
        )

    def is_in(self, target):
        """ 
        Find whether there is a tensor in target with name self.field_name.
        """
        for operator in target.operators:
            for tensor in operator.tensors:
                if tensor.name == self.field_name:
                    return True

        return False


class EOMSolutionSystem(object):
    def __init__(self, solutions):
        self.solutions = solutions
        
    def substitute(self, target, max_dimension):
        """ 
        Substitute everywhere iteratively, until there are not any
        remaining substitutions to be made
        """
        for solution in self.solutions:
            target = solution.substitute_once(target, max_dimension)
            
        if any(solution.is_in(target) for solution in self.solutions):
            return self.substitute(target, max_dimension)

        return target

    def solve(self, max_dimension):
        """
        Converts a system of solutions and to a system in which they have
        been used to remove any occurrence of their field names in their 
        replacements.
        """
        new_solution = EOMSolutionSystem([
            EOMSolution(
                target_solution.field_name,
                self.substitute(target_solution.replacement, max_dimension)
                .filter_by_max_dimension(max_dimension)
            )
            for target_solution in self.solutions
        ])

        if any(solution.is_in(target_solution.replacement)
               for solution in self.solutions
               for target_solution in self.solutions):
            return new_solution.solve(max_dimension)

        return new_solution
            
    
    # @staticmethod
    # def substitute(solutions, target, max_dimension):
    #     for solution in solutions:
    #         target = solution.apply(target, max_dimension)

    #     name_in_target = any(
    #         tensor.name in (solution.field_name for solution in solutions)
    #         for operator in target.operators
    #         for tensor in operator.tensors
    #     )

    #     if name_in_target:
    #         return substitute(solutions, target, max_dimension)

    #     return solutions
