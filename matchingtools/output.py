"""
Module dedicated to the conversion of 
operator sums to plain text and latex code.

Defines the class :class:`Writer` to represent
the coefficients of some given operators in
both formats.
"""

from matchingtools.core import Operator, Op, OpSum, Tensor, number_op, i_op
from matchingtools.transformations import collect, sum_numbers

from fractions import Fraction
from matchingtools.lsttools import concat
from subprocess import call
import copy

def display_tensor_aux(structure, indices, num_of_der):
    for _ in range(num_of_der):
        structure = "(D_{{}}{})".format(structure)
    return structure.format(*list(map(str, indices)))

def display_tensor(tensor, structures, assigned_inds, no_parens=None):
    if tensor.name == "$re":
        ins = " ".join(
            [display_tensor(t, structures, assigned_inds, no_parens)
             for t in tensor.content])
        return r"\operatorname{{Re}}{{\left({}\right)}}".format(ins)
    elif tensor.name == "$im":
        ins = " ".join(
            [display_tensor(t, structures, assigned_inds, no_parens)
             for t in tensor.content])
        return r"\operatorname{{Im}}{{\left({}\right)}}".format(ins)

    base = display_tensor_aux(
        structures[tensor.name],
        list(map(assigned_inds.get, tensor.indices)),
        tensor.num_of_der)
    if tensor.exponent is None:
        exp = ""
    else:
        exp = display_exponent(abs(tensor.exponent))
        if (abs(tensor.exponent) != 1 and
            (no_parens is None or not no_parens(tensor.name))):
            base = r"\left({}\right)".format(base)
    return base + exp

def partition(predicate, lst):
    passed, rest = [], []
    for elem in lst:
        if predicate(elem):
            passed.append(elem)
        else:
            rest.append(elem)
    return passed, rest

def num_of_free_inds(operator):
    return max([max([-i for tensor in operator.tensors
                     for i in tensor.indices] or [0]), 0])

def display_operator(operator, structures, inds, num, no_parens=None,
                     numeric=None):
    if numeric is None:
        numeric = []
    
    # Nice representation for fractions.Fraction and integers
    if isinstance(num, (int, Fraction)):
        pre = "+ " if num > 0 else "- "
        n = abs(num.numerator)
        d = num.denominator
        pre_up = "" if n == 1 else str(n)
        pre_down = "" if d == 1 else str(d)
    else:
        pre = "+ (" + str(num) + ")"
        pre_up = ""
        pre_down = ""
        
    if operator.tensors[0].name == "$i":
        i_up = "i "
        operator = Op(*operator.tensors[1:])
    else:
        i_up = ""

    # Assign the indices    
    assigned_inds = {}
    left_inds = inds[:]
    n = num_of_free_inds(operator)
    for i in range(n):
        assigned_inds[-i - 1] = left_inds[i]
    left_inds = left_inds[n:]
    for tensor in operator.tensors:
        for index in tensor.indices:
            if index not in assigned_inds.keys():
                if index > -1:
                    assigned_inds[index] = left_inds[0]
                    left_inds = left_inds[1:]

    # Main tensors in with positive and negative powers
    numerator = " ".join(
        [display_tensor(tensor, structures, assigned_inds, no_parens)
         for tensor in operator.tensors
         if ((tensor.exponent > -1 or tensor.exponent is None) and
             tensor.name not in numeric)])
    denominator = " ".join(
        [display_tensor(tensor, structures, assigned_inds, no_parens)
         for tensor in operator.tensors
         if ((tensor.exponent < 0 and tensor.exponent is not None) and
             tensor.name not in numeric)])

    # Numeric factors with symbolic expression as tensors
    num_up = " ".join(
        [display_tensor(tensor, structures, assigned_inds, no_parens)
         for tensor in operator.tensors
         if ((tensor.exponent > -1 or tensor.exponent is None) and
             tensor.name in numeric)])
    num_down = " ".join(
        [display_tensor(tensor, structures, assigned_inds, no_parens)
         for tensor in operator.tensors
         if ((tensor.exponent < 0 and tensor.exponent is not None) and
             tensor.name in numeric)])

    return "{} \\frac{{{} {} {} {}}}{{{} {} {}}}".format(
        pre,
        pre_up, num_up, i_up, numerator,
        pre_down, num_down, denominator)

def display_number(number):
    #    return number.latex_code()
    if number == 1:
        return "+"
    elif number == -1:
        return "-"
    elif number == 1j:
        return "+i"
    elif number == -1j:
        return "-i"
    elif number.imag == 0:
        if int(number.real) == number.real:
            return "{:+d}".format(int(number.real))
        else:
            return "{:+.3}".format(number.real)
    elif number.real == 0:
        return "{:+.3}i".format(number.imag)
    else:
        return "+({:+.3} + {1:+.3}i)".format(number.real, number.imag)

def display_exponent(number):
    if number == 1:
        return ""
    elif int(number) == number:
        return "^{{{:d}}}".format(int(number))
    else:
        return "^{{{:.3}}}".format(number)

def collect_conjugates(coef, conjugates):
    if conjugates == None:
        return coef
    
    ctensors = [t for t in conjugates if conjugates[t] != t]
    new_op_sum = OpSum()
    
    def convert_tensor(tensor):
        if tensor.name == "$i":
            return tensor
        new_tensor = copy.copy(tensor)
        new_tensor.name = conjugates[tensor.name]
        return new_tensor

    rest_ops = coef
    while len(rest_ops) > 0:
        op, num = rest_ops[0]
        opc = Operator(map(convert_tensor, op.tensors))
        for i, (other, other_num) in enumerate(rest_ops[1:]):
            if ((other == opc and other_num == num) or
                (other == -opc and other_num == -num)):
                rest_ops = rest_ops[1:i+1] + rest_ops[i+2:]
                new_op_sum += OpSum(
                number_op(2 * num) * real_part(op, ctensors))
                break
            if ((other == -opc and other_num == num) or
                (other == opc and other_num == -num)):
                rest_ops = rest_ops[1:i+1] + rest_ops[i+2:]
                new_op_sum += OpSum(
                number_op(2 * num) * i_op * imaginary_part(op, ctensors))
                break
        else:
            rest_ops = rest_ops[1:]
            new_op_sum += OpSum(number_op(num) * op)
    return sum_numbers(new_op_sum)

def real_part(op, complex_tensors):
    op_complex_tensors = [
        tensor for tensor in op.tensors
        if tensor.name in complex_tensors]
    op_real_tensors = [
        tensor for tensor in op.tensors
        if tensor.name not in complex_tensors]
    indices = concat([tensor.indices for tensor in op_complex_tensors])
    re_tensor = Tensor("$re", indices, content=op_complex_tensors)
    return Op(re_tensor) * Op(*op_real_tensors)

def imaginary_part(op, complex_tensors):
    op_complex_tensors = [
        tensor for tensor in op.tensors
        if tensor.name in complex_tensors]
    op_real_tensors = [
        tensor for tensor in op.tensors
        if tensor.name not in complex_tensors]
    indices = concat([tensor.indices for tensor in op_complex_tensors])
    im_tensor = Tensor("$im", indices, content=op_complex_tensors)
    return Op(im_tensor) * Op(*op_real_tensors)
    
class Writer(object):
    """
    Class to write an operator sum in various formats.
    
    The coefficients of the tensors with the given names
    are collected and prepared to be written.

    Attributes:
        collection (list of pairs (string, list of pairs 
            (complex, Operator)): the first element of each
            pair in the main list is the name of the tensor having
            the second part as coefficient. The pairs in the list
            that is the second element represent a numeric coefficient
            and an Operator (product of tensors) that multiplied
            together and summed with the others give the complete
            coefficient.
        rest (list of pairs (complex, Operator)): represents a sum
            of the operators that couldn't be collected with their
            corresponding coefficients.
        conjugates (dictionary string: string): name of the complex
            conjugate corresponding to the name of each tensor. If
            it is not None, this is used for collecting conjugate
            pairs to give their real or imaginary parts.
        no_parens (list of strings): names of tensors that, when
            having a non-unit exponent are to represented in latex
            as tensor^exp instead of (tensor)^exp
        numeric (list of strings): names of tensors that symbolically
            represent numbers. The effect of this is the latex output:
            they come after the "$number" tensors and before the "$i".
    """
    def __init__(self, op_sum, op_names, conjugates=None, verbose=True):
        """
        Args:
            op_sum (OperatorSum): to be represented
            op_names (list of strings): the names of the tensors
                (represented as tensors) whose coefficients are
                to be collected and written
        """
        self.collection, self.rest = collect(op_sum, op_names, conjugates)
        self.collection = [(key, collect_conjugates(val, conjugates))
                           for key, val in self.collection]

    def __str__(self):
        """
        Plain text representation
        """
        total = "Collected:\n{coll}\nRest:\n{rest}"
        op_form = "  {name}:\n{coef}\n"
        coef_term_form = "    {} {}"
        return total.format(
            coll = "\n".join(op_form.format(
                name = op_name.format(*list(range(-1, -n_inds-1, -1))),
                coef = "\n".join(coef_term_form.format(num, op_coef)
                                  for op_coef, num in coef_lst))
                             for (op_name, n_inds), coef_lst in
                             self.collection),
            rest = "\n".join(str(num) + str(op) for op, num in self.rest))

    def latex_code(self, structures, op_reps, inds, no_parens=None, numeric=None):
        """
        Representation as LaTeX's amsmath ``align`` environments

        Args:
            structures (dict): the keys are the names of all the tensors.
                The corresponding values are the LaTeX formula
                representation, using python`s ``str.format`` notation 
                ``"{}"`` to specify the popsitions where the indices
                should appear.
            structures (dict): the keys are the names of all the operators
                in the basis. The corresponding values are the LaTeX formula
                representation.
            inds (list of strings): the symbols to be used to represent
                the indices, in the order in which they should appear.
            no_parens (function): receives the name of a tensor and returns
                True if its LaTeX representation should not have parenthesis
                around it when exponentiated to some power.
            numeric (list of strings): list of names of tensors that should
                appear with the numeric coefficients in the representation,
                before the rest of the tensors.
        """
        out_str = "Collected operators:\n"
        for (op_name, n_inds), coef_lst in self.collection:
            out_str += r"\begin{align*}" + "\n"
            out_str += op_reps[op_name].format(*inds[:n_inds]) + "= & \n "
            for i, (op_coef, num) in enumerate(coef_lst):
                out_str += display_operator(op_coef, structures, inds,
                                            num, no_parens, numeric)
                if i < len(coef_lst) - 1:
                    if i%3 == 2:
                        out_str += r"\\ &"
                    if i%48 == 47:
                        out_str += "\n" r"\cdots\end{align*}" + "\n"
                        out_str += r"\begin{align*}" + "\n" + r"\cdots &"
                out_str += "\n  "
            out_str += r"\end{align*}" + "\n"
        if self.rest:
            out_str += "\nRest:\n"
            out_str += r"\begin{align*}" + "\n &"
            for i, (op, num) in enumerate(self.rest):
                out_str += display_operator(op, structures, inds, num,
                                            no_parens, numeric)
                out_str += r"\\ &"
                if i%15 == 14:
                    out_str += "\n" + r"+\cdots\end{align*}" + "\n"
                    out_str += r"\begin{align*}" + "\n" + r"\cdots &"
                out_str += "\n"
            out_str += r"\end{align*}" + "\n"
        return out_str

    def write_text_file(self, filename):
        with open(filename, "w") as f:
            f.write(str(self))

    def write_latex(self, filename, structures, op_reps, inds, no_parens=None,
                    numeric=None):
        """
        Write a LaTeX document with the representation.

        Args:
            filename (string): the name of the file without the extension
                ``".tex"`` in which to write
            structures (dict): the keys are the names of all the tensors.
                The corresponding values are the LaTeX formula
                representation, using python`s ``str.format`` notation 
                ``"{}"`` to specify the positions where the indices
                should appear.
            structures (dict): the keys are the names of all the operators
                in the basis. The corresponding values are the LaTeX formula
                representation.
            inds (list of strings): the symbols to be used to represent
                the indices, in the order in which they should appear.
            no_parens (function): receives the name of a tensor and returns
                True if its LaTeX representation should not have parenthesis
                around it when exponentiated to some power.
            numeric (list of strings): list of names of tensors that should
                appear with the numeric coefficients in the representation,
                before the rest of the tensors.
        """
        out_str = self.latex_code(structures, op_reps, inds, no_parens, numeric)
        with open(filename + ".tex", "w") as f:
            f.write((r"\documentclass{{article}}" + "\n" +
                     r"\usepackage{{amsmath}}" + "\n" +
                     r"\usepackage{{amssymb}}" + "\n" +
                     r"\begin{{document}}" + "\n" + "{}" + "\n" +
                     r"\end{{document}}").format(out_str))

    def write_pdf(self, filename, structures, op_reps, inds):
        """
        Directly show the pdf file with the results, obtained
        using ``pdflatex`` on the ``.tex`` file.

        Args:
            filename (string): the name of the files without the extension
                ``".tex"`` and ``.pdf`` in which to write.
            pdfviewer (string): name of the program (callable from the
                command-line) to show the pdf.
            structures (dict): the keys are the names of all the tensors.
                The corresponding values are the LaTeX formula
                representation, using python`s ``str.format`` notation 
                ``"{}"`` to specify the positions where the indices
                should appear.
            structures (dict): the keys are the names of all the operators
                in the basis. The corresponding values are the LaTeX formula
                representation.
            inds (list of strings): the symbols to be used to represent
                the indices, in the order in which they should appear.v
        """
        self.write_latex(filename, structures, op_reps, inds)
        call(["pdflatex", filename + ".tex"])

    def show_pdf(self, filename, pdf_viewer, structures, op_reps, inds):
        """
        Directly show the pdf file with the results, obtained
        using ``pdflatex`` on the ``.tex`` file.

        Args:
            filename (string): the name of the files without the extension
                ``".tex"`` and ``.pdf`` in which to write.
            pdfviewer (string): name of the program (callable from the
                command-line) to show the pdf.
            structures (dict): the keys are the names of all the tensors.
                The corresponding values are the LaTeX formula
                representation, using python`s ``str.format`` notation 
                ``"{}"`` to specify the positions where the indices
                should appear.
            structures (dict): the keys are the names of all the operators
                in the basis. The corresponding values are the LaTeX formula
                representation.
            inds (list of strings): the symbols to be used to represent
                the indices, in the order in which they should appear.v
        """
        self.write_latex(filename, structures, op_reps, inds)
        call(["pdflatex", filename + ".tex"])
        call([pdf_viewer, filename + ".pdf"])


