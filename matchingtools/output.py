"""
Module dedicated to the conversion of 
operator sums to plain text and latex code.

Defines the class :class:`Writer` to represent
the coefficients of some given operators in
both formats.
"""

from matchingtools.transformations import collect

from subprocess import call

def display_tensor_aux(structure, indices, num_of_der):
    for _ in range(num_of_der):
        structure = "(D_{{}}{})".format(structure)
    return structure.format(*list(map(str, indices)))

def display_tensor(tensor, structures, assigned_inds):
    base = display_tensor_aux(
        structures[name(tensor.name)],
        list(map(assigned_inds.get, tensor.indices)),
        tensor.num_of_der)
    exp = display_exponent(abs(exponent(tensor.name)))
    return base + exp

def exponent(name):
    if name[0] == "{" and name[-1] == "}":
        return float(name[name.index("^")+1:-1])
    return 1

def name(name):
    if name[0] == "{" and name[-1] == "}":
        return name[1:name.index("^")]
    return name

def partition(predicate, lst):
    passed, rest = [], []
    for elem in lst:
        if predicate(elem):
            passed.append(elem)
        else:
            rest.append(elem)
    return passed, rest

def num_of_free_inds(operator):
    return max([max(-i for tensor in operator.tensors
                    for i in tensor.indices), 0])

def display_operator(operator, structures, inds):
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
    numerator = " ".join([display_tensor(tensor, structures, assigned_inds)
                          for tensor in operator.tensors
                          if exponent(tensor.name) > -1])
    denominator = " ".join([display_tensor(tensor, structures, assigned_inds)
                            for tensor in operator.tensors
                            if exponent(tensor.name) < 0])
    return "\\frac{{{}}}{{{}}}".format(numerator, denominator)

def display_number(number):
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
    """
    def __init__(self, op_sum, op_names, verbose=True):
        """
        Args:
            op_sum (OperatorSum): to be represented
            op_names (list of strings): the names of the tensors
                (represented as tensors) whose coefficients are
                to be collected and written
        """
        self.collection, self.rest = collect(op_sum, op_names, verbose)
        self.verbose = verbose

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

    def latex_code(self, structures, op_reps, inds):
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
                the indices, in the order in which they should appear.v
        """
        out_str = "Collected operators:\n"
        for (op_name, n_inds), coef_lst in self.collection:
            out_str += r"\begin{align*}" + "\n"
            out_str += op_reps[op_name].format(*inds[:n_inds]) + "= & \n "
            for i, (op_coef, num) in enumerate(coef_lst):
                out_str += display_number(num) + " "
                out_str += display_operator(op_coef, structures, inds)
                if i < len(coef_lst) - 1:
                    if i%3 == 2:
                        out_str += r"\\ &"
                    if i%50 == 49:
                        out_str += "\n" r"\cdots\end{align*}" + "\n"
                        out_str += r"\begin{align*}\cdots"
                out_str += "\n  "
            out_str += r"\end{align*}" + "\n"
        if self.rest:
            out_str += "\nRest:\n"
            out_str += r"\begin{align*}" + "\n &"
            for i, (op, num) in enumerate(self.rest):
                out_str += display_number(num) + " "
                out_str += display_operator(op, structures, inds)
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

    def write_latex(self, filename, structures, op_reps, inds):
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
                the indices, in the order in which they should appear.v
        """
        out_str = self.latex_code(structures, op_reps, inds)
        with open(filename + ".tex", "w") as f:
            f.write((r"\documentclass{{article}}" + "\n" +
                     r"\usepackage{{amsmath}}" + "\n" +
                     r"\usepackage{{amssymb}}" + "\n" +
                     r"\begin{{document}}" + "\n" + "{}" + "\n" +
                     r"\end{{document}}").format(
                         self.latex_code(structures, op_reps, inds)))

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


