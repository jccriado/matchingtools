from efttools.algebra import collect

import sys

from subprocess import call

def display_tensor(structure, indices):
    return structure.format(*map(str, indices))

def power(name):
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

def display_operator(operator, structures, inds):
    assigned_inds = {}
    left_inds = inds[:]
    for tensor in operator.tensors:
        for index in tensor.indices:
            if index not in assigned_inds.keys():
                if index < 0:
                    assigned_inds[index] = left_inds[0]
                    left_inds = left_inds[1:]
    for tensor in operator.tensors:
        for index in tensor.indices:
            if index not in assigned_inds.keys():
                if index > -1:
                    assigned_inds[index] = left_inds[0]
                    left_inds = left_inds[1:]
    numerator = " ".join([display_tensor(structures.get(name(tensor.name), "T"),
                                         map(assigned_inds.get, tensor.indices)) +
                          ("^{" + display_number(power(tensor.name)) + "}"
                           if power(tensor.name) > 1 else "")
                          for tensor in operator.tensors
                          if power(tensor.name) > -1])
    denominator = " ".join([display_tensor(structures.get(name(tensor.name), "T"),
                                           map(assigned_inds.get, tensor.indices)) +
                            ("^{" + display_number(-power(tensor.name)) + "}"
                             if power(tensor.name) < -1 else "")
                            for tensor in operator.tensors
                            if power(tensor.name) < 0])
    return "\\frac{{{}}}{{{}}}".format(numerator, denominator)

def display_number(number):
    if number == 1:
        return ""
    elif number == -1:
        return "-"
    elif number == 1j:
        return "i"
    elif number == -1j:
        return "-i"
    elif number.imag == 0:
        if int(number.real) == number.real:
            return str(int(number.real))
        else:
            return str(number.real)
    elif number.real == 0:
        return str(number.imag) + "i"
    else:
        return str(number).replace("j", "i")

class Writer(object):
    def __init__(self, op_sum, op_names, verbose=True):
        self.collection, self.rest = collect(op_sum, op_names, verbose)
        self.verbose = verbose

    def __str__(self):
        total = "Collected:\n{coll}\nRest:\n  {rest}"
        op_form = "  {name}:\n{coef}\n"
        coef_term_form = "    {} {}"
        return total.format(
            coll = "\n".join(op_form.format(
                name = str(op_name),
                coef = "\n".join(coef_term_form.format(num, op_coef)
                                  for num, op_coef in coef_lst))
                              for op_name, coef_lst in self.collection),
            rest = "\n".join(str(op) for op in self.rest))

    def latex_code(self, structures, op_reps, inds):
        out_str = "Collected operators:\n"
        for op_name, coef_lst in self.collection:
            out_str += "\\begin{align*}\n"
            out_str += op_reps.get(op_name, str(op_name)) + "= \n  "
            for i, (op_coef, num) in enumerate(coef_lst):
                out_str += "& " + display_number(num) + " "
                out_str += display_operator(op_coef, structures, inds)
                if i < len(coef_lst) - 1:
                    out_str += " +"
                    if i%4 == 0:
                        out_str += "\\\\"
                        out_str += "\n  "
            out_str += "\\end{align*}\n"
        if self.rest:
            out_str += "\nRest:\n"
            out_str += "\\begin{align*}\n"
            for i, op in enumerate(self.rest):
                out_str += display_operator(op) + " + "
                if i%5 == 0:
                    out_str += "\n"
            out_str += "\\end{align*}\n"
        return out_str

    def write_text_file(self, filename):
        with open(filename, "w") as f:
            f.write(str(self))

    def write_latex(self, filename, structures, op_reps, inds):
        out_str = self.latex_code(structures, op_reps, inds)
        with open(filename + ".tex", "w") as f:
            f.write(("\\documentclass{{article}}\n" +
                     "\\usepackage{{amsmath}}\n" +
                     "\\begin{{document}}\n{}" +
                     "\\end{{document}}").format(self.latex_code(structures,
                                                                 op_reps, inds)))

    def show_pdf(self, filename, pdf_viewer, structures, op_reps, inds):
        self.write_latex(filename, structures, op_reps, inds)
        call(["pdflatex", filename + ".tex"])
        call([pdf_viewer, filename + ".pdf"])


