from efttools.algebra import (
    collect_numbers_and_symbols, collect_by_tensors)

def display_tensor(structure, indices):
    print structure, indices, structure.format(*map(str, indices))
    return structure.format(*map(str, indices))

def power(name):
    if name[0] == "{" and name[-1] == "}":
        return float(name[name.index("^")+1:-1])
    return 1

def name(name):
    if name[0] == "{" and name[-1] == "}":
        print name, name[1:name.index("^")]
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
                          ("^{" + display_number(power(tensor.name)) + "}" if power(tensor.name) > 1 else "")
                          for tensor in operator.tensors
                          if power(tensor.name) > -1])
    denominator = " ".join([display_tensor(structures.get(name(tensor.name), "T"),
                                           map(assigned_inds.get, tensor.indices)) +
                            ("^{" + display_number(-power(tensor.name)) + "}" if power(tensor.name) < -1 else "")
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


def display(op_sum, structures, op_reps, inds):
    op_sum = collect_numbers_and_symbols(op_sum)
    collection, rest = collect_by_tensors(op_sum, op_reps.keys())
    out_str = ""
    for op_name, coef_lst in collection:
        out_str += "\\begin{align*}\n"
        out_str += op_reps.get(op_name, str(op_name)) + "= \n"
        for i, (op_coef, num) in enumerate(coef_lst):
            out_str += "& " + display_number(num) + " "
            out_str += display_operator(op_coef, structures, inds)
            if i < len(coef_lst) - 1:
                out_str += " +"
                if i%4 == 0:
                    out_str += "\\\\"
            out_str += "\n"
        out_str += "\\end{align*}\n"
    return out_str

def write_latex(filename, op_sum, structures, op_reps, inds):
    out_str = display(op_sum, structures, op_reps, inds)
    f = open(filename, "w")
    f.write(("\\documentclass{{article}}\n\\usepackage{{amsmath}}\n" +
             "\\begin{{document}}\n{}\n\\end{{document}}").format(out_str))
