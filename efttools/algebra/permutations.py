def splittings(lst):
    return [(i, elem, lst[:i] + lst[i+1:]) for i, elem in enumerate(lst)]

def tuple_permutations(tpl):
    if not tpl:
        return {(): 1}
    return {(elem,) + rest_perm: (-1)**i * rest_sign
            for i, elem, rest in splittings(tpl)
            for rest_perm, rest_sign in tuple_permutations(rest).items()}

def permutations(n):
    return tuple_permutations(tuple(range(n)))
