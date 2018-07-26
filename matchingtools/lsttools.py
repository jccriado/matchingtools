import itertools


def concat(lst):
    return list(itertools.chain(*lst))


def enum_product(*iters):
    return itertools.product(*map(enumerate, iters))


def iterate(f, x, n):
    if n > 0:
        return f(iterate(f, x, n - 1))
    else:
        return x
