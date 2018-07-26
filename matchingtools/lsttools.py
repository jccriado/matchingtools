import itertools


def concat(lst):
    return list(itertools.chain(*lst))


def enum_product(*iters):
    return itertools.product(*map(enumerate, iters))


def iterate(f, x, n):
    if n > 0:
        iterated_collection = iterate(f, x, n - 1)
        iterated_collection.append(f(iterated_collection[-1]))
        return iterated_collection
    else:
        return [x]
