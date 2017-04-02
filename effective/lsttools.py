import itertools

def concat(lst):
    return list(itertools.chain(*lst))

def enum_product(*iters):
    return itertools.product(*map(enumerate, iters))
