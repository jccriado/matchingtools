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

class LookUpTable(object):
    def __init__(self, items=None):
        if items is None:
            items = []
        self.items = items

    @property
    def keys(self):
        return (key for key, _ in self.items)

    @property
    def values(self):
        return (value for _, value in self.items)
        
    def __contains__(self, key):
        return key in self.keys

    def lookup(self, key, default=None):
        for candidate_key, value in self.items:
            if candidate_key == key:
                return value
        if default is None:
            # TODO: improve?
            raise Exception("Key not found: {}".format(key))
        else:
            return default
    
    def update(self, key, value, binary_operator=None):
        for position, (old_key, old_value) in enumerate(self.items):
            if old_key == key:
                if binary_operator is None:
                    self.items[position] = (key, value)
                    break
                
                else:
                    self.items[position] = (
                        old_key,
                        binary_operator(old_value, value)
                    )
                    break
                
        else:
            self.items.append((key, value))
            
                    
