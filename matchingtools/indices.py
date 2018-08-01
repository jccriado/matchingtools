import uuid


class Index(object):
    def __init__(self, name):
        self.name = name
        self._id = uuid.uuid4()

    def __eq__(self, other):
        return self._id == other._id

    def __str__(self):
        return self.name

    __repr__ = __str__

    def __hash__(self):
        return hash(self._id)

    @staticmethod
    def make(*names):
        return [Index(name) for name in names]

    @staticmethod
    def assign_unique_str(indices):
        groups = {}
        for index in indices:
            groups.setdefault(index.name, set())
            groups[index.name].add(index)

        return {
            index: name + (str(pos) if len(corresponding_indices) != 1 else '')
            for name, corresponding_indices in groups.items()
            for pos, index in enumerate(corresponding_indices)
        }
            
            
