import uuid


class Index(object):
    def __init__(self, name):
        self.name = name
        self._id = uuid.uuid4()

    def __eq__(self, other):
        return self._id == other._id

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self._id)

    @staticmethod
    def make(*names):
        return [Index(name) for name in names]
