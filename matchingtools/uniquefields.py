import uuid

from matchingtools.core import ComplexField
from matchingtools.utils import merge_dicts


class _UniqueField(ComplexField):
    def __init__(self, _id, **kwargs):
        super().__init__(**kwargs)
        self._id = _id

    @staticmethod
    def from_operator(operator):
        _id = uuid.uuid4()

        return _UniqueField(
            _id,
            name="_UniqueField.from_operator({})".format(operator),
            indices=operator.free_indices,
            derivatives_indices=[],
            statistics=operator.statistics,
            dimension=operator.dimension
        )

    @staticmethod
    def from_tensor(tensor, _id=None):
        if _id is None:
            _id = uuid.uuid4()

        return _UniqueField(
            _id,
            name=tensor.name,
            indices=tensor.indices,
            derivatives_indices=tensor.derivatives_indices,
            statistics=tensor.statistics,
            dimension=tensor._tensor_dimension,
            is_conjugated=tensor.is_conjugated
        )

    def clone(self):
        return _UniqueField(
            self._id,
            name=self.name,
            indices=self.indices,
            derivatives_indices=self.derivatives_indices,
            statistics=self.statistics,
            dimension=self._tensor_dimension,
            is_conjugated=self.is_conjugated
        )

    def _match_attributes(self):
        return merge_dicts(super()._match_attributes(), {
            '_id': self._id
        })
