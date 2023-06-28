from abc import ABCMeta, abstractmethod

from torch import BoolTensor, FloatTensor, IntTensor

from encord_active.analysis.base import BaseEvaluation
from encord_active.analysis.types import (
    EmbeddingTensor,
    ImageTensor,
    MaskTensor,
    PointTensor,
)


class PureImageEmbedding(BaseEvaluation, metaclass=ABCMeta):
    """
    Pure image based embedding, can optionally be calculated on a per-object basis as well as per-image by default.
    """

    def __init__(
        self, ident: str, dependencies: set[str], allow_object_embedding: bool = True, allow_queries: bool = False
    ) -> None:
        super().__init__(ident, dependencies)
        self.allow_object_embedding = allow_object_embedding
        self.allow_queries = allow_queries

    @abstractmethod
    def evaluate_embedding(self, image: ImageTensor, mask: MaskTensor | None) -> EmbeddingTensor:
        ...

    def evaluate_object_embedding(
        self, image: ImageTensor, mask: MaskTensor, coordinates: PointTensor
    ) -> EmbeddingTensor:
        _coordinates = coordinates  # Only used in override logic for special case shortcuts
        return self.evaluate_embedding(image, mask)


class PureObjectEmbedding(BaseEvaluation, metaclass=ABCMeta):
    def __init__(self, ident: str, dependencies: set[str], allow_queries: bool = False) -> None:
        super().__init__(ident, dependencies)
        self.allow_queries = allow_queries

    """
    Pure object based embedding, only depends on object metadata
    """

    @abstractmethod
    def evaluate_embedding(self, mask: MaskTensor, coordinates: PointTensor) -> EmbeddingTensor:
        ...


class NearestImageEmbeddingQuery(BaseEvaluation):
    """
    Pseudo embedding, returns the embedding for the nearest image embedding (in the same category).
    Extra dependency tracking is automatically added whenever this is used.
    """

    def __init__(self, ident: str, source: str) -> None:
        super().__init__(ident=ident, dependencies={source})
        self.source = source
