from typing import Set, Optional

from torch import FloatTensor, IntTensor, BoolTensor
from abc import ABCMeta, abstractmethod

from encord_active.analysis.base import BaseEvaluation


class PureImageEmbedding(BaseEvaluation, metaclass=ABCMeta):
    """
    Pure image based embedding, can optionally be calculated on a per-object basis as well as per-image by default.
    """

    def __init__(self, ident: str, dependencies: Set[str],
                 allow_object_embedding: bool = True, allow_queries: bool = False) -> None:
        super().__init__(ident, dependencies)
        self.allow_object_embedding = allow_object_embedding
        self.allow_queries = allow_queries

    @abstractmethod
    def evaluate_embedding(self, image: IntTensor, mask: Optional[BoolTensor]) -> FloatTensor:
        ...

    def evaluate_object_embedding(self, image: IntTensor, mask: BoolTensor, coordinates: FloatTensor) -> FloatTensor:
        _coordinates = coordinates  # Only used in override logic for special case shortcuts
        return self.evaluate_embedding(image, mask)


class PureObjectEmbedding(BaseEvaluation, metaclass=ABCMeta):
    def __init__(self, ident: str, dependencies: Set[str], allow_queries: bool = False) -> None:
        super().__init__(ident, dependencies)
        self.allow_queries = allow_queries

    """
    Pure object based embedding, only depends on object metadata
    """
    @abstractmethod
    def evaluate_embedding(self, mask: BoolTensor, coordinates: FloatTensor) -> FloatTensor:
        ...


class NearestImageEmbeddingQuery(BaseEvaluation):
    """
    Pseudo embedding, returns the embedding for the nearest image embedding (in the same category).
    Extra dependency tracking is automatically added whenever this is used.
    """

    def __init__(self, ident: str, source: str) -> None:
        super().__init__(ident=ident, dependencies={source})
        self.source = source
