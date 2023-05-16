from typing import Set

from torch import FloatTensor
from abc import ABC, abstractmethod

from encord_active.analysis.base import BaseAnalysis
from encord_active.analysis.context import ImageEvalContext
from encord_active.analysis.image import ImageContext


class ImageEmbedding(BaseAnalysis, ABC):
    def __init__(self, ident: str, dependencies: Set[str], long_name: str, short_desc: str, long_desc: str,
                 allow_nearby_query: bool) -> None:
        super().__init__(ident=ident, dependencies=dependencies)
        self.long_name = long_name
        self.short_desc = short_desc
        self.long_desc = long_desc
        self.allow_nearby_query = allow_nearby_query

    @abstractmethod
    def calc_embedding(self, context: ImageEvalContext, image: ImageContext) -> FloatTensor:
        ...


class ImageAndLabelEmbedding(ImageEmbedding, ABC):
    @abstractmethod
    def calc_object_embedding(self, context: ImageEvalContext, image: ImageContext) -> FloatTensor:
        ...
