from abc import ABC, abstractmethod
from typing import List
from torch import FloatTensor


class ImageEvalContext(ABC):
    @abstractmethod
    def lookup_image_embedding(self, ident: str) -> FloatTensor:
        """Lookup a calculated embedding for the current image"""
        ...

    @abstractmethod
    def lookup_nearby_image_embeddings(self, ident: str, limit: int) -> List[(str, FloatTensor)]:
        """Lookup {limit} nearest embeddings and the associated image embeddings"""
        ...

    @abstractmethod
    def lookup_image_metric_value(self, ident: str) -> float:
        """Lookup the calculated result of an image metric"""
        ...


class LabelEvalContext(ImageEvalContext):
    @abstractmethod
    def lookup_label_metric_value(self, ident: str) -> float:
        """Lookup the calculated result of a label metric"""
        ...

    @abstractmethod
    def lookup_label_object_embedding(self, ident: str) -> FloatTensor:
        """Lookup the label embedding"""
        ...
