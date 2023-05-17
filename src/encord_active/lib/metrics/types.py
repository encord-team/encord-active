from enum import Enum
from typing import Union

from encord_active.lib.labels.classification import ClassificationType
from encord_active.lib.labels.object import ObjectShape

AnnotationTypeUnion = Union[ObjectShape, ClassificationType]


class AnnotationType:
    NONE: list[AnnotationTypeUnion] = []
    OBJECT = ObjectShape
    CLASSIFICATION = ClassificationType
    ALL = [*OBJECT, *CLASSIFICATION]


class MetricType(str, Enum):
    SEMANTIC = "semantic"
    GEOMETRIC = "geometric"
    HEURISTIC = "heuristic"


class DataType(str, Enum):
    IMAGE = "image"
    SEQUENCE = "sequence"


class EmbeddingType(str, Enum):
    CLASSIFICATION = "classification"
    OBJECT = "object"
    HU_MOMENTS = "hu_moments"
    IMAGE = "image"
