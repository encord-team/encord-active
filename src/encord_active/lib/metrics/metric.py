from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from hashlib import md5
from typing import List, Optional, Union

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.metrics.writer import CSVMetricWriter


class MetricType(Enum):
    SEMANTIC = "semantic"
    GEOMETRIC = "geometric"
    HEURISTIC = "heuristic"


class DataType(Enum):
    IMAGE = "image"
    SEQUENCE = "sequence"


class EmbeddingType(Enum):
    CLASSIFICATION = "classification"
    OBJECT = "object"
    HU_MOMENTS = "hu_moments"
    IMAGE = "image"


# copy from encord but as a string enum
class ClassificationType(str, Enum):
    RADIO = "radio"
    TEXT = "text"
    CHECKLIST = "checklist"


# copy from encord but as a string enum
class ObjectShape(str, Enum):
    POLYGON = "polygon"
    POLYLINE = "polyline"
    BOUNDING_BOX = "bounding_box"
    KEY_POINT = "point"
    SKELETON = "skeleton"
    ROTATABLE_BOUNDING_BOX = "rotatable_bounding_box"


AnnotationTypeUnion = Union[ObjectShape, ClassificationType]


@dataclass
class StatsMetadata:
    threshold: float
    max_value: float
    mean_value: float
    min_value: float
    num_rows: int


@dataclass
class MetricMetadata:
    title: str
    short_description: str
    long_description: str
    metric_type: MetricType
    data_type: DataType
    needs_images: bool
    annotation_type: Optional[List[AnnotationTypeUnion]]
    embedding_type: Optional[EmbeddingType] = None

    def get_unique_name(self):
        name_hash = md5((self.title + self.short_description + self.long_description).encode()).hexdigest()
        return f"{name_hash[:8]}_{self.title.lower().replace(' ', '_')}"


class Metric(ABC):
    def __init__(
        self,
        title: str,
        short_description: str,
        long_description: str,
        metric_type: MetricType,
        data_type: DataType,
        annotation_type: List[Union[ObjectShape, ClassificationType]],
        embedding_type: Optional[EmbeddingType] = None,
        needs_images: bool = False,
    ):
        self.metadata = MetricMetadata(
            title=title,
            short_description=short_description,
            long_description=long_description,
            metric_type=metric_type,
            data_type=data_type,
            annotation_type=annotation_type,
            embedding_type=embedding_type,
            needs_images=False if not needs_images and metric_type == MetricType.GEOMETRIC else True,
        )

    @abstractmethod
    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        """
        This is where you should perform your data indexing.

        :param iterator: The iterator with which you can iterate through the dataset as
            many times you like. The iterator scans through one label rows at a time,
            continuously indexing each frame of a video.

            Use::

                for data_unit, img_pth in iterator.iterate(desc="Progress bar description"):
                    pass


        :param writer:  The writer where you should store your scores.
            Use::

                writer.write(score, objects)


        :return:
        """
        pass
