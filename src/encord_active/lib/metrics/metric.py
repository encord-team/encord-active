from abc import ABC, abstractmethod
from hashlib import md5
from typing import List, Optional

import numpy as np
from pydantic import BaseModel

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.writer import StatisticsObserver

# TODO: delete AnnotationType import on major version bump 👇
from encord_active.lib.metrics.types import (  # pylint: disable=unused-import
    AnnotationType,
    AnnotationTypeUnion,
    DataType,
    EmbeddingType,
    MetricType,
)
from encord_active.lib.metrics.writer import CSVMetricWriter


class StatsMetadata(BaseModel):
    min_value: float = 0.0
    max_value: float = 0.0
    mean_value: float = 0.0
    num_rows: int = 0

    @classmethod
    def from_stats_observer(cls, stats_observer: StatisticsObserver):
        return cls(
            min_value=stats_observer.min_value,
            max_value=stats_observer.max_value,
            mean_value=stats_observer.mean_value,
            num_rows=stats_observer.num_rows,
        )


class MetricMetadata(BaseModel):
    title: str
    short_description: str
    long_description: str
    metric_type: MetricType
    data_type: DataType
    annotation_type: List[AnnotationTypeUnion]
    embedding_type: Optional[EmbeddingType] = None
    doc_url: Optional[str] = None
    stats: StatsMetadata

    def get_unique_name(self):
        name_hash = md5((self.title + self.short_description + self.long_description).encode()).hexdigest()
        return f"{name_hash[:8]}_{self.title.lower().replace(' ', '_')}"


class SimpleMetric(ABC):
    def __init__(
        self,
        title: str,
        short_description: str,
        long_description: str,
        metric_type: MetricType,
        data_type: DataType,
        annotation_type: List[AnnotationTypeUnion],
        embedding_type: Optional[EmbeddingType] = None,
        doc_url: Optional[str] = None,
    ):
        self.metadata = MetricMetadata(
            title=title,
            short_description=short_description,
            long_description=long_description,
            metric_type=metric_type,
            data_type=data_type,
            annotation_type=annotation_type,
            embedding_type=embedding_type,
            doc_url=doc_url,
            stats=StatsMetadata(),
        )

    @abstractmethod
    def execute(self, image: np.ndarray):
        pass


class Metric(ABC):
    def __init__(
        self,
        title: str,
        short_description: str,
        long_description: str,
        metric_type: MetricType,
        data_type: DataType,
        annotation_type: List[AnnotationTypeUnion] = [],
        embedding_type: Optional[EmbeddingType] = None,
        doc_url: Optional[str] = None,
    ):
        self.metadata = MetricMetadata(
            title=title,
            short_description=short_description,
            long_description=long_description,
            metric_type=metric_type,
            data_type=data_type,
            annotation_type=annotation_type,
            embedding_type=embedding_type,
            doc_url=doc_url,
            stats=StatsMetadata(),
        )

    @abstractmethod
    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        """
        This is where you should perform your data indexing.

        :param iterator: The iterator with which you can iterate through the dataset as
            many times you like. The iterator scans through one label rows at a time,
            continuously indexing each frame of a video.

            Use::

                for data_unit, image in iterator.iterate(desc="Progress bar description"):
                    pass


        :param writer:  The writer where you should store your scores.
            Use::

                writer.write(score, objects)


        :return:
        """
        pass
