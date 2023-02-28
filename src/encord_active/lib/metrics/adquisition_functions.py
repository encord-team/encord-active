from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.metrics.metric import (
    AnnotationType,
    DataType,
    Metric,
    MetricType,
)
from encord_active.lib.metrics.writer import CSVMetricWriter


class AcquisitionFunction(Metric):
    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        for _, img_pth in iterator.iterate(desc=f"Running {self.metadata.title} acquisition function"):
            img = self.read_image(img_pth)
            preds = self.get_predictions(img)
            score = self.score_acquisition_function(preds)
            writer.write(score)

    @abstractmethod
    def get_predictions(self, image) -> np.ndarray:
        pass

    @abstractmethod
    def read_image(self, image_path: Path):
        pass

    @abstractmethod
    def score_predictions(self, predictions: np.ndarray) -> float:
        pass


class Entropy(AcquisitionFunction):
    def __init__(self):
        super().__init__(
            title="Entropy",
            short_description="Ranks images by their entropy.",
            long_description=r"""Ranks images by their entropy.

The mathematical formula for entropy calculation is:
$$H(p) = -\sum{_i=1}{N} p_i \log{_2}{p_i}$$""",
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.NONE,
        )

    def score_predictions(self, predictions: np.ndarray) -> float:
        with np.errstate(divide="ignore"):
            return -np.multiply(predictions, np.log2(predictions)).sum(axis=1).mean()


class Margin(AcquisitionFunction):
    def __init__(self):
        super().__init__(
            title="Margin",
            short_description="",
            long_description=r"""""",
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.NONE,
        )


class Variance(AcquisitionFunction):
    def __init__(self):
        super().__init__(
            title="Variance",
            short_description="",
            long_description=r"""""",
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.NONE,
        )
