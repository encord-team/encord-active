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
            long_description=(
                "Ranks images by their entropy. \n \n"
                'In information theory, the entropy of a random variable is the average level of "information", '
                '"surprise", or "uncertainty" inherent to the variable\'s possible outcomes. '
                'The higher the entropy, the more "uncertain" the variable outcome. \n \n'
                r"The mathematical formula of entropy is: $H(p) = -\sum_{i=1}^{N} p_i \log_{2}{p_i}$"
                " \n \nIt can be used to define a heuristic that measures a model’s uncertainty about the classes in "
                "an image using the average of the entropies of the classes. As before, the higher the entropy, the "
                'more "confused" the model is. As a result, data samples with higher entropy '
                "should be offered for annotation."
            ),
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.NONE,
        )

    def score_predictions(self, predictions: np.ndarray) -> float:
        with np.errstate(divide="ignore"):
            return -np.multiply(predictions, np.log2(predictions)).sum(axis=1).mean()


class LeastConfidence(AcquisitionFunction):
    def __init__(self):
        super().__init__(
            title="Least Confidence",
            short_description="Ranks images by their least confidence score.",
            long_description=(
                "Ranks images by their least confidence score. \n \n"
                "**Least confidence** (**LC**) score of a model's prediction is the difference between 1 "
                "(100% confidence) and its most confidently predicted class label. The higher the **LC** score, the "
                'more "uncertain" the model\'s prediction. \n \n'
                "The mathematical formula of the **LC** score of a model's prediction $x$ is: "
                r"$H(p) = 1 - \underset{y}{\max}(P(y|x))$"
                " \n \nIt can be used to define a heuristic that measures a model’s uncertainty about the classes in "
                "an image using the average of the **LC** score of the predicted classes' instances. "
                "Like before, the "
                'higher the image **LC** score, the more "confused" the model is. As a result, data samples with '
                "higher **LC** score should be offered for annotation."
            ),
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.NONE,
        )

    def score_predictions(self, predictions: np.ndarray) -> float:
        with np.errstate(divide="ignore"):
            return (1 - predictions.max(axis=1)).mean()


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
