from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import numpy as np
from PIL import Image

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.labels.classification import ClassificationType
from encord_active.lib.labels.object import ObjectShape
from encord_active.lib.metrics.metric import Metric
from encord_active.lib.metrics.types import (
    AnnotationType,
    DataType,
    EmbeddingType,
    MetricType,
)
from encord_active.lib.metrics.writer import CSVMetricWriter


@dataclass
class BoundingBoxPrediction:
    x: float
    y: float
    w: float
    h: float
    confidence: float


class BaseBoundingBoxModelWrapper:
    @abstractmethod
    def predict_boundingboxes(self, image: Image) -> List[BoundingBoxPrediction]:
        """
        Calculate the model-predicted bounding boxes.

        Args:
            data: Input data sample.

        Returns:
            An array of shape ``(N, K)`` of model-predicted bounding boxes.
            Each row of this matrix corresponds to an example `x` and contains the model-predicted probabilities that
            `x` belongs to each possible class, for each of the K classes.
            In the case the model can't extract any example `x` from the data sample, the method returns ``None``.
        """
        pass


class BoundingBoxAcquisitionFunction(Metric):
    def __init__(
        self,
        title: str,
        short_description: str,
        long_description: str,
        metric_type: MetricType,
        data_type: DataType,
        model: BaseBoundingBoxModelWrapper,
        annotation_type: list[Union[ObjectShape, ClassificationType]] = [],
        embedding_type: Optional[EmbeddingType] = None,
        doc_url: Optional[str] = None,
    ):
        """
        Creates an instance of the acquisition function with a custom model to score data samples.

        Args:
            model (BaseBoundingBoxModelWrapper): Machine learning model used to score data samples.
        """
        self._model = model
        super().__init__(
            title, short_description, long_description, metric_type, data_type, annotation_type, embedding_type
        )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        for _, image in iterator.iterate(desc=f"Running {self.metadata.title} acquisition function"):
            if image is None:
                continue
            predicted_bboxes = self._model.predict_boundingboxes(image)
            if predicted_bboxes is None:
                continue
            score = self.score_bbox_predictions(predicted_bboxes)
            writer.write(score)

    @abstractmethod
    def score_bbox_predictions(self, predictions: List[BoundingBoxPrediction]) -> float:
        """
        Scores model-predicted class probabilities according the acquisition function description.

        Args:
            pred_proba: An array of shape ``(N, K)`` of model-predicted class probabilities, ``P(label=k|x)``.
                Each row of this matrix corresponds to an example `x` and contains the model-predicted probabilities
                that `x` belongs to each possible class, for each of the K classes.

        Returns:
             score: Score of the model-predicted class probabilities.
        """
        pass


class AverageFrameScore(BoundingBoxAcquisitionFunction):
    def __init__(self, model):
        super().__init__(
            title="Average Frame Score",
            short_description="Ranks images by average of the prediction confidences",
            long_description="For each image, this acquisition function returns the average confidence of the "
            "predictions. If there is no prediction for the given image, it assigns a value of "
            "zero. This acquisition function makes sense when at least one ground truth prediction is expected "
            "for each image.",
            doc_url="",
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.NONE,
            model=model,
        )

    def score_bbox_predictions(self, predictions: List[BoundingBoxPrediction]) -> float:
        if predictions:
            sum([prediction.confidence for prediction in predictions]) / len(predictions)
        else:
            return 0.0
