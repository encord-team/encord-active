from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

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


@dataclass
class SegmentationPrediction:
    points: np.ndarray  # an array of shape Nx2, where columns refer to relative x and y coordinates, respectively
    confidence: float


class BaseObjectModel:
    @abstractmethod
    def predict_objects(self, image: Image) -> List[Union[BoundingBoxPrediction, SegmentationPrediction]]:
        """
        Calculate the model-predicted objects.

        Args:
            image: Input image.

        Returns:
            The list of predicted objects. If there is no prediction, an empty list is returned.
        """
        pass


class BaseObjectAcquisitionFunction(Metric):
    def __init__(
        self,
        title: str,
        short_description: str,
        long_description: str,
        metric_type: MetricType,
        data_type: DataType,
        model: BaseObjectModel,
        annotation_type: list[Union[ObjectShape, ClassificationType]] = [],
        embedding_type: Optional[EmbeddingType] = None,
        doc_url: Optional[str] = None,
    ):
        """
        Creates an instance of the acquisition function with a custom model to score data samples.

        Args:
            model (BaseObjectModel): Machine learning model used to score data samples.
        """
        self._model = model
        super().__init__(
            title, short_description, long_description, metric_type, data_type, annotation_type, embedding_type
        )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        for _, image in iterator.iterate(desc=f"Running {self.metadata.title} acquisition function"):
            if image is None:
                continue
            predicted_objects = self._model.predict_objects(image)
            if predicted_objects is None:
                continue
            score = self.score_object_predictions(predicted_objects)
            writer.write(score)

    @abstractmethod
    def score_object_predictions(
        self, predictions: List[Union[BoundingBoxPrediction, SegmentationPrediction]]
    ) -> float:
        """
        Scores model-predicted class probabilities according to the acquisition function description.

        Args:
            predictions: A list of predicted objects.

        Returns:
             score: Score given to image regarding the predictions.
        """
        pass


class AverageFrameScore(BaseObjectAcquisitionFunction):
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

    def score_object_predictions(
        self, predictions: List[Union[BoundingBoxPrediction, SegmentationPrediction]]
    ) -> float:
        if predictions:
            return sum([prediction.confidence for prediction in predictions]) / len(predictions)
        else:
            return 0.0
