from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from PIL import Image
from pydantic import BaseModel, root_validator

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.labels.classification import ClassificationType
from encord_active.lib.labels.object import ObjectShape
from encord_active.lib.metrics.metric import Metric
from encord_active.lib.metrics.types import DataType, EmbeddingType, MetricType
from encord_active.lib.metrics.writer import CSVMetricWriter


class BaseObjectPrediction(BaseModel):
    class_probs: np.ndarray

    @root_validator
    def class_probs_should_be_one_dimensional(cls, values):  # pylint: disable=no-self-argument
        v = values.get("class_probs")
        if v.ndim != 1:
            raise ValueError("Class probabilities should be 1 dimensional (1xN)")
        if v.shape[0] < 2:
            raise ValueError(
                "There should be at least 2 probability values. If there is only one class in the ontology"
                ", probabilities should be [P(class), 1-P(class)]"
            )
        return values

    class Config:
        arbitrary_types_allowed = True


class BoundingBoxPrediction(BaseObjectPrediction):
    x: float
    y: float
    w: float
    h: float


@dataclass
class SegmentationPrediction(BaseObjectPrediction):
    points: np.ndarray


class BaseClassificationModel:
    @abstractmethod
    def predict_probabilities(self, image: Image) -> np.ndarray:
        pass

    def predict(self, image: Image) -> np.ndarray:
        return self.predict_probabilities(image)


class BaseObjectModel:
    @abstractmethod
    def predict_objects(self, image: Image) -> List[Union[BoundingBoxPrediction, SegmentationPrediction]]:
        """
        Calculate the model-predicted objects.

        Args:
            image: Input images.

        Returns:
            A list of list of predicted objects.
        """
        pass

    def predict(self, image: Image) -> List[Union[BoundingBoxPrediction, SegmentationPrediction]]:
        return self.predict_objects(image)


class SKLearnClassificationModel(BaseClassificationModel):
    def __init__(self, model):
        self._model = model

    def predict_probabilities(self, image: Image) -> np.ndarray:
        data = np.asarray(image).flatten() / 255
        return self._model.predict_proba(data)


class AcquisitionFunction(Metric):
    def __init__(
        self,
        title: str,
        short_description: str,
        long_description: str,
        metric_type: MetricType,
        data_type: DataType,
        model: Union[BaseClassificationModel, BaseObjectModel],
        annotation_type: list[Union[ObjectShape, ClassificationType]] = [],
        embedding_type: Optional[EmbeddingType] = None,
        doc_url: Optional[str] = None,
    ):
        """
        Abstract base class for the acquisition functions.

        Args:
            model (BaseClassificationModel or BaseObjectModel): Machine learning model wrapper used to score data
            samples.
        """
        self.doc_url = doc_url
        self._model = model
        super().__init__(
            title, short_description, long_description, metric_type, data_type, annotation_type, embedding_type
        )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        for _, image in iterator.iterate(desc=f"Running {self.metadata.title} acquisition function"):
            if image is None:
                continue
            predictions = self._model.predict(image)
            if predictions is None:
                continue
            score = self.score_predictions(predictions)
            writer.write(score)

    @abstractmethod
    def score_predictions(
        self, predictions: Union[np.ndarray, List[Union[BoundingBoxPrediction, SegmentationPrediction]]]
    ) -> float:
        """
        Scores model predictions according the acquisition function description.

        Args:
            predictions: An array of shape ``(N, K)`` of model-predicted class probabilities, ``P(label=k|x)``.
                Each row of this matrix corresponds to an example `x` and contains the model-predicted probabilities
                that `x` belongs to each possible class, for each of the K classes.

        Returns:
             score: Score of the model-predicted class probabilities.
        """
        pass
