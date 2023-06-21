from abc import abstractmethod
from typing import Any, Optional, Union
from encord_active.lib.labels.object import ObjectShape
from encord_active.lib.metrics.metric import Metric
from encord_active.lib.metrics.types import (
    AnnotationType,
    DataType,
    EmbeddingType,
    MetricType,
)
from PIL import Image
from encord_active.lib.common.iterator import Iterator
from encord_active.lib.metrics.writer import CSVMetricWriter
import numpy as np

class BaseBoundingBoxModelWrapper:
    def __init__(self, model):
        self._model = model

    @classmethod
    @abstractmethod
    def prepare_data(cls, images: list[Image]) -> list[Any]:
        """
        Reads and prepares data samples from local storage to feed the model with it.

        Args:
            images (list[Image]): Images to use as data samples.

        Returns:
            Data samples prepared to be used as input of `self.predict_probabilities()` method.
        """
        pass

    def predict_boundingboxes(self, data) -> List[BoundingBox]:
        """
        Calculate the model-predicted bounding boxes.

        Args:
            data: Input data sample.

        Returns:
            An array of shape ``(N, K)`` of model-predicted class probabilities, ``P(label=k|x)``.
            Each row of this matrix corresponds to an example `x` and contains the model-predicted probabilities that
            `x` belongs to each possible class, for each of the K classes.
            In the case the model can't extract any example `x` from the data sample, the method returns ``None``.
        """
        pred_proba = self._predict_proba(data)
        if pred_proba is not None and pred_proba.min() < 0:
            raise ValueError("Model-predicted class probabilities cannot be less than zero.")
        return pred_proba

    @abstractmethod
    def _predict_proba(self, X) -> Optional[np.ndarray]:
        """
        Probability estimates.

        Note that in the multilabel case, each sample can have any number of labels.
        This returns the marginal probability that the given sample has the label in question.

        Args:
            X ({array-like} of shape (n_samples, n_features)): Input data.

        Returns:
            An array of shape (n_samples, n_classes). Probability of the sample for each class in the model.
            In the case the model fails, the method returns ``None``.
        """
        pass

class AcquisitionFunction(Metric):
    def __init__(
        self,
        title: str,
        short_description: str,
        long_description: str,
        metric_type: MetricType,
        data_type: DataType,
        model: BaseModelWrapper,
        annotation_type: list[Union[ObjectShape, ClassificationType]] = [],
        embedding_type: Optional[EmbeddingType] = None,
        doc_url: Optional[str] = None,
    ):
        """
        Creates an instance of the acquisition function with a custom model to score data samples.

        Args:
            model (BaseModelWrapper): Machine learning model used to score data samples.
        """
        self._model = model
        super().__init__(
            title, short_description, long_description, metric_type, data_type, annotation_type, embedding_type
        )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        for _, image in iterator.iterate(desc=f"Running {self.metadata.title} acquisition function"):
            if image is None:
                continue
            prepared_data = self._model.prepare_data([image])
            if not prepared_data:
                continue
            pred_proba = self._model.predict_probabilities(prepared_data)
            if pred_proba is None:
                continue
            score = self.score_predicted_class_probabilities(pred_proba)
            writer.write(score)

    @abstractmethod
    def score_predicted_class_probabilities(self, pred_proba: np.ndarray) -> float:
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


class AverageFrameScore():