from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL import Image

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.labels.classification import ClassificationType
from encord_active.lib.labels.object import ObjectShape
from encord_active.lib.metrics.metric import (
    AnnotationType,
    DataType,
    EmbeddingType,
    Metric,
    MetricType,
)
from encord_active.lib.metrics.writer import CSVMetricWriter


class BaseModelWrapper:
    def __init__(self, model):
        self._model = model

    @classmethod
    @abstractmethod
    def prepare_data(cls, data_path: Path) -> Optional[Any]:
        """
        Reads and prepares a data sample from local storage to feed the model with it.

        Args:
            data_path (Path): Path to the data sample.

        Returns:
            Data sample prepared to be used as input of `self.predict_probabilities()` method.
        """
        pass

    def predict_probabilities(self, data) -> Optional[np.ndarray]:
        """
        Calculate the model-predicted class probabilities of the examples in the data sample found by the model.

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


class SKLearnModelWrapper(BaseModelWrapper):
    @classmethod
    def prepare_data(cls, data_paths: list[Path]) -> Optional[Any]:
        return [np.asarray(Image.open(data_path)).flatten() / 255 for data_path in data_paths]

    def _predict_proba(self, X) -> Optional[np.ndarray]:
        return self._model.predict_proba(X)


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
        for _, img_pth in iterator.iterate(desc=f"Running {self.metadata.title} acquisition function"):
            if img_pth is None:
                continue
            prepared_data = self._model.prepare_data(img_pth)
            if prepared_data is None:
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


class Entropy(AcquisitionFunction):
    def __init__(self, model):
        super().__init__(
            title="Entropy",
            short_description="Ranks images by their entropy.",
            long_description=(
                "Ranks images by their entropy. \n \n"
                "In information theory, the **entropy** of a random variable is the average level of “information”, "
                "“surprise”, or “uncertainty” inherent to the variable's possible outcomes. "
                "The higher the entropy, the more “uncertain” the variable outcome. \n \n"
                r"The mathematical formula of entropy is: $H(p) = -\sum_{i=1}^{n} p_i \log_{2}{p_i}$"
                " \n \nIt can be employed to define a heuristic that measures a model’s uncertainty about the classes "
                "in an image using the average of the entropies of the model-predicted class probabilities in the "
                "image. Like before, the higher the image's score, the more “confused” the model is. "
                "As a result, data samples with higher entropy score should be offered for annotation."
            ),
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.NONE,
            model=model,
        )

    def score_predicted_class_probabilities(self, pred_proba: np.ndarray) -> float:
        # silence divide by zero warning as the result will be correct (log2(0) is -inf, when multiplied by 0 gives 0)
        # raise exception if invalid (negative) values are found in the pred_proba array
        with np.errstate(divide="ignore", invalid="raise"):
            return -np.multiply(pred_proba, np.nan_to_num(np.log2(pred_proba))).sum(axis=1).mean()


class LeastConfidence(AcquisitionFunction):
    def __init__(self, model):
        super().__init__(
            title="Least Confidence",
            short_description="Ranks images by their least confidence score.",
            long_description=(
                "Ranks images by their least confidence score. \n \n"
                "**Least confidence** (**LC**) score of a model's prediction is the difference between 1 "
                "(100% confidence) and its most confidently predicted class label. The higher the **LC** score, the "
                "more “uncertain” the prediction. \n \n"
                "The mathematical formula of the **LC** score of a model's prediction $x$ is: "
                r"$H(p) = 1 - \underset{y}{\max}(P(y|x))$"
                " \n \nIt can be employed to define a heuristic that measures a model’s uncertainty about the classes "
                "in an image using the average of the **LC** score of the model-predicted class probabilities in the "
                "image. Like before, the higher the image's score, the more “confused” the model is. "
                "As a result, data samples with higher **LC** score should be offered for annotation."
            ),
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.NONE,
            model=model,
        )

    def score_predicted_class_probabilities(self, pred_proba: np.ndarray) -> float:
        return (1 - pred_proba.max(axis=1)).mean()


class Margin(AcquisitionFunction):
    def __init__(self, model):
        super().__init__(
            title="Margin",
            short_description="Ranks images by their margin score.",
            long_description=(
                "Ranks images by their margin score. \n \n"
                "**Margin** score of a model's prediction is the difference between the two classes with the highest "
                "probabilities. The lower the margin score, the more “uncertain” the prediction. \n \n"
                "It can be employed to define a heuristic that measures a model’s uncertainty about the classes "
                "in an image using the average of the margin score of the model-predicted class probabilities in the"
                " image. Like before, the lower the image's score, the more “confused” the model is. "
                "As a result, data samples with lower margin score should be offered for annotation."
            ),
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.NONE,
            model=model,
        )

    def score_predicted_class_probabilities(self, pred_proba: np.ndarray) -> float:
        # move the second highest and highest class prediction values to the last two columns respectively
        preds = np.partition(pred_proba, -2)
        return (preds[:, -1] - preds[:, -2]).mean()


class Variance(AcquisitionFunction):
    def __init__(self, model):
        super().__init__(
            title="Variance",
            short_description="Ranks images by their variance.",
            long_description=(
                "Ranks images by their variance. \n \n"
                "Variance is a measure of dispersion that takes into account the spread of all data points in a "
                "data set. The variance is the mean squared difference between each data point and the centre of the "
                "distribution measured by the mean. The lower the variance, the more “clustered” the data points. \n \n"
                "The mathematical formula of variance of a data set is: \n"
                r"$Var(X) = \frac{1}{n} \sum_{i=1}^{n}(x_i - \mu)^2, \text{where } \mu = \frac{1}{n} \sum_{i=1}^{n}x_i$"
                " \n \nIt can be employed to define a heuristic that measures a model’s uncertainty about the classes "
                "in an image using the average of the variance of the model-predicted class probabilities in the "
                "image. Like before, the lower the image's score, the more “confused” the model is. "
                "As a result, data samples with lower variance score should be offered for annotation."
            ),
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.NONE,
            model=model,
        )

    def score_predicted_class_probabilities(self, pred_proba: np.ndarray) -> float:
        return pred_proba.var(axis=1).mean()
