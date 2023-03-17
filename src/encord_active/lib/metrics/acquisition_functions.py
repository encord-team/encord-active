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


class AcquisitionFunction(Metric):
    def __init__(
        self,
        title: str,
        short_description: str,
        long_description: str,
        metric_type: MetricType,
        data_type: DataType,
        annotation_type: list[Union[ObjectShape, ClassificationType]] = [],
        embedding_type: Optional[EmbeddingType] = None,
        model=None,
    ):
        self._model = model
        super().__init__(
            title, short_description, long_description, metric_type, data_type, annotation_type, embedding_type
        )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        for _, img_pth in iterator.iterate(desc=f"Running {self.metadata.title} acquisition function"):
            if img_pth is None:
                continue
            img = self.read_image(img_pth)
            if img is None:
                continue
            pred_probs = self.get_predicted_class_probabilities(img)
            if pred_probs is None:
                continue
            score = self.score_predicted_class_probabilities(pred_probs)
            writer.write(score)

    def get_predicted_class_probabilities(self, image) -> Optional[np.ndarray]:
        """
        Calculate the model-predicted class probabilities of the examples found by the model in the image.

        :param image: Input image.
            *Requirement*: It must allow conversion to numpy ndarray via ``np.asarray(image)``.
        :return: An array of shape ``(N, K)`` of model-predicted class probabilities, ``P(label=k|x)``.
            Each row of this matrix corresponds to an example `x` and contains the model-predicted probabilities that
            `x` belongs to each possible class, for each of the K classes.
            In the case the model can't extract any example `x` from the image, the method will return ``None``.
        """
        image_array = np.asarray(image).flatten()
        pred_proba = self._model.predict_proba([image_array])
        return None if len(pred_proba) == 0 else pred_proba

    def read_image(self, image_path: Path) -> Optional[Any]:
        return Image.open(image_path)

    @abstractmethod
    def score_predicted_class_probabilities(self, predictions: np.ndarray) -> float:
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

    def score_predicted_class_probabilities(self, predictions: np.ndarray) -> float:
        # silence divide by zero warning as the result will be correct (log2(0) is -inf, when multiplied by 0 gives 0)
        # raise exception if invalid (negative) values are found in the predictions
        with np.errstate(divide="ignore", invalid="raise"):
            return -np.multiply(predictions, np.nan_to_num(np.log2(predictions))).sum(axis=1).mean()


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

    def score_predicted_class_probabilities(self, predictions: np.ndarray) -> float:
        return (1 - predictions.max(axis=1)).mean()


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

    def score_predicted_class_probabilities(self, predictions: np.ndarray) -> float:
        # move the second highest and highest class prediction values to the last two columns respectively
        preds = np.partition(predictions, -2)
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

    def score_predicted_class_probabilities(self, predictions: np.ndarray) -> float:
        return predictions.var(axis=1).mean()
