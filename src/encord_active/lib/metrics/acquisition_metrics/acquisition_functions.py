from typing import List, Union

import numpy as np

from encord_active.lib.metrics.acquisition_metrics.common import (
    AcquisitionFunction,
    BaseClassificationModel,
    BaseObjectModel,
    BoundingBoxPrediction,
    SegmentationPrediction,
)
from encord_active.lib.metrics.types import AnnotationType, DataType, MetricType


class MeanObjectConfidence(AcquisitionFunction):
    def __init__(self, model: Union[BaseClassificationModel, BaseObjectModel]):
        super().__init__(
            title="Mean Object Confidence",
            short_description="Ranks images by average of its predicted objects' confidences",
            long_description="For each image, this acquisition function returns the average confidence of the "
            "predictions. If there is no prediction for the given image, it assigns a value of "
            "zero. This acquisition function is only valid for objects and makes sense when at least one ground truth "
            "object is expected for each image.",
            doc_url="",
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.NONE,
            model=model,
        )

    def score_predictions(
        self, predictions: Union[np.ndarray, List[Union[BoundingBoxPrediction, SegmentationPrediction]]]
    ) -> float:

        if isinstance(predictions, np.ndarray):
            raise Exception("This acquisition function only evaluates object predictions (bounding-box/segmentation)")
        if predictions:
            return sum([prediction.class_probs.max() for prediction in predictions]) / len(predictions)
        else:
            return 0.0


class Entropy(AcquisitionFunction):
    def __init__(self, model: Union[BaseClassificationModel, BaseObjectModel]):
        super().__init__(
            title="Entropy",
            short_description="Ranks images by their entropy.",
            long_description=(
                "Ranks images by their predictions' entropy. \n \n"
                "In information theory, the **entropy** of a random variable is the average level of “information”, "
                "“surprise”, or “uncertainty” inherent to the variable's possible outcomes. "
                "The higher the entropy, the more “uncertain” the variable outcome. \n \n"
                r"The mathematical formula of entropy is: $H(p) = -\sum_{i=1}^{n} p_i \log_{2}{p_i}$"
                " \n \nIt can be employed to define a heuristic that measures a model’s uncertainty about the classes "
                "in an image using the average of the entropies of the model-predicted class probabilities in the "
                "image. Like before, the higher the image's score, the more “confused” the model is. "
                "As a result, data samples with higher entropy score can be offered for annotation."
            ),
            doc_url="https://docs.encord.com/docs/active-model-quality-metrics#entropy",
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.NONE,
            model=model,
        )

    def score_predictions(
        self, predictions: Union[np.ndarray, List[Union[BoundingBoxPrediction, SegmentationPrediction]]]
    ) -> float:
        # silence divide by zero warning as the result will be correct (log2(0) is -inf, when multiplied by 0 gives 0)
        # raise exception if invalid (negative) values are found in the pred_proba array
        if isinstance(predictions, np.ndarray):
            predictions_transformed = predictions.reshape(1, -1)
        else:
            predictions_transformed = np.stack([prediction.class_probs for prediction in predictions])

        with np.errstate(divide="ignore", invalid="raise"):
            return (
                -np.multiply(predictions_transformed, np.nan_to_num(np.log2(predictions_transformed)))
                .sum(axis=1)
                .mean()
            )


class LeastConfidence(AcquisitionFunction):
    def __init__(self, model: Union[BaseClassificationModel, BaseObjectModel]):
        super().__init__(
            title="Least Confidence",
            short_description="Ranks images by their least confidence score.",
            long_description=(
                "Ranks images by their predictions' least confidence score. \n \n"
                "**Least confidence** (**LC**) score of a model's prediction is the difference between 1 "
                "(100% confidence) and its most confidently predicted class label. The higher the **LC** score, the "
                "more “uncertain” the prediction. \n \n"
                "The mathematical formula of the **LC** score of a model's prediction $x$ is: "
                r"$H(p) = 1 - \underset{y}{\max}(P(y|x))$"
                " \n \nIt can be employed to define a heuristic that measures a model’s uncertainty about the classes "
                "in an image using the average of the **LC** score of the model-predicted class probabilities in the "
                "image. Like before, the higher the image's score, the more “confused” the model is. "
                "As a result, data samples with higher **LC** score can be offered for annotation."
            ),
            doc_url="https://docs.encord.com/docs/active-model-quality-metrics#least-confidence",
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.NONE,
            model=model,
        )

    def score_predictions(
        self, predictions: Union[np.ndarray, List[Union[BoundingBoxPrediction, SegmentationPrediction]]]
    ) -> float:
        if isinstance(predictions, np.ndarray):
            predictions_transformed = predictions.reshape(1, -1)
        else:
            predictions_transformed = np.stack([prediction.class_probs for prediction in predictions])

        return (1 - predictions_transformed.max(axis=1)).mean()


class Margin(AcquisitionFunction):
    def __init__(self, model: Union[BaseClassificationModel, BaseObjectModel]):
        super().__init__(
            title="Margin",
            short_description="Ranks images by their margin score.",
            long_description=(
                "Ranks images by their predictions' margin score. \n \n"
                "**Margin** score of a model's prediction is the difference between the two classes with the highest "
                "probabilities. The lower the margin score, the more “uncertain” the prediction. \n \n"
                "It can be employed to define a heuristic that measures a model’s uncertainty about the classes "
                "in an image using the average of the margin score of the model-predicted class probabilities in the"
                " image. Like before, the lower the image's score, the more “confused” the model is. "
                "As a result, data samples with lower margin score can be offered for annotation."
            ),
            doc_url="https://docs.encord.com/docs/active-model-quality-metrics#margin",
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.NONE,
            model=model,
        )

    def score_predictions(
        self, predictions: Union[np.ndarray, List[Union[BoundingBoxPrediction, SegmentationPrediction]]]
    ) -> float:
        if isinstance(predictions, np.ndarray):
            predictions_transformed = predictions.reshape(1, -1)
        else:
            predictions_transformed = np.stack([prediction.class_probs for prediction in predictions])

        # move the second highest and highest class prediction values to the last two columns respectively
        preds = np.partition(predictions_transformed, -2)
        return (preds[:, -1] - preds[:, -2]).mean()


class Variance(AcquisitionFunction):
    def __init__(self, model: Union[BaseClassificationModel, BaseObjectModel]):
        super().__init__(
            title="Variance",
            short_description="Ranks images by their variance.",
            long_description=(
                "Ranks images by their predictions' variance. \n \n"
                "Variance is a measure of dispersion that takes into account the spread of all data points in a "
                "data set. The variance is the mean squared difference between each data point and the centre of the "
                "distribution measured by the mean. The lower the variance, the more “clustered” the data points. \n \n"
                "The mathematical formula of variance of a data set is: \n"
                r"$Var(X) = \frac{1}{n} \sum_{i=1}^{n}(x_i - \mu)^2, \text{where } \mu = \frac{1}{n} \sum_{i=1}^{n}x_i$"
                " \n \nIt can be employed to define a heuristic that measures a model’s uncertainty about the classes "
                "in an image using the average of the variance of the model-predicted class probabilities in the "
                "image. Like before, the lower the image's score, the more “confused” the model is. "
                "As a result, data samples with lower variance score can be offered for annotation."
            ),
            doc_url="https://docs.encord.com/docs/active-model-quality-metrics#variance",
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.NONE,
            model=model,
        )

    def score_predictions(
        self, predictions: Union[np.ndarray, List[Union[BoundingBoxPrediction, SegmentationPrediction]]]
    ) -> float:
        if isinstance(predictions, np.ndarray):
            predictions_transformed = predictions.reshape(1, -1)
        else:
            predictions_transformed = np.stack([prediction.class_probs for prediction in predictions])

        return predictions_transformed.var(axis=1).mean()
