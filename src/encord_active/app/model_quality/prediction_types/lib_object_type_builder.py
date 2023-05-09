from enum import Enum
from functools import lru_cache
from typing import Annotated, Optional, Union

from pydantic import BaseModel, Field

import encord_active.lib.model_predictions.reader as reader
from encord_active.lib.model_predictions.filters import (
    filter_labels_for_frames_wo_predictions,
    prediction_and_label_filtering,
)
from encord_active.lib.model_predictions.map_mar import compute_mAP_and_mAR
from encord_active.lib.model_predictions.reader import (
    PredictionMatchSchema,
    get_class_idx,
)
from encord_active.lib.model_predictions.writer import MainPredictionType
from encord_active.lib.project.project_file_structure import ProjectFileStructure


class ClassificationOutcomeType(str, Enum):
    CORRECT_CLASSIFICATIONS = "Correct Classifications"
    MISCLASSIFICATIONS = "Misclassifications"


class ObjectDetectionOutcomeType(str, Enum):
    TRUE_POSITIVES = "True Positive"
    FALSE_POSITIVES = "False Positive"
    FALSE_NEGATIVES = "False Negative"


class PredictionsFilters(BaseModel):
    outcome: Optional[Union[ClassificationOutcomeType, ObjectDetectionOutcomeType]] = None
    iou_threshold: Annotated[float, Field(ge=0, le=1)] = 0.5
    ignore_frames_without_predictions: bool = False

    class Config:
        frozen = True


@lru_cache
def read_prediction_files(project_file_structure: ProjectFileStructure, prediction_type: MainPredictionType):
    metrics_dir = project_file_structure.metrics
    predictions_dir = project_file_structure.predictions / prediction_type.value

    predictions_metric_datas = reader.get_prediction_metric_data(predictions_dir, metrics_dir)
    label_metric_datas = reader.get_label_metric_data(metrics_dir)

    model_predictions = reader.get_model_predictions(predictions_dir, predictions_metric_datas, prediction_type)
    labels = reader.get_labels(predictions_dir, label_metric_datas, prediction_type)

    return predictions_metric_datas, label_metric_datas, model_predictions, labels


@lru_cache
def get_model_predictions(
    project_file_structure: ProjectFileStructure,
    predictions_filters: PredictionsFilters = PredictionsFilters(),
):
    predictions_dir = project_file_structure.predictions / MainPredictionType.OBJECT.value
    predictions_metric_datas, label_metric_datas, model_predictions, labels = read_prediction_files(
        project_file_structure, MainPredictionType.OBJECT
    )
    if model_predictions is None:
        raise Exception("Couldn't load model predictions")

    if labels is None:
        raise Exception("Couldn't load labels properly")

    matched_gt = reader.get_gt_matched(predictions_dir)
    if not matched_gt:
        raise Exception("Couldn't match ground truths")

    all_classes_objects = get_class_idx(project_file_structure.predictions / MainPredictionType.OBJECT.value)
    # selected_classes_objects = list(all_classes_objects.values())[0]

    (predictions_filtered, labels_filtered, metrics, precisions,) = compute_mAP_and_mAR(
        model_predictions,
        labels,
        matched_gt,
        all_classes_objects,
        iou_threshold=predictions_filters.iou_threshold,
        ignore_unmatched_frames=predictions_filters.ignore_frames_without_predictions,
    )

    # Sort predictions and labels according to selected metrics.
    pred_sort_column = predictions_metric_datas[0].name
    sorted_model_predictions = predictions_filtered.sort_values([pred_sort_column], axis=0)

    label_sort_column = label_metric_datas[0].name
    sorted_labels = labels_filtered.sort_values([label_sort_column], axis=0)

    if predictions_filters.ignore_frames_without_predictions:
        labels_filtered = filter_labels_for_frames_wo_predictions(predictions_filtered, sorted_labels)
    else:
        labels_filtered = sorted_labels

    labels, metrics, model_predictions, precisions = prediction_and_label_filtering(
        all_classes_objects,
        labels_filtered,
        metrics,
        sorted_model_predictions,
        precisions,
    )

    model_predictions = model_predictions.set_index("identifier")

    if predictions_filters.outcome:
        value = 1.0 if predictions_filters.outcome == ObjectDetectionOutcomeType.TRUE_POSITIVES else 0.0
        model_predictions = model_predictions[model_predictions[PredictionMatchSchema.is_true_positive] == value]

    return model_predictions, predictions_metric_datas
