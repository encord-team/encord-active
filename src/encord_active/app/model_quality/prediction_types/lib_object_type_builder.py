from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, FrozenSet, List, NamedTuple, Optional, Set, Union, cast

import pandas as pd
from pandera.typing import DataFrame
from pydantic import BaseModel, Field

from encord_active.lib.metrics.utils import MetricData
from encord_active.lib.model_predictions.classification_metrics import (
    match_predictions_and_labels,
)
from encord_active.lib.model_predictions.filters import (
    filter_labels_for_frames_wo_predictions,
    prediction_and_label_filtering_classification,
    prediction_and_label_filtering_detection,
)
from encord_active.lib.model_predictions.map_mar import compute_mAP_and_mAR
from encord_active.lib.model_predictions.reader import (
    ClassificationLabelSchema,
    ClassificationPredictionMatchSchemaWithClassNames,
    ClassificationPredictionSchema,
    LabelMatchSchema,
    LabelSchema,
    PredictionMatchSchema,
    PredictionSchema,
    get_class_idx,
    get_gt_matched,
    get_label_metric_data,
    get_labels,
    get_prediction_metric_data,
    load_model_predictions,
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
    type: MainPredictionType
    outcome: Optional[Union[ClassificationOutcomeType, ObjectDetectionOutcomeType]] = None
    iou_threshold: Annotated[float, Field(ge=0, le=1)] = 0.5
    ignore_frames_without_predictions: bool = True

    class Config:
        frozen = True


@lru_cache
def read_prediction_files(project_file_structure: ProjectFileStructure, prediction_type: MainPredictionType):
    metrics_dir = project_file_structure.metrics
    predictions_dir = project_file_structure.predictions / prediction_type.value

    predictions_metric_datas = get_prediction_metric_data(predictions_dir, metrics_dir)
    label_metric_datas = get_label_metric_data(metrics_dir)

    model_predictions = load_model_predictions(predictions_dir, predictions_metric_datas, prediction_type)
    labels = get_labels(predictions_dir, label_metric_datas, prediction_type)

    return predictions_metric_datas, label_metric_datas, model_predictions, labels


def get_model_prediction_by_id(project_file_structure: ProjectFileStructure, id: str):
    for prediction_type in MainPredictionType:
        _, _, model_predictions, _ = read_prediction_files(project_file_structure, prediction_type)
        if model_predictions is not None and id in pd.Index(model_predictions["identifier"]):
            try:
                df, _ = get_model_predictions(project_file_structure, PredictionsFilters(type=prediction_type))
                if df is not None:
                    return {**df.loc[id].dropna().to_dict(), "identifier": id}
            except Exception as err:
                pass


@lru_cache
def get_model_predictions(
    project_file_structure: ProjectFileStructure,
    predictions_filters: PredictionsFilters,
):
    predictions_dir = project_file_structure.predictions / predictions_filters.type.value
    predictions_metric_datas, label_metric_datas, model_predictions, labels = read_prediction_files(
        project_file_structure, predictions_filters.type
    )
    if model_predictions is None:
        raise Exception("Couldn't load model predictions")

    if labels is None:
        raise Exception("Couldn't load labels properly")

    if predictions_filters.type == MainPredictionType.OBJECT:
        return get_object_detection_predictions(
            predictions_filters,
            predictions_dir,
            predictions_metric_datas,
            label_metric_datas,
            cast(DataFrame[PredictionSchema], model_predictions),
            cast(DataFrame[LabelSchema], labels),
        )
    else:
        return get_classification_predictions(
            predictions_filters,
            predictions_dir,
            predictions_metric_datas,
            label_metric_datas,
            cast(DataFrame[ClassificationPredictionSchema], model_predictions),
            cast(DataFrame[ClassificationLabelSchema], labels),
        )


def get_object_detection_predictions(
    predictions_filters: PredictionsFilters,
    predictions_dir: Path,
    predictions_metric_datas: List[MetricData],
    label_metric_datas: List[MetricData],
    model_predictions: DataFrame[PredictionSchema],
    labels: DataFrame[LabelSchema],
):
    matched_gt = get_gt_matched(predictions_dir)
    if not matched_gt:
        raise Exception("Couldn't match ground truths")

    all_classes_objects = get_class_idx(predictions_dir)

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

    labels, metrics, model_predictions, precisions = prediction_and_label_filtering_detection(
        all_classes_objects,
        labels_filtered,
        metrics,
        sorted_model_predictions,
        precisions,
    )

    if predictions_filters.outcome in [
        ObjectDetectionOutcomeType.TRUE_POSITIVES,
        ObjectDetectionOutcomeType.FALSE_POSITIVES,
    ]:
        value = 1.0 if predictions_filters.outcome == ObjectDetectionOutcomeType.TRUE_POSITIVES else 0.0
        model_predictions = model_predictions[model_predictions[PredictionMatchSchema.is_true_positive] == value]
    elif predictions_filters.outcome == ObjectDetectionOutcomeType.FALSE_NEGATIVES:
        model_predictions = labels[labels[LabelMatchSchema.is_false_negative]]

    model_predictions = model_predictions.set_index("identifier")

    return model_predictions, predictions_metric_datas


def get_classification_predictions(
    predictions_filters: PredictionsFilters,
    predictions_dir: Path,
    predictions_metric_datas: List[MetricData],
    label_metric_datas: List[MetricData],
    model_predictions: DataFrame[ClassificationPredictionSchema],
    labels: DataFrame[ClassificationLabelSchema],
):
    model_predictions_matched = match_predictions_and_labels(model_predictions, labels)

    all_classes_classifications = get_class_idx(predictions_dir)

    (
        labels_filtered,
        predictions_filtered,
        model_predictions_matched_filtered,
    ) = prediction_and_label_filtering_classification(
        all_classes_classifications,
        all_classes_classifications,
        labels,
        model_predictions,
        model_predictions_matched,
    )

    img_id_intersection = list(
        set(labels_filtered[ClassificationLabelSchema.img_id]).intersection(
            set(predictions_filtered[ClassificationPredictionSchema.img_id])
        )
    )
    labels_filtered_intersection = labels_filtered[
        labels_filtered[ClassificationLabelSchema.img_id].isin(img_id_intersection)
    ]
    predictions_filtered_intersection = predictions_filtered[
        predictions_filtered[ClassificationPredictionSchema.img_id].isin(img_id_intersection)
    ]

    _labels, _predictions = (
        list(labels_filtered_intersection[ClassificationLabelSchema.class_id]),
        list(predictions_filtered_intersection[ClassificationPredictionSchema.class_id]),
    )

    _model_predictions = model_predictions_matched_filtered.copy()[
        model_predictions_matched_filtered[ClassificationPredictionMatchSchemaWithClassNames.img_id].isin(
            img_id_intersection
        )
    ]

    _model_predictions = _model_predictions.set_index("identifier")

    if predictions_filters.outcome:
        value = 1.0 if predictions_filters.outcome == ClassificationOutcomeType.CORRECT_CLASSIFICATIONS else 0.0
        _model_predictions = _model_predictions[
            _model_predictions[ClassificationPredictionMatchSchemaWithClassNames.is_true_positive] == value
        ]

    return _model_predictions, predictions_metric_datas
