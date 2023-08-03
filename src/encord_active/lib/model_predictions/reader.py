from pathlib import Path
from typing import Iterable, List, Optional, Union, cast

import pandas as pd
from cachetools import LRUCache, cached
from loguru import logger
from natsort import natsorted
from pandera.typing import DataFrame

from encord_active.lib.common.utils import load_json
from encord_active.lib.metrics.utils import (
    IdentifierSchema,
    MetricData,
    filter_none_empty_metrics,
    load_available_metrics,
    load_metric_dataframe,
)
from encord_active.lib.model_predictions.filters import (
    filter_labels_for_frames_wo_predictions,
    prediction_and_label_filtering_classification,
    prediction_and_label_filtering_detection,
)
from encord_active.lib.model_predictions.map_mar import compute_mAP_and_mAR
from encord_active.lib.model_predictions.types import (
    ClassificationLabelSchema,
    ClassificationOutcomeType,
    ClassificationPredictionMatchSchema,
    ClassificationPredictionMatchSchemaWithClassNames,
    ClassificationPredictionSchema,
    LabelMatchSchema,
    LabelSchema,
    MetricEntryPoint,
    ObjectDetectionOutcomeType,
    OntologyObjectJSON,
    PredictionMatchSchema,
    PredictionSchema,
    PredictionsFilters,
)
from encord_active.lib.model_predictions.writer import MainPredictionType
from encord_active.lib.project.project_file_structure import ProjectFileStructure


def check_model_prediction_availability(predictions_dir):
    predictions_path = predictions_dir / "predictions.csv"
    return predictions_path.is_file()


def filter_label_metrics_for_predictions(metric: MetricData) -> bool:
    key = filter_none_empty_metrics(metric)
    if not key:
        return False
    _, _, _, *rest = key.split("_")
    return not rest


def read_metric_data(entry_points: List[MetricEntryPoint], append_level_to_titles: bool = True) -> List[MetricData]:
    all_metrics: List[MetricData] = []

    for entry in entry_points:
        available_metrics: Iterable[MetricData] = list(
            filter(entry.filter_fn, load_available_metrics(entry.metric_path, None))
        )
        if not append_level_to_titles:
            all_metrics += natsorted(available_metrics, key=lambda x: x.name)
            continue

        all_metrics += natsorted(available_metrics, key=lambda x: x.name[-3:] + x.name[:-3])

    return all_metrics


def append_metric_columns(df: pd.DataFrame, metric_entries: List[MetricData]) -> pd.DataFrame:
    IdentifierSchema.validate(df)
    df = df.copy()
    df["identifier_no_oh"] = df[IdentifierSchema.identifier].str.replace(r"^(\S{73}_\d+)(.*)", r"\1", regex=True)

    for metric in metric_entries:
        metric_scores = load_metric_dataframe(metric, normalize=False)
        if metric_scores.index.name != "identifier":
            metric_scores.set_index("identifier", inplace=True)

        has_object_level_keys = len(cast(str, metric_scores.index[0]).split("_")) > 3
        metric_column = "identifier" if has_object_level_keys else "identifier_no_oh"

        # TODO: When EA supports different classification questions, the following must be fixed
        if (len(df[IdentifierSchema.identifier][0].split("_")) == 3) and (has_object_level_keys):
            metric_scores = metric_scores.copy()  # type: ignore
            metric_scores.index = metric_scores.index.str.replace(r"^(\S{73}_\d+)(.*)", r"\1", regex=True)

        # Join data and rename column to metric name.
        df = df.join(metric_scores["score"], on=[metric_column])
        df[metric.name] = df["score"]
        df.drop("score", axis=1, inplace=True)

    df.drop("identifier_no_oh", axis=1, inplace=True)
    return df


def _load_csv_and_merge_metrics(path: Path, metric_data: List[MetricData]) -> Optional[pd.DataFrame]:
    if not path.suffix == ".csv":
        return None

    if not path.is_file():
        return None

    df = cast(pd.DataFrame, pd.read_csv(path, iterator=False))
    return append_metric_columns(df, metric_data)


def read_prediction_metric_data(predictions_dir: Path, metrics_dir: Path) -> List[MetricData]:
    predictions_metric_dir = predictions_dir / "metrics"
    if not predictions_metric_dir.is_dir():
        return []

    entry_points = [
        MetricEntryPoint(metrics_dir, is_predictions=False, filter_fn=filter_label_metrics_for_predictions),
        MetricEntryPoint(predictions_metric_dir, is_predictions=True, filter_fn=filter_none_empty_metrics),
    ]
    return read_metric_data(entry_points)


def read_model_predictions(
    predictions_dir: Path, metric_data: List[MetricData], prediction_type: MainPredictionType
) -> Union[DataFrame[PredictionSchema], DataFrame[ClassificationPredictionSchema], None]:
    df = _load_csv_and_merge_metrics(predictions_dir / "predictions.csv", metric_data)

    if prediction_type == MainPredictionType.CLASSIFICATION:
        return df.pipe(DataFrame[ClassificationPredictionSchema]) if df is not None else None
    elif prediction_type == MainPredictionType.OBJECT:
        return df.pipe(DataFrame[PredictionSchema]) if df is not None else None
    else:
        logger.error(f"Undefined prediction type: {prediction_type}")
        return None


def read_label_metric_data(metrics_dir: Path) -> List[MetricData]:
    if not metrics_dir.is_dir():
        return []

    entry_points = [
        MetricEntryPoint(metrics_dir, is_predictions=False, filter_fn=filter_none_empty_metrics),
    ]
    return read_metric_data(entry_points)


def read_labels(
    predictions_dir: Path, metric_data: List[MetricData], prediction_type: MainPredictionType
) -> Union[DataFrame[LabelSchema], DataFrame[ClassificationLabelSchema], None]:
    df = _load_csv_and_merge_metrics(predictions_dir / "labels.csv", metric_data)

    if prediction_type == MainPredictionType.OBJECT:
        return df.pipe(DataFrame[LabelSchema]) if df is not None else None
    elif prediction_type == MainPredictionType.CLASSIFICATION:
        return df.pipe(DataFrame[ClassificationLabelSchema]) if df is not None else None
    else:
        logger.error(f"Undefined prediction type: {prediction_type}")
        return None


def read_gt_matched(predictions_dir: Path) -> Optional[dict]:
    gt_path = predictions_dir / "ground_truths_matched.json"
    return load_json(gt_path)


def read_class_idx(predictions_dir: Path) -> dict[str, OntologyObjectJSON]:
    class_idx_pth = predictions_dir / "class_idx.json"
    return load_json(class_idx_pth) or {}


def read_classification_labels(predictions_dir: Path) -> Optional[DataFrame[ClassificationLabelSchema]]:
    predictions = pd.read_csv(predictions_dir / "labels.csv")
    return predictions.pipe(DataFrame[ClassificationLabelSchema])


def read_classification_predictions(predictions_dir: Path) -> Optional[DataFrame[ClassificationPredictionSchema]]:
    labels = pd.read_csv(predictions_dir / "predictions.csv")
    return labels.pipe(DataFrame[ClassificationPredictionSchema])


def match_predictions_and_labels(
    model_predictions: DataFrame[ClassificationPredictionSchema], labels: DataFrame[ClassificationLabelSchema]
) -> DataFrame[ClassificationPredictionMatchSchema]:
    _model_predictions = model_predictions.copy()
    _labels = labels.copy()

    _model_predictions[ClassificationPredictionMatchSchema.is_true_positive] = (
        _model_predictions[ClassificationPredictionSchema.class_id]
        .eq(_labels[ClassificationLabelSchema.class_id])
        .astype(float)
    )
    _model_predictions[ClassificationPredictionMatchSchema.gt_class_id] = _labels[ClassificationLabelSchema.class_id]

    return _model_predictions.pipe(DataFrame[ClassificationPredictionMatchSchema])


@cached(cache=LRUCache(maxsize=10))
def read_prediction_files(project_file_structure: ProjectFileStructure, prediction_type: MainPredictionType):
    metrics_dir = project_file_structure.metrics
    predictions_dir = project_file_structure.predictions / prediction_type.value

    predictions_metric_datas = read_prediction_metric_data(predictions_dir, metrics_dir)
    label_metric_datas = read_label_metric_data(metrics_dir)

    model_predictions = read_model_predictions(predictions_dir, predictions_metric_datas, prediction_type)
    labels = read_labels(predictions_dir, label_metric_datas, prediction_type)

    return predictions_metric_datas, label_metric_datas, model_predictions, labels


def get_model_prediction_by_id(project_file_structure: ProjectFileStructure, id: str, iou: Optional[float] = None):
    for prediction_type in MainPredictionType:
        _, _, model_predictions, _ = read_prediction_files(project_file_structure, prediction_type)
        if model_predictions is not None and id in pd.Index(model_predictions["identifier"]):
            try:
                df, _ = get_model_predictions(
                    project_file_structure,
                    PredictionsFilters(type=prediction_type, iou_threshold=iou),
                )
                if df is not None:
                    return {**df.loc[id].dropna().to_dict(), "identifier": id}
            except Exception as err:
                pass


@cached(cache=LRUCache(maxsize=10))
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
    matched_gt = read_gt_matched(predictions_dir)
    if not matched_gt:
        raise Exception("Couldn't match ground truths")

    all_classes_objects = read_class_idx(predictions_dir)

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
    labels_filtered.sort_values([label_sort_column], axis=0, inplace=True)

    if predictions_filters.ignore_frames_without_predictions:
        labels_filtered = filter_labels_for_frames_wo_predictions(predictions_filtered, labels_filtered)

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
        model_predictions = cast(
            DataFrame[PredictionSchema],
            model_predictions[model_predictions[PredictionMatchSchema.is_true_positive] == value],
        )
    elif predictions_filters.outcome == ObjectDetectionOutcomeType.FALSE_NEGATIVES:
        model_predictions = cast(DataFrame[PredictionSchema], labels[labels[LabelMatchSchema.is_false_negative]])

    model_predictions.set_index("identifier", inplace=True)
    labels.set_index("identifier", inplace=True)

    return model_predictions, labels


def get_classification_predictions(
    predictions_filters: PredictionsFilters,
    predictions_dir: Path,
    predictions_metric_datas: List[MetricData],
    label_metric_datas: List[MetricData],
    model_predictions: DataFrame[ClassificationPredictionSchema],
    labels: DataFrame[ClassificationLabelSchema],
):
    model_predictions_matched = match_predictions_and_labels(model_predictions, labels)

    all_classes_classifications = read_class_idx(predictions_dir)

    (labels, model_predictions, model_predictions_matched_filtered,) = prediction_and_label_filtering_classification(
        all_classes_classifications,
        all_classes_classifications,
        labels,
        model_predictions,
        model_predictions_matched,
    )

    img_id_intersection = list(
        set(labels[ClassificationLabelSchema.img_id]).intersection(
            set(model_predictions[ClassificationPredictionSchema.img_id])
        )
    )

    _model_predictions = model_predictions_matched_filtered.copy()[
        model_predictions_matched_filtered[ClassificationPredictionMatchSchemaWithClassNames.img_id].isin(
            img_id_intersection
        )
    ]

    _model_predictions.set_index("identifier", inplace=True)
    labels.set_index("identifier", inplace=True)

    if predictions_filters.outcome:
        value = 1.0 if predictions_filters.outcome == ClassificationOutcomeType.CORRECT_CLASSIFICATIONS else 0.0
        _model_predictions = _model_predictions[
            _model_predictions[ClassificationPredictionMatchSchemaWithClassNames.is_true_positive] == value
        ]

    return _model_predictions, labels
