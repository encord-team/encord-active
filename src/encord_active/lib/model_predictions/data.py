from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, cast

import pandas as pd
import pandera as pa
import pandera.dtypes as padt
from natsort import natsorted
from pandera.typing import DataFrame, Series

from encord_active.lib.common.image_utils import load_json
from encord_active.lib.metrics.load_metrics import (
    IdentifierSchema,
    MetricData,
    load_available_metrics,
    load_metric,
)


@dataclass
class MetricEntryPoint:
    metric_path: Path
    is_predictions: bool
    filter_fn: Optional[Callable[[MetricData], Any]] = None


class LabelSchema(IdentifierSchema):
    url: Series[str] = pa.Field()
    img_id: Series[padt.Int64] = pa.Field(coerce=True)
    class_id: Series[padt.Int64] = pa.Field(coerce=True)
    x1: Series[padt.Int64] = pa.Field(coerce=True)
    y1: Series[padt.Int64] = pa.Field(coerce=True)
    x2: Series[padt.Int64] = pa.Field(coerce=True)
    y2: Series[padt.Int64] = pa.Field(coerce=True)
    rle: Series[object] = pa.Field()


class PredictionSchema(LabelSchema):
    confidence: Series[padt.Float64] = pa.Field(coerce=True)
    iou: Series[padt.Float64] = pa.Field(coerce=True)


def check_model_prediction_availability(predictions_dir):
    predictions_path = predictions_dir / "predictions.csv"
    return predictions_path.is_file()


def filter_none_empty_metrics(metric: MetricData) -> str:
    with metric.path.open("r", encoding="utf-8") as f:
        f.readline()  # header
        key, *_ = f.readline().split(",")
    return key


def filter_label_metrics_for_predictions(metric: MetricData) -> bool:
    key = filter_none_empty_metrics(metric)
    if not key:
        return False
    _, _, _, *rest = key.split("_")
    return not rest


def get_metric_data(entry_points: List[MetricEntryPoint], append_level_to_titles: bool = True) -> List[MetricData]:
    all_metrics: List[MetricData] = []

    for entry in entry_points:
        available_metrics: Iterable[MetricData] = list(
            filter(entry.filter_fn, load_available_metrics(entry.metric_path, None))
        )
        if not append_level_to_titles:
            all_metrics += natsorted(available_metrics, key=lambda x: x.name)
            continue

        for metric in available_metrics:
            metric.name += " (P)" if entry.is_predictions else f" ({metric.level})"
        all_metrics += natsorted(available_metrics, key=lambda x: x.name[-3:] + x.name[:-3])

    return all_metrics


def append_metric_columns(df: pd.DataFrame, metric_entries: List[MetricData]) -> pd.DataFrame:
    IdentifierSchema.validate(df)
    df = df.copy()
    df["identifier_no_oh"] = df[IdentifierSchema.identifier].str.replace(r"^(\S{73}_\d+)(.*)", r"\1", regex=True)

    for metric in metric_entries:
        metric_scores = load_metric(metric, normalize=False)
        metric_scores = metric_scores.set_index(keys=["identifier"])

        has_object_level_keys = len(cast(str, metric_scores.index[0]).split("_")) > 3
        metric_column = "identifier" if has_object_level_keys else "identifier_no_oh"

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


def get_prediction_metric_data(predictions_dir: Path, metrics_dir: Path) -> List[MetricData]:
    predictions_metric_dir = predictions_dir / "metrics"
    if not predictions_metric_dir.is_dir():
        return []

    entry_points = [
        MetricEntryPoint(metrics_dir, is_predictions=False, filter_fn=filter_label_metrics_for_predictions),
        MetricEntryPoint(predictions_metric_dir, is_predictions=True, filter_fn=filter_none_empty_metrics),
    ]
    return get_metric_data(entry_points)


def get_model_predictions(
    predictions_dir: Path, metric_data: List[MetricData]
) -> Optional[DataFrame[PredictionSchema]]:
    df = _load_csv_and_merge_metrics(predictions_dir / "predictions.csv", metric_data)
    return df.pipe(DataFrame[PredictionSchema]) if df is not None else None


def get_label_metric_data(metrics_dir: Path) -> List[MetricData]:
    if not metrics_dir.is_dir():
        return []

    entry_points = [
        MetricEntryPoint(metrics_dir, is_predictions=False, filter_fn=filter_none_empty_metrics),
    ]
    return get_metric_data(entry_points)


def get_labels(predictions_dir: Path, metric_data: List[MetricData]) -> Optional[pd.DataFrame]:
    df = _load_csv_and_merge_metrics(predictions_dir / "labels.csv", metric_data)
    return df.pipe(DataFrame[LabelSchema]) if df is not None else None


def get_gt_matched(predictions_dir: Path) -> Optional[dict]:
    gt_path = predictions_dir / "ground_truths_matched.json"
    return load_json(gt_path)


def get_class_idx(predictions_dir: Path) -> Optional[dict]:
    class_idx_pth = predictions_dir / "class_idx.json"
    return load_json(class_idx_pth)


def filter_labels_for_frames_wo_predictions(
    model_predictions: DataFrame[PredictionSchema], sorted_labels: pd.DataFrame
):
    pred_keys = model_predictions["img_id"].unique()
    return sorted_labels[sorted_labels["img_id"].isin(pred_keys)]


def prediction_and_label_filtering(
    selected_class_idx: dict,
    labels: pd.DataFrame,
    metrics: pd.DataFrame,
    model_pred: pd.DataFrame,
    precisions: pd.DataFrame,
):
    # Predictions
    class_idx = selected_class_idx
    row_selection = model_pred["class_id"].isin(set(map(int, class_idx.keys())))
    _model_pred = model_pred[row_selection].copy()

    # Labels
    row_selection = labels["class_id"].isin(set(map(int, class_idx.keys())))
    _labels = labels[row_selection]

    chosen_name_set = set(map(lambda x: x["name"], class_idx.values())).union({"Mean"})
    _metrics = metrics[metrics["class_name"].isin(chosen_name_set)]
    _precisions = precisions[precisions["class_name"].isin(chosen_name_set)]
    name_map = {int(k): v["name"] for k, v in class_idx.items()}
    _model_pred["class_name"] = _model_pred["class_id"].map(name_map)
    _labels["class_name"] = _labels["class_id"].map(name_map)
    return _labels, _metrics, _model_pred, _precisions
