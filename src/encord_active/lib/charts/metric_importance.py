from typing import Any, List, Union

import altair as alt
import pandas as pd
from pandera.typing import DataFrame
from sklearn.feature_selection import mutual_info_regression

from encord_active.lib.model_predictions.types import (
    ClassificationPredictionMatchSchema,
    PredictionMatchSchema,
)
from encord_active.lib.model_predictions.writer import MainPredictionType


def create_metric_importance_charts(
    model_predictions: Union[DataFrame[PredictionMatchSchema], DataFrame[ClassificationPredictionMatchSchema]],
    metric_columns: List[str],
    num_samples: int,
    prediction_type: MainPredictionType,
):
    if num_samples < model_predictions.shape[0]:
        _predictions = model_predictions.sample(num_samples, axis=0, random_state=42)
    else:
        _predictions = model_predictions

    _P_COLS: Any
    if prediction_type == MainPredictionType.OBJECT:
        _P_COLS = PredictionMatchSchema
        scores = _predictions[_P_COLS.iou] * _predictions[_P_COLS.is_true_positive]
    elif prediction_type == MainPredictionType.CLASSIFICATION:
        _P_COLS = ClassificationPredictionMatchSchema
        scores = _predictions[_P_COLS.is_true_positive]
    else:
        raise ValueError(f"Undefined prediction type {prediction_type}")

    num_tps = (_predictions[_P_COLS.is_true_positive].iloc[:num_samples] != 0).sum()
    if num_tps < 50:
        raise ValueError(
            f"Not enough true positives ({num_tps}) to calculate reliable metric importance. "
            "Try increasing the number of samples or lower the IoU threshold in the top bar."
        )

    metrics = _predictions[metric_columns]

    correlations = metrics.fillna(0).corrwith(scores, axis=0).to_frame("correlation")
    correlations["index"] = correlations.index.T

    mi = pd.DataFrame.from_dict(
        {"index": metrics.columns, "importance": mutual_info_regression(metrics.fillna(0), scores, random_state=42)}
    )
    # pylint: disable=unsubscriptable-object
    sorted_metrics: List[str] = mi.sort_values("importance", ascending=False, inplace=False)["index"].to_list()

    mutual_info_bars = (
        alt.Chart(mi, title="Metric Importance")
        .mark_bar()
        .encode(
            alt.X("importance", title="Importance", scale=alt.Scale(domain=[0.0, 1.0])),
            alt.Y("index", title="Metric", sort=sorted_metrics),
            alt.Color("importance", scale=alt.Scale(scheme="blues")),
            tooltip=[
                alt.Tooltip("index", title="Metric"),
                alt.Tooltip("importance", title="Importance", format=",.3f"),
            ],
        )
    )

    correlation_bars = (
        alt.Chart(correlations, title="Metric Correlations")
        .mark_bar()
        .encode(
            alt.X("correlation", title="Correlation", scale=alt.Scale(domain=[-1.0, 1.0])),
            alt.Y("index", title="Metric", sort=sorted_metrics),
            alt.Color("correlation", scale=alt.Scale(scheme="redyellowgreen", align=0.0)),
            tooltip=[
                alt.Tooltip("index", title="Metric"),
                alt.Tooltip("correlation", title="Correlation", format=",.3f"),
            ],
        )
    )

    return alt.hconcat(mutual_info_bars, correlation_bars).resolve_scale(color="independent")
