from typing import cast

import plotly.graph_objects as go
from pandera.typing import DataFrame
from plotly.subplots import make_subplots

from encord_active.lib.model_predictions.map_mar import (
    PerformanceMetricSchema,
    PrecisionRecallSchema,
)

_M_COLS = PerformanceMetricSchema


def create_pr_chart_plotly(
    metrics: DataFrame[PerformanceMetricSchema],
    precisions: DataFrame[PrecisionRecallSchema],
    annotation_colors: list[dict],
):
    _metrics = cast(DataFrame[PerformanceMetricSchema], metrics[~metrics[_M_COLS.metric].isin({"mAR", "mAP"})].copy())

    tmp = "m" + _metrics["metric"].str.split("_", n=1, expand=True)
    tmp.columns = ["group", "_"]  # type:ignore
    _metrics["group"] = tmp["group"]  # type:ignore
    _metrics["average"] = "average"  # Legend title

    _metrics.sort_values(by=["group", _M_COLS.value], ascending=[False, True], inplace=True)

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Per class Average Precision and Recall", "Precision-Recall Curve")
    )

    colors = {}
    for obj in annotation_colors:
        colors[obj["name"]] = obj["color"]

    for class_name in _metrics[PerformanceMetricSchema.class_name].unique():
        fig.add_trace(
            go.Bar(
                x=_metrics.loc[_metrics[PerformanceMetricSchema.class_name] == class_name][_M_COLS.value],
                y=_metrics.loc[_metrics[PerformanceMetricSchema.class_name] == class_name][_M_COLS.metric],
                orientation="h",
                name=class_name,
                legendgroup=class_name,
                marker={"color": colors[class_name]},
            ),
            row=1,
            col=1,
        )

    # Average
    fig.add_trace(
        go.Bar(
            x=[
                _metrics.loc[_metrics["group"] == "mAR"][_M_COLS.value].mean(),
                _metrics.loc[_metrics["group"] == "mAP"][_M_COLS.value].mean(),
            ],
            y=["AR_average", "AP_average"],
            orientation="h",
            name="Average",
            legendgroup="Average",
        ),
        row=1,
        col=1,
    )

    for class_name in precisions[PrecisionRecallSchema.class_name].unique():
        fig.add_trace(
            go.Scatter(
                x=precisions.loc[precisions[PrecisionRecallSchema.class_name] == class_name][
                    PrecisionRecallSchema.recall
                ],
                y=precisions.loc[precisions[PrecisionRecallSchema.class_name] == class_name][
                    PrecisionRecallSchema.precision
                ],
                mode="lines+markers",
                name=class_name,
                legendgroup=class_name,
                showlegend=False,
                marker={"color": colors[class_name]},
            ),
            row=1,
            col=2,
        )

    fig["layout"]["xaxis"]["title"] = "Score"
    fig["layout"]["xaxis2"]["title"] = "Recall"
    fig["layout"]["yaxis"]["title"] = "Class"
    fig["layout"]["yaxis2"]["title"] = "Precision"

    return fig
