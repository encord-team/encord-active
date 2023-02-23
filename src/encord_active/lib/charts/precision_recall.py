import json
from pathlib import Path

import altair as alt
import plotly.graph_objects as go
from pandera.typing import DataFrame
from plotly.subplots import make_subplots

from encord_active.lib.model_predictions.map_mar import (
    PerformanceMetricSchema,
    PrecisionRecallSchema,
)

_PR_COLS = PrecisionRecallSchema
_M_COLS = PerformanceMetricSchema


def create_pr_charts(metrics: DataFrame[PerformanceMetricSchema], precisions: DataFrame[PrecisionRecallSchema]):
    _metrics: DataFrame[PerformanceMetricSchema] = metrics[~metrics[_M_COLS.metric].isin({"mAR", "mAP"})].copy()

    tmp = "m" + _metrics["metric"].str.split("_", n=1, expand=True)
    tmp.columns = ["group", "_"]
    _metrics["group"] = tmp["group"]
    _metrics["average"] = "average"  # Legend title

    class_selection = alt.selection_multi(fields=[_M_COLS.class_name])

    _metrics.sort_values(by=["group", _M_COLS.value], ascending=[True, False], inplace=True)

    class_bars = (
        alt.Chart(_metrics, title="Mean scores")
        .mark_bar()
        .encode(
            alt.X(_M_COLS.value, title="", scale=alt.Scale(domain=[0.0, 1.0])),
            alt.Y(_M_COLS.metric, title="", sort=None),
            alt.Color(_M_COLS.class_name),
            tooltip=[
                alt.Tooltip(_M_COLS.metric, title="Metric"),
                alt.Tooltip(_M_COLS.value, title="Value", format=",.3f"),
            ],
            opacity=alt.condition(class_selection, alt.value(1), alt.value(0.1)),
        )
        .properties(height=300)
    )
    # Average
    mean_bars = class_bars.encode(
        alt.X(f"mean({_M_COLS.value}):Q", title="", scale=alt.Scale(domain=[0.0, 1.0])),
        alt.Y("group:N", title="", sort=None),
        alt.Color("average:N"),
        tooltip=[
            alt.Tooltip("group:N", title="Metric"),
            alt.Tooltip(f"mean({_M_COLS.value}):Q", title="Value", format=",.3f"),
        ],
    )
    bar_chart = (class_bars + mean_bars).add_selection(class_selection)

    class_precisions = (
        alt.Chart(precisions, title="Precision-Recall Curve")
        .mark_line(point=True)
        .encode(
            alt.X(_PR_COLS.recall, title="Recall", scale=alt.Scale(domain=[0.0, 1.0])),
            alt.Y(_PR_COLS.precision, scale=alt.Scale(domain=[0.0, 1.0])),
            alt.Color(_PR_COLS.class_name),
            tooltip=[
                alt.Tooltip(_PR_COLS.class_name),
                alt.Tooltip(_PR_COLS.recall, title="Recall"),
                alt.Tooltip(_PR_COLS.precision, title="Precision", format=",.3f"),
            ],
            opacity=alt.condition(class_selection, alt.value(1.0), alt.value(0.2)),
        )
        .properties(height=300)
    )

    mean_precisions = (
        class_precisions.transform_calculate(average="'average'")
        .mark_line(point=True)
        .encode(
            alt.X(_PR_COLS.recall),
            alt.Y(f"average({_PR_COLS.precision}):Q"),
            alt.Color("average:N"),
            tooltip=[
                alt.Tooltip("average:N", title="Aggregate"),
                alt.Tooltip(_PR_COLS.recall, title="Recall"),
                alt.Tooltip(f"average({_PR_COLS.precision})", title="Avg. precision", format=",.3f"),
            ],
        )
    )
    precision_chart = (class_precisions + mean_precisions).add_selection(class_selection)
    return bar_chart | precision_chart


def create_pr_chart_plotly(
    metrics: DataFrame[PerformanceMetricSchema], precisions: DataFrame[PrecisionRecallSchema], ontology_path: Path
):
    _metrics: DataFrame[PerformanceMetricSchema] = metrics[~metrics[_M_COLS.metric].isin({"mAR", "mAP"})].copy()

    tmp = "m" + _metrics["metric"].str.split("_", n=1, expand=True)
    tmp.columns = ["group", "_"]
    _metrics["group"] = tmp["group"]
    _metrics["average"] = "average"  # Legend title

    _metrics.sort_values(by=["group", _M_COLS.value], ascending=[False, True], inplace=True)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("AP and AR", "Precision-Recall Curve"))

    project_ontology = json.loads(ontology_path.read_text(encoding="utf-8"))

    colors = {}
    for object in project_ontology["objects"]:
        colors[object["name"]] = object["color"]

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
                _metrics.loc[_metrics["group"] == "mAP"][_M_COLS.value].mean(),
                _metrics.loc[_metrics["group"] == "mAR"][_M_COLS.value].mean(),
            ],
            y=["AP_average", "AR_average"],
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
