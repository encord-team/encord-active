import numpy as np
import pandas as pd
import pandera as pa
import plotly.express as px
import plotly.graph_objects as go
from pandera.typing import DataFrame, Series

from encord_active.lib.dataset.outliers import AllMetricsOutlierSchema


class LabelStatisticsSchema(pa.SchemaModel):
    name: Series[str] = pa.Field()
    count: Series[int] = pa.Field()
    status: Series[bool] = pa.Field()


class CrossMetricSchema(pa.SchemaModel):
    identifier: Series[str]
    x: Series[float] = pa.Field(coerce=True)
    y: Series[float] = pa.Field(coerce=True)


def create_outlier_distribution_chart(
    all_metrics_outliers_summary: DataFrame[AllMetricsOutlierSchema],
    severe_outlier_color: str,
    moderate_outlier_color: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=all_metrics_outliers_summary[AllMetricsOutlierSchema.metric_name],
            y=all_metrics_outliers_summary[AllMetricsOutlierSchema.total_severe_outliers],
            name="Severe outliers",
            marker_color=severe_outlier_color,
        )
    )
    fig.add_trace(
        go.Bar(
            x=all_metrics_outliers_summary[AllMetricsOutlierSchema.metric_name],
            y=all_metrics_outliers_summary[AllMetricsOutlierSchema.total_moderate_outliers],
            name="Moderate outliers",
            marker_color=moderate_outlier_color,
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0),
        title_text="Outliers",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_labels_distribution_chart(
    labels: dict, title: str, x_title: str = "Class", y_title: str = "Count"
) -> go.Figure:
    labels_df = pd.DataFrame.from_dict(labels, orient="index").reset_index()
    labels_df.rename(columns={"index": "name", 0: "count"}, inplace=True)
    labels_df.insert(0, "status", False)

    labels_df = DataFrame[LabelStatisticsSchema](labels_df)
    labels_df.sort_values(by=LabelStatisticsSchema.count, ascending=False, inplace=True)

    Q2 = labels_df[LabelStatisticsSchema.count].quantile(0.5)

    labels_df.loc[labels_df[LabelStatisticsSchema.count] <= (Q2 * 0.5), LabelStatisticsSchema.status] = True

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels_df.loc[labels_df[LabelStatisticsSchema.status] == False][LabelStatisticsSchema.name],
                y=labels_df[labels_df[LabelStatisticsSchema.status] == False][LabelStatisticsSchema.count],
                name="representative",
                marker_color="#3380FF",
            ),
            go.Bar(
                x=labels_df.loc[labels_df[LabelStatisticsSchema.status] == True][LabelStatisticsSchema.name],
                y=labels_df[labels_df[LabelStatisticsSchema.status] == True][LabelStatisticsSchema.count],
                name="undersampled",
                marker_color="tomato",
            ),
        ]
    )

    fig.update_layout(
        margin=dict(l=0, r=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
    )

    return fig


def create_image_size_distribution_chart(image_sizes: np.ndarray) -> go.Figure:
    fig = px.scatter(x=image_sizes[:, 0], y=image_sizes[:, 1])
    fig.update_layout(
        title="Image resolutions",
        xaxis_title="Width",
        yaxis_title="Height",
        height=400,
    )

    return fig


def create_2d_metric_chart(
    metrics_df: DataFrame[CrossMetricSchema], x_axis_title: str, y_axis_title: str, show_trendline: bool = True
) -> go.Figure:

    fig = px.scatter(
        metrics_df,
        x=metrics_df[CrossMetricSchema.x],
        y=metrics_df[CrossMetricSchema.y],
        hover_data=["identifier"],
        trendline="ols" if show_trendline else None,
        trendline_color_override="black",
    )
    fig.update_layout(
        title=f"{x_axis_title} vs. {y_axis_title}",
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
    )

    return fig
