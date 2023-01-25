import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_outlier_distribution_chart(
    all_metrics_outliers_summary: pd.DataFrame, severe_outlier_color: str, moderate_outlier_color: str
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=all_metrics_outliers_summary["metric"],
            y=all_metrics_outliers_summary["total_severe_outliers"],
            name="Severe outliers",
            marker_color=severe_outlier_color,
        )
    )
    fig.add_trace(
        go.Bar(
            x=all_metrics_outliers_summary["metric"],
            y=all_metrics_outliers_summary["total_moderate_outliers"],
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
    labels_df.rename(columns={"index": x_title, 0: y_title}, inplace=True)

    labels_df.sort_values(by=y_title, ascending=False, inplace=True)

    Q2 = labels_df[y_title].quantile(0.5)

    labels_df["undersampled"] = False
    labels_df.loc[labels_df[y_title] <= (Q2 * 0.5), "undersampled"] = True

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels_df.loc[labels_df["undersampled"] == False][x_title],
                y=labels_df[labels_df["undersampled"] == False][y_title],
                name="representative",
                marker_color="#3380FF",
            ),
            go.Bar(
                x=labels_df.loc[labels_df["undersampled"] == True][x_title],
                y=labels_df[labels_df["undersampled"] == True][y_title],
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
