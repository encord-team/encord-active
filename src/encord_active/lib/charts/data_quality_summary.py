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
        title_text="Outliers", height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def create_labels_distribution_chart(labels: dict, title: str, x_title: str='Class', y_title: str='Count') -> go.Figure:
    object_df = pd.DataFrame.from_dict(labels, orient='index').reset_index()
    object_df.rename(columns={'index': x_title, 0: y_title}, inplace=True)

    object_df.sort_values(by=y_title, ascending=False, inplace=True)

    fig = px.bar(object_df, x = x_title, y=y_title)
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=300,
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
