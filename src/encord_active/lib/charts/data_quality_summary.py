import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def create_outlier_distribution_chart(all_metrics_outliers_summary: pd.DataFrame,
                                      severe_outlier_color: str,
                                      moderate_outlier_color: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=all_metrics_outliers_summary['metric'],
        y=all_metrics_outliers_summary['total_severe_outliers'],
        name='Severe outliers',
        marker_color=severe_outlier_color
    ))
    fig.add_trace(go.Bar(
        x=all_metrics_outliers_summary['metric'],
        y=all_metrics_outliers_summary['total_moderate_outliers'],
        name='Moderate outliers',
        marker_color=moderate_outlier_color
    ))
    fig.update_layout(title_text="Outliers",
                      height=400,
                      legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    return fig


def create_image_size_distribution_chart(image_sizes:np.ndarray)-> go.Figure:
    fig = px.scatter(x=image_sizes[:, 0], y=image_sizes[:, 1])
    fig.update_layout(
        title="Image resolutions",
        xaxis_title="Width",
        yaxis_title="Height",
        height=400,
    )

    return fig