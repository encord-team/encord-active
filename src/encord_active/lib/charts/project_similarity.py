import pandas as pd
import pandera as pa
import plotly.express as px
import plotly.graph_objects as go
from pandera.typing import DataFrame, Series


class ProjectSimilaritySchema(pa.SchemaModel):
    metric: Series[str] = pa.Field()
    similarity_score: Series[float] = pa.Field(coerce=True)


def plot_project_similarity_metric_wise(metric_wise_similarity: DataFrame[ProjectSimilaritySchema]) -> go.Figure:
    metric_wise_similarity.sort_values(by=ProjectSimilaritySchema.similarity_score, inplace=True)

    fig = px.bar(
        metric_wise_similarity,
        x=ProjectSimilaritySchema.similarity_score,
        y=ProjectSimilaritySchema.metric,
        orientation="h",
    )
    fig.update_layout(
        title="Metric-wise similarity score",
        xaxis_title="Similarity score",
        yaxis_title="Metric",
    )

    return fig


def render_2d_metric_similarity_plot(
    project_values: pd.DataFrame, metric_name_1: str, metric_name_2: str, project_name_1: str, project_name_2: str
):
    fig = px.scatter(
        project_values,
        x=metric_name_1,
        y=metric_name_2,
        color="project",
        color_discrete_map={project_name_1: "#5658dd", project_name_2: "red"},  # Encord brand colors: #5658dd, #89e7b6
        opacity=0.7,
        title="2D metric comparison",
        template="plotly",
    )

    fig.update_traces(
        marker=dict(
            symbol="circle",  # Choose the shape of the marker
            size=8,  # Set the size of the marker
            # line=dict(  # Configure the lines around the markers
            #     color="black", width=1  # Set the line color  # Set the line width
            # ),
        )
    )

    fig.update_layout(
        margin=dict(l=0, r=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
