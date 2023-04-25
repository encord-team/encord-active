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


def plot_project_metric_values_comparison():
    pass
