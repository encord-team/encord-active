from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn import metrics


def get_confusion_matrix(labels: list, predictions: list, class_names: list) -> go.Figure:
    cm = metrics.confusion_matrix(labels, predictions)

    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Real"),
        color_continuous_scale=px.colors.sequential.Blues,
        title="Confusion Matrix",
        x=class_names,
        y=class_names,
        aspect="auto",
    )
    fig.update_traces(hovertemplate="<b>Real:</b> %{y}<br><b>Predicted:</b> %{x} <br>" "<b>Count:</b> %{z}")
    fig.update_coloraxes(showscale=False)
    return fig


def get_precision_recall_f1(labels: list, predictions: list) -> List[np.ndarray]:

    return metrics.precision_recall_fscore_support(labels, predictions, zero_division=0)


def get_accuracy(labels: list, predictions: list) -> float:
    return metrics.accuracy_score(labels, predictions)


# TODO implement this function after writing all class confidences of a prediction into pickle file (currently, only confidence of the predicted class is written)
def get_roc_curve(labels: list, prediction_probs: list) -> go.Figure:
    pass


def get_precision_recall_graph(precision: np.ndarray, recall: np.ndarray, class_names: List) -> go.Figure:
    pr_df = pd.DataFrame(
        {
            "class": class_names + class_names,
            "score": np.append(precision, recall),
            "metric": ["Precision"] * len(class_names) + ["Recall"] * len(class_names),
        }
    )

    pr_df.sort_values(by=["metric", "score"], ascending=[True, False], inplace=True)

    fig = px.histogram(
        pr_df,
        x="class",
        y="score",
        color="metric",
        title="Precision-Recall",
        barmode="group",
        color_discrete_sequence=["#5658dd", "#89e7b6"],
    )
    fig.update_traces(hovertemplate="<b>Score:</b> %{y}<br><b>Class:</b> %{x} <br>")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig
