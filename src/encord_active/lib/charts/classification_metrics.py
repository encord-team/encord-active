from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn import metrics

from encord_active.lib.model_predictions.reader import OntologyClassificationJSON


def get_confusion_matrix(labels: list, predictions: list, class_names: list) -> go.Figure:

    cm = metrics.confusion_matrix(labels, predictions)
    # class_names = sorted(list(set(labels).union(predictions)))
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Real", color="RdBu"),
        title="Confusion Matrix",
        x=class_names,
        y=class_names,
        aspect="auto",
    )

    return fig


def get_precision_recall_f1(labels: list, predictions: list) -> List[float]:

    return metrics.precision_recall_fscore_support(labels, predictions, zero_division=0)


def get_accuracy(labels: list, predictions: list) -> float:
    return metrics.accuracy_score(labels, predictions)


def get_roc_curve(labels: list, predictions: list, probs: list) -> go.Figure:
    pass


def get_precision_recall_graph(precision: np.ndarray, recall: np.ndarray, class_names: List) -> go.Figure:
    pr_df = pd.DataFrame(
        {
            "class": class_names + class_names,
            "score": np.append(precision, recall),
            "metric": ["Precision"] * len(class_names) + ["Recall"] * len(class_names),
        }
    )
    fig = px.histogram(pr_df, x="class", y="score", color="metric", barmode="group")

    return fig
