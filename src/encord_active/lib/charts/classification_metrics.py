from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn import metrics

from encord_active.lib.model_predictions.reader import OntologyClassificationJSON


def get_confusion_matrix(labels: list, predictions: list, class_names: List[str]) -> go.Figure:
    cm = metrics.confusion_matrix(labels, predictions)

    fig = px.imshow(
        cm, text_auto=True, labels=dict(x="Predicted", y="Real", color="Bluered"), x=class_names, y=class_names
    )

    return fig


def get_precision_recall_f1(labels: list, predictions: list) -> List[float]:

    return metrics.precision_recall_fscore_support(labels, predictions)


def get_accuracy(labels: list, predictions: list) -> float:
    return metrics.accuracy_score(labels, predictions)


def get_roc_curve(labels: list, predictions: list, probs: list) -> go.Figure:
    pass
