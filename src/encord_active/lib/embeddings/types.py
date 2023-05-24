from typing import Optional, TypedDict

import numpy as np


class ClassificationAnswer(TypedDict):
    answer_featureHash: str
    answer_name: str
    annotator: str


class LabelEmbedding(TypedDict):
    label_row: str
    data_unit: str
    frame: int
    url: str
    labelHash: Optional[str]
    lastEditedBy: Optional[str]
    featureHash: Optional[str]
    name: Optional[str]
    dataset_title: str
    embedding: np.ndarray
    classification_answers: Optional[ClassificationAnswer]
