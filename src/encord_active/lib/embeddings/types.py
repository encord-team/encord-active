from typing import Optional, TypedDict

import numpy as np
import pandera as pa
from pandera.typing import Series

from encord_active.lib.metrics.utils import IdentifierSchema


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


class PointSchema2D(pa.SchemaModel):
    x: Series[float] = pa.Field(coerce=True)
    y: Series[float] = pa.Field(coerce=True)


class PointSelectionSchema(PointSchema2D):
    curveNumber: Series[float] = pa.Field(coerce=True)
    pointNumber: Series[float] = pa.Field(coerce=True)
    pointIndex: Series[float] = pa.Field(coerce=True)


class Embedding2DSchema(IdentifierSchema, PointSchema2D):
    label: Series[str]


class Embedding2DScoreSchema(Embedding2DSchema):
    score: Series[float]
