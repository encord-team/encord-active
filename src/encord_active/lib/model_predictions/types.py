from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Callable, Optional, TypedDict, Union

import pandera as pa
import pandera.dtypes as padt
from pandera.typing import Series
from pydantic import BaseModel, Field

from encord_active.lib.metrics.utils import IdentifierSchema, MetricData
from encord_active.lib.model_predictions.writer import MainPredictionType


class OntologyObjectJSON(TypedDict):
    featureHash: str
    name: str
    color: str


@dataclass
class MetricEntryPoint:
    metric_path: Path
    is_predictions: bool
    filter_fn: Optional[Callable[[MetricData], Any]] = None


class ClassificationLabelSchema(IdentifierSchema):
    url: Series[str] = pa.Field()
    img_id: Series[padt.Int64] = pa.Field(coerce=True)
    class_id: Series[padt.Int64] = pa.Field(coerce=True)


class ClassificationPredictionSchema(ClassificationLabelSchema):
    confidence: Series[padt.Float64] = pa.Field(coerce=True)


class ClassificationPredictionMatchSchema(ClassificationPredictionSchema):
    is_true_positive: Series[float] = pa.Field()
    gt_class_id: Series[padt.Int64] = pa.Field(coerce=True)


class ClassificationPredictionMatchSchemaWithClassNames(ClassificationPredictionMatchSchema):
    class_name: Series[str] = pa.Field()
    gt_class_name: Series[str] = pa.Field()


class ClassificationLabelMatchSchema(ClassificationLabelSchema):
    is_false_negative: Series[bool] = pa.Field()


class LabelSchema(IdentifierSchema):
    url: Series[str] = pa.Field()
    img_id: Series[padt.Int64] = pa.Field(coerce=True)
    class_id: Series[padt.Int64] = pa.Field(coerce=True)
    x1: Series[padt.Float64] = pa.Field(nullable=True, coerce=True)
    y1: Series[padt.Float64] = pa.Field(nullable=True, coerce=True)
    x2: Series[padt.Float64] = pa.Field(nullable=True, coerce=True)
    y2: Series[padt.Float64] = pa.Field(nullable=True, coerce=True)
    rle: Series[object] = pa.Field(nullable=True, coerce=True)


class PredictionSchema(LabelSchema):
    confidence: Series[padt.Float64] = pa.Field(coerce=True)
    iou: Series[padt.Float64] = pa.Field(coerce=True)


class PredictionMatchSchema(PredictionSchema):
    is_true_positive: Series[float] = pa.Field()
    false_positive_reason: Series[str] = pa.Field()


class LabelMatchSchema(LabelSchema):
    is_false_negative: Series[bool] = pa.Field()


class ClassificationOutcomeType(str, Enum):
    CORRECT_CLASSIFICATIONS = "Correct Classifications"
    MISCLASSIFICATIONS = "Misclassifications"


class ObjectDetectionOutcomeType(str, Enum):
    TRUE_POSITIVES = "True Positive"
    FALSE_POSITIVES = "False Positive"
    FALSE_NEGATIVES = "False Negative"


class PredictionsFilters(BaseModel):
    type: MainPredictionType
    outcome: Optional[Union[ClassificationOutcomeType, ObjectDetectionOutcomeType]] = None
    iou_threshold: Optional[Annotated[float, Field(ge=0, le=1)]] = None
    ignore_frames_without_predictions: bool = False

    class Config:
        frozen = True
