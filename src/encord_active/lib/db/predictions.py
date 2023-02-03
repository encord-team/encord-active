import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Dict, List, NamedTuple, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, validator

from encord_active.lib.metrics.execute import run_all_prediction_metrics
from encord_active.lib.model_predictions.iterator import PredictionIterator
from encord_active.lib.model_predictions.writer import PredictionWriter
from encord_active.lib.project import Project

RelativeFloat = Annotated[float, Field(ge=0, le=1)]


class Format(str, Enum):
    BOUNDING_BOX = "bounding-box"
    MASK = "mask"
    POLYGON = "polygon"


class BoundingBox(BaseModel):
    x: RelativeFloat
    y: RelativeFloat
    h: RelativeFloat
    w: RelativeFloat


class Point(NamedTuple):
    x: RelativeFloat
    y: RelativeFloat


class Prediction(BaseModel):
    data_hash: str
    frame: Optional[int] = 0
    class_id: str
    confidence: float
    format: Format
    data: Union[BoundingBox, np.ndarray]
    track_id: Optional[Union[str, int]] = None

    @validator("data", pre=True)
    def transform(v):  # pylint: disable=no-self-argument
        if isinstance(v, str):
            v = json.loads(v)
        if isinstance(v, list):
            v = np.array(v)

        return v

    @validator("data")
    def format_and_data_valid(cls, v, values):  # pylint: disable=no-self-argument
        format = values.get("format")
        if format == Format.BOUNDING_BOX:
            assert v.x + v.w <= 1 and v.y + v.h <= 1, f"{v} is not a valid relative bounding box."
        elif format == Format.MASK:
            uni = set(np.unique(v))
            assert (uni - {0, 1}) == set()
        elif format == Format.POLYGON:
            if isinstance(v, np.ndarray):
                assert np.all(np.logical_and(v >= 0.0, v <= 1.0)), "`data` contains an invalid point."

        else:
            raise ValueError("Invalid format")

        return v

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {np.ndarray: lambda v: json.dumps(v.tolist())}


def import_predictions(project: Project, data_dir: Path, predictions: List[Prediction]):
    with PredictionWriter(project) as writer:
        for pred in predictions:
            data: Dict[str, Any] = {}

            if isinstance(pred.data, np.ndarray):
                data["polygon"] = pred.data
            elif isinstance(pred.data, BoundingBox):
                data["bbox"] = pred.data.dict()
            else:
                raise Exception("Data format not supported.")

            writer.add_prediction(
                data_hash=pred.data_hash,
                class_uid=pred.class_id,
                confidence_score=pred.confidence,
                frame=pred.frame,
                **data,
            )

    run_all_prediction_metrics(data_dir=data_dir, iterator_cls=PredictionIterator, use_cache_only=True)
