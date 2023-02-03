import json
from enum import Enum
from typing import Annotated, NamedTuple, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, validator

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


class ObjectDetection(BaseModel):
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


class Prediction(BaseModel):
    data_hash: str
    frame: Optional[int] = 0
    class_id: str
    confidence: float
    object: Optional[ObjectDetection]
