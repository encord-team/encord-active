import json
from enum import Enum
from typing import Annotated, NamedTuple, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, root_validator, validator

RelativeFloat = Annotated[float, Field(ge=0, le=1)]
DegreeFloat = Annotated[float, Field(ge=0, le=360)]


class Format(str, Enum):
    BOUNDING_BOX = "bounding-box"
    MASK = "mask"
    POLYGON = "polygon"


class BoundingBox(BaseModel):
    x: RelativeFloat
    y: RelativeFloat
    h: RelativeFloat
    w: RelativeFloat
    theta: DegreeFloat = 0.0


class Point(NamedTuple):
    x: RelativeFloat
    y: RelativeFloat


class ObjectDetection(BaseModel):
    format: Format
    data: Union[BoundingBox, np.ndarray]
    feature_hash: str
    track_id: Optional[Union[str, int]] = None

    @validator("data", pre=True, allow_reuse=True)
    def transform(v):  # pylint: disable=no-self-argument
        if isinstance(v, str):
            v = json.loads(v)
        if isinstance(v, list):
            v = np.array(v)

        return v

    @validator("data", allow_reuse=True)
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


class FrameClassification(BaseModel):
    feature_hash: str
    attribute_hash: str
    option_hash: str

    class Config:
        frozen = True


class Prediction(BaseModel):
    data_hash: str
    frame: Optional[int] = 0
    confidence: float
    object: Optional[ObjectDetection] = None
    classification: Optional[FrameClassification] = None

    @root_validator(pre=True, allow_reuse=True)
    def one_of_object_classification(cls, values):  # pylint: disable=no-self-argument
        object = values.get("object")
        classification = values.get("classification")
        exactly_one_of = (object and not classification) or (classification and not object)
        assert exactly_one_of, "Prediction must have exactly one of `object` or `classification"
        return values
