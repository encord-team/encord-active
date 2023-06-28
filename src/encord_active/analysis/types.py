from typing import Annotated, NamedTuple

import torch
from pydantic import BaseModel, Field
from torch import Tensor

MaskTensor = Annotated[Tensor, torch.bool, "height width"]
"""
Boolean masks of shape `[height, width]` where True means 'here is the object'
"""
ImageTensor = Annotated[Tensor, torch.uint8, "3 height width"]
"""
Image Tensors are raw RGB Values (uint8) of shape `[3, height, width]`
"""
HSVTensor = Annotated[Tensor, torch.float, "3 height width"]
"""
An HSVTensor (float) of shape `[3, h, w]` will have a hue, saturation, and a value channel. 
The ranges are:
    hue: [0, 2pi]
    saturation: [0, 1]
    value: [0, 1]
"""
LaplacianTensor = Annotated[Tensor, torch.float, "height width"]
"""
A float tensor of shape [height width].
"""
EmbeddingTensor = Annotated[Tensor, torch.float, "d"]
"""
A one-dimensional float vector.
"""
PointTensor = Annotated[Tensor, torch.float, "num_points 2"]
"""
Tensor of Points (float) within the size of an image (unnormalized)  # TODO verify this
"""
MetricResult = Annotated[Tensor | float | int, None, ""]
"""
One floating point or integer value.
"""


class Point(NamedTuple):
    x: int
    y: int


class OntologyIdentifier(BaseModel):
    feature_hash: str


class ObjectMetadata(OntologyIdentifier):
    object_hash: str
    points: list[Point]
    mask: MaskTensor

    def torch_points(self) -> PointTensor:
        return torch.tensor(self.points, dtype=torch.float)


class ClassificationMetadata(OntologyIdentifier):
    classification_hash: str


class ObjectAndClassifications(BaseModel):
    objects: list[ObjectMetadata] = Field(default_factory=list)
    classifications: list[ClassificationMetadata] = Field(default_factory=list)


class LabelMetadata(BaseModel):
    id: int
    label_hash: str
    user_hash: str
    data_hash: str
    project_hash: str
    label_status: int
    labels: ObjectAndClassifications
    # labels: Dict[str, ObjectAndClassifications]  -- Not clear what the key would be here?
