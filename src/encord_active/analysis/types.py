from __future__ import annotations

from typing import Annotated, NamedTuple

import torch
from pydantic import BaseModel, Field
from torch import Tensor


class MetricKey(NamedTuple):
    project_hash: str
    label_hash: str
    data_unit_hash: str
    frame: int | None
    annotation: str | None

    def with_annotation(self, annotation: str):
        return MetricKey(self.project_hash, self.label_hash, self.data_unit_hash, self.frame, annotation)


class Point(NamedTuple):
    x: int
    y: int


class OntologyIdentifier(BaseModel):
    feature_hash: str


class ObjectMetadata(OntologyIdentifier):
    key: MetricKey
    feature_hash: str
    points: PointTensor
    mask: MaskTensor

    def torch_points(self) -> PointTensor:
        return torch.tensor(self.points, dtype=torch.float)


class ClassificationMetadata(OntologyIdentifier):
    key: MetricKey
    feature_hash: str


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


class NearestNeighbors(NamedTuple):
    feature_hashes: list[str]
    annotation_hashes: list[str]
    similarities: list[float]


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
MetricResult = Annotated[Tensor | float | int | str | None, None, ""]
"""
One floating point or integer value. 
`None` means not applicable.
"""
MetricDependencies = dict[str, MetricResult | EmbeddingTensor | NearestNeighbors]
AnnotationsMetricDependencies = dict[MetricKey, MetricDependencies] | None  # TODO this needs to be fixed
"""
Results for all objects in a frame
`None` means not applicable.
"""
