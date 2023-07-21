from __future__ import annotations

import uuid
from typing import Annotated, NamedTuple, Optional, Union, Tuple
from dataclasses import dataclass
import torch
from torch import Tensor

from encord_active.db.models import AnnotationType


class Point(NamedTuple):
    x: int
    y: int


@dataclass(frozen=True)
class AnnotationMetadata:
    feature_hash: str
    annotation_type: AnnotationType
    annotation_email: str
    annotation_manual: bool
    annotation_confidence: float
    points: Optional[PointTensor]
    mask: Optional[MaskTensor]


@dataclass(frozen=True)
class NearestNeighbors:
    metric_deps: list[MetricDependencies]
    metric_keys: list[Union[Tuple[uuid.UUID, int], Tuple[uuid.UUID, int, str]]]
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
MetricResult = Annotated[Union[Tensor, float, int, str, None], None, ""]
"""
One floating point or integer value. 
`None` means not applicable.
"""
MetricDependencies = dict[str, Union[MetricResult, EmbeddingTensor, NearestNeighbors]]
"""
Results for all objects in a frame
`None` means not applicable.
"""
