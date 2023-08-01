from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Annotated, Dict, NamedTuple, Optional, Tuple, Union

import torch
from torch import Tensor

from encord_active.db.models import AnnotationType

ANNOTATION_TYPE_TO_INT: Dict[AnnotationType, int] = {
    AnnotationType.CLASSIFICATION: 0,
    AnnotationType.BOUNDING_BOX: 1,
    AnnotationType.ROT_BOUNDING_BOX: 2,
    AnnotationType.POINT: 3,
    AnnotationType.POLYLINE: 4,
    AnnotationType.POLYGON: 5,
    AnnotationType.SKELETON: 6,
    AnnotationType.BITMASK: 7,
}


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
    bounding_box: Optional[BoundingBoxTensor]


@dataclass(frozen=True)
class NearestNeighbors:
    metric_deps: list[MetricDependencies]
    metric_keys: list[Union[Tuple[uuid.UUID, int], Tuple[uuid.UUID, int, str]]]
    similarities: list[float]


@dataclass(frozen=True)
class RandomSampling:
    metric_deps: list[MetricDependencies]
    metric_keys: list[Union[Tuple[uuid.UUID, int], Tuple[uuid.UUID, int, str]]]


MaskTensor = Annotated[Tensor, torch.bool, "height width"]
"""
Boolean masks of shape `[height, width]` where True means 'here is the object'
"""

MaskBatchTensor = Annotated[Tensor, torch.bool, "batch height width"]
"""
Batch of boolean masks of shape `[batch, height, width]` where True means 'here is the object'.
"""

BoundingBoxTensor = Annotated[Tensor, torch.int32, "4(x1, y1, x2, y2)"]
"""
Bounding box calculated from the mask.
"""

BoundingBoxBatchTensor = Annotated[Tensor, torch.int32, "batch, 4(x1, y1, x2, y2)"]
"""
Batch of bounding boxes for annotations.
"""

ImageTensor = Annotated[Tensor, torch.uint8, "3 height width"]
"""
Image Tensors are raw RGB Values (uint8) of shape `[3, height, width]`
"""

ImageBatchTensor = Annotated[Tensor, torch.uint8, "batch 3 height width"]
"""
Batch of image tensors that are raw RGB Values (uint8) of shape `[batch, 3, height, width]`
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

EmbeddingBatchTensor = Annotated[Tensor, torch.float, "batch d"]
"""
A batch of one-dimensional float vectors
"""

PointTensor = Annotated[Tensor, torch.float, "num_points 2"]
"""
Tensor of Points (float) within the size of an image (unnormalized)  # TODO verify this
"""

ImageIndexBatchTensor = Annotated[Tensor, torch.int32, "batch"]
"""
Batch of indices for the image associated with a given object.
"""

FeatureHashBatchTensor = Annotated[Tensor, torch.int64, "batch"]
"""
Batch of feature hashes converted into int64 (8 x byte)
"""

MetricResult = Annotated[Union[Tensor, float, int, str, None], None, ""]
"""
One floating point or integer value. 
`None` means not applicable.
"""

MetricBatchResult = Annotated[Optional[Tensor], None, "float[B], int[B], embedding[B]=float[B, E]"]
"""
Batch of metric results.
"""


MetricDependencies = dict[str, Union[MetricResult, EmbeddingTensor, NearestNeighbors, RandomSampling]]
"""
Results for all objects in a frame
`None` means not applicable.
"""

MetricBatchDependencies = dict[str, Tensor]
"""
Results for 
"""
