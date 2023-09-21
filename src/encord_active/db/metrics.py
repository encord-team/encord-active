from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Iterable, List, Tuple, Type, TypeVar, Union

from pydantic import BaseModel
from sqlmodel import CheckConstraint, ForeignKeyConstraint, Index

"""
Number of custom metrics supported for each table
"""
CUSTOM_METRIC_COUNT: int = 4


class MetricType(Enum):
    """Type of the metric"""

    NORMAL = "normal"
    """ Normal float between 0 and 1"""
    UINT = "uint"
    """ Positive integer >= 0 """
    UFLOAT = "ufloat"
    """ Positive float >= 0 """


@dataclass
class MetricDefinition:
    title: str
    short_desc: str
    long_desc: str
    type: MetricType


DataAnnotationSharedMetrics: Dict[str, MetricDefinition] = {
    "metric_width": MetricDefinition(
        title="Width",
        short_desc="Width in pixels",
        long_desc="",
        type=MetricType.UINT,
    ),
    "metric_height": MetricDefinition(
        title="Height",
        short_desc="Height in pixels",
        long_desc="",
        type=MetricType.UINT,
    ),
    "metric_area": MetricDefinition(
        title="Area",
        short_desc="Area in pixels",
        long_desc="",
        type=MetricType.UINT,
    ),
    "metric_aspect_ratio": MetricDefinition(
        title="Aspect Ratio",
        short_desc="Aspect ratio = (Width / Height)",
        long_desc="",
        type=MetricType.UFLOAT,
    ),
    "metric_brightness": MetricDefinition(
        title="Brightness",
        short_desc="Brightness of the image data",
        long_desc="",
        type=MetricType.NORMAL,
    ),
    "metric_contrast": MetricDefinition(
        title="Contrast",
        short_desc="Contrast of the image data",
        long_desc="",
        type=MetricType.NORMAL,
    ),
    "metric_sharpness": MetricDefinition(
        title="Sharpness",
        short_desc="",
        long_desc="",
        type=MetricType.NORMAL,
    ),
    "metric_red": MetricDefinition(
        title="Red",
        short_desc="Redness of the image data",
        long_desc="",
        type=MetricType.NORMAL,
    ),
    "metric_green": MetricDefinition(
        title="Green",
        short_desc="Greenness of the image data",
        long_desc="",
        type=MetricType.NORMAL,
    ),
    "metric_blue": MetricDefinition(
        title="Blue",
        short_desc="Blueness of the image data",
        long_desc="",
        type=MetricType.NORMAL,
    ),
    "metric_random": MetricDefinition(
        title="Random",
        short_desc="Random value",
        long_desc="",
        type=MetricType.NORMAL,
    ),
}

_DataOnlyMetrics: Dict[str, MetricDefinition] = {
    "metric_object_count": MetricDefinition(
        title="Object Count",
        short_desc="",
        long_desc="",
        type=MetricType.UINT,
    ),
    "metric_object_density": MetricDefinition(
        title="Object Density",
        short_desc="",
        long_desc="",
        type=MetricType.NORMAL,
    ),
    "metric_image_difficulty": MetricDefinition(
        title="Image Difficulty",
        short_desc="",
        long_desc="",
        type=MetricType.NORMAL,
    ),
    "metric_image_uniqueness": MetricDefinition(
        title="Image Uniqueness",
        short_desc="",
        long_desc="",
        type=MetricType.NORMAL,
    ),
}

AnnotationOnlyMetrics: Dict[str, MetricDefinition] = {
    "metric_area_relative": MetricDefinition(
        title="Relative Area",
        short_desc="Relative area compared to the data source",
        long_desc="",
        type=MetricType.NORMAL,
    ),
    "metric_max_iou": MetricDefinition(
        title="Annotation Duplicates",
        short_desc="",
        long_desc="The maximum iou collision with another annotation",
        type=MetricType.NORMAL,
    ),
    "metric_border_relative": MetricDefinition(
        title="Border Closeness",
        short_desc="",
        long_desc="",
        type=MetricType.NORMAL,
    ),
    "metric_polygon_similarity": MetricDefinition(
        title="Track Shape Outlier",
        short_desc="",
        long_desc="",
        type=MetricType.NORMAL,
    ),
    "metric_missing_or_broken_track": MetricDefinition(
        title="Broken tracks",
        short_desc="",
        long_desc="",
        type=MetricType.NORMAL,
    ),
    "metric_annotation_quality": MetricDefinition(
        title="Annotation Quality",
        short_desc="",
        long_desc="",
        type=MetricType.NORMAL,
    ),
    "metric_inconsistent_class": MetricDefinition(
        title="Inconsistent Classification And Track",
        short_desc="",
        long_desc="",
        type=MetricType.NORMAL,
    ),
    "metric_shape_outlier": MetricDefinition(
        title="Shape Outlier",
        short_desc="",
        long_desc="",
        type=MetricType.NORMAL,
    ),
    "metric_confidence": MetricDefinition(
        title="Confidence",
        short_desc="Annotation or prediction confidence for non-manual labelling",
        long_desc="",
        type=MetricType.NORMAL,
    ),
}


DataMetrics = DataAnnotationSharedMetrics | _DataOnlyMetrics
AnnotationMetrics = DataAnnotationSharedMetrics | AnnotationOnlyMetrics

TSQLClass = TypeVar("TSQLClass", bound=BaseModel)


def assert_cls_metrics_match(
    metrics: Dict[str, MetricDefinition], custom: int = 0
) -> Callable[[Type[TSQLClass]], Type[TSQLClass]]:
    def wrapper(cls: Type[TSQLClass]) -> Type[TSQLClass]:
        fields: Dict[str, None] = cls.__fields__  # type: ignore
        # No missing metrics
        for metric_name, metric_definition in metrics.items():
            if metric_name.startswith("metric_custom"):
                raise ValueError(f"Class: {cls.__name__} has explicit custom metric: {metric_name}")  # type: ignore
            field = fields.get(metric_name, None)
            if field is None:
                raise ValueError(f"Class: {cls.__name__} is missing metric field: {metric_name}")  # type: ignore

        # No extra metrics
        for field in fields:
            if field.startswith("metric_custom"):
                custom_id = int(field[len("metric_custom") :])
                if custom_id >= custom or custom_id < 0:
                    raise ValueError(f"Class: {cls.__name__} has wrong custom metric count: {field}")  # type: ignore
            elif field.startswith("metric_"):
                metric = metrics.get(field, None)
                if metric is None:
                    raise ValueError(f"Class: {cls.__name__} has extra metric: {field}")  # type: ignore
        return cls

    return wrapper


def define_metric_indices(
    metric_prefix: str,
    metrics: Dict[str, MetricDefinition],
    extra: Iterable[Union[Index, CheckConstraint, ForeignKeyConstraint]],
    add_constraints: bool = True,
    grouping: str = "project_hash",
) -> Tuple[Union[Index, CheckConstraint, ForeignKeyConstraint], ...]:
    values: List[Union[Index, CheckConstraint, ForeignKeyConstraint]] = [
        Index(f"ix_{metric_prefix}_ph_{metric_name.replace('metric_', 'mtc_')}", grouping, metric_name)
        for metric_name, metric_metadata in metrics.items()
    ]
    if add_constraints:
        for metric_name, metric_metadata in metrics.items():
            name = f"{metric_prefix}_{metric_name.replace('metric_', 'mtc_')}"
            if metric_metadata.type == MetricType.NORMAL:
                values.append(CheckConstraint(f"{metric_name} BETWEEN 0.0 AND 1.0", name=name))
            elif metric_metadata.type == MetricType.UINT:
                values.append(CheckConstraint(f"{metric_name} >= 0", name=name))
            elif metric_metadata.type == MetricType.UFLOAT:
                values.append(CheckConstraint(f"{metric_name} >= 0.0", name=name))
    values = values + list(extra)
    return tuple(values)
