from typing import Dict, Union, Callable, Optional, TypeVar
from enum import Enum
from dataclasses import dataclass
from sqlmodel import column
from pydantic import BaseModel


class MetricType(Enum):
    """ Type of the metric """
    NORMAL = "normal"
    """ Normal float between 0 and 1"""
    UINT = "uint"
    """ Positive integer >= 0 """
    UFLOAT = "ufloat"
    """ Positive float >= 0 """
    RANK = "rank"
    """ Float value where the value is the relative order"""


@dataclass
class MetricVirtualDefinition:
    src: str
    flip_ord: bool
    map: Callable[[Union[float, int, column]], Union[float, int, column]]


@dataclass
class MetricDefinition:
    title: str
    short_desc: str
    long_desc: str
    type: MetricType
    virtual: Optional[MetricVirtualDefinition] = None


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
    #"metric_blur": MetricDefinition(
    #    title="Blur",
    #    short_desc="",
    #    long_desc="",
    #    type=MetricType.NORMAL,
    #    virtual=MetricVirtualDefinition(
    #        src="metric_sharpness",
    #        flip_ord=True,
    #        map=lambda x: 1.0 - x,
    #    )
    #),
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
    )
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
        type=MetricType.RANK,
    ),
    "metric_image_singularity": MetricDefinition(
        title="Image Singularity",
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
}

AnnotationOnlyMetrics: Dict[str, MetricDefinition] = {
    "metric_area_relative": MetricDefinition(
        title="Relative Area",
        short_desc="Relative area compared to the data source",
        long_desc="",
        type=MetricType.UINT,
    ),
    "metric_label_duplicates": MetricDefinition(
        title="Label Duplicates",
        short_desc="",
        long_desc="",
        type=MetricType.NORMAL,
    ),
    "metric_label_border_closeness": MetricDefinition(
        title="Border Closeness",
        short_desc="",
        long_desc="",
        type=MetricType.NORMAL,
    ),
    "metric_label_poly_similarity": MetricDefinition(
        title="Polygon Simularity",
        short_desc="",
        long_desc="",
        type=MetricType.NORMAL,
    ),
    "metric_label_missing_or_broken_tracks": MetricDefinition(
        title="Missing or Broken Tracks",
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
    "metric_label_inconsistent_classification_and_track": MetricDefinition(
        title="Inconsistent Classification And Track",
        short_desc="",
        long_desc="",
        type=MetricType.NORMAL,
    ),
    "metric_label_shape_outlier": MetricDefinition(
        title="Shape Outlier",
        short_desc="",
        long_desc="",
        type=MetricType.NORMAL,
    ),
}


DataMetrics = DataAnnotationSharedMetrics | _DataOnlyMetrics
AnnotationMetrics = DataAnnotationSharedMetrics | AnnotationOnlyMetrics

TSQLClass = TypeVar("TSQLClass", bound=BaseModel)


def assert_cls_metrics_match(metrics: Dict[str, MetricDefinition]) -> Callable[[TSQLClass], TSQLClass]:
    def wrapper(cls: TSQLClass) -> TSQLClass:
        fields = cls.__fields__
        # No missing metrics
        for metric_name, metric_definition in metrics.items():
            if metric_definition.virtual is not None:
                continue
            field = fields.get(metric_name, None)
            if field is None:
                raise ValueError(f"Class: {cls.__name__} is missing metric field: {metric_name}")

        # No extra metrics
        for field in fields:
            if field.startswith("metric_"):
                metric = metrics.get(field, None)
                if metric is None or metric.virtual is not None:
                    raise ValueError(f"Class: {cls.__name__} has extra metric: {field}")
        return cls

    return wrapper
