from loguru import logger
from shapely.ops import unary_union

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.utils import (
    get_bbox_from_encord_label_object,
    get_du_size,
    get_polygon,
)
from encord_active.lib.metrics.metric import (
    AnnotationType,
    DataType,
    Metric,
    MetricType,
)
from encord_active.lib.metrics.writer import CSVMetricWriter

logger = logger.opt(colors=True)


def get_area(obj: dict) -> float:
    if obj["shape"] in {"bounding_box", "polygon"}:
        polygon = get_polygon(obj)
        area = 0.0 if polygon is None else polygon.area
    else:
        logger.warning(f"Unknown shape {obj['shape']} in get_area function")
        area = 0.0
    return area


class RelativeObjectAreaMetric(Metric):
    def __init__(self):
        super().__init__(
            title="Object Area - Relative",
            short_description="Computes object area as a percentage of total image area.",
            long_description=r"""Computes object area as a percentage of total image area.""",
            metric_type=MetricType.GEOMETRIC,
            data_type=DataType.IMAGE,
            annotation_type=[
                AnnotationType.OBJECT.BOUNDING_BOX,
                AnnotationType.OBJECT.POLYGON,
            ],
        )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        valid_annotation_types = {annotation_type.value for annotation_type in self.metadata.annotation_type}
        found_any = False

        for data_unit, _ in iterator.iterate(desc="Computing object area"):
            for obj in data_unit["labels"].get("objects", []):
                if obj["shape"] not in valid_annotation_types:
                    continue
                obj_area = get_area(obj)
                writer.write(100 * obj_area, obj)
                found_any = True

        if not found_any:
            logger.info(
                f"<yellow>[Skipping]</yellow> No object labels of types {{{', '.join(valid_annotation_types)}}}."
            )


class OccupiedTotalAreaMetric(Metric):
    def __init__(self):
        super().__init__(
            title="Frame object density",
            short_description="Computes the percentage of image area that's occupied by objects.",
            long_description=r"""Computes the percentage of image area that's occupied by objects.""",
            metric_type=MetricType.GEOMETRIC,
            data_type=DataType.IMAGE,
            annotation_type=[
                AnnotationType.OBJECT.BOUNDING_BOX,
                AnnotationType.OBJECT.POLYGON,
            ],
        )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        if not iterator.project.ontology.objects:
            return

        valid_annotation_types = {annotation_type.value for annotation_type in self.metadata.annotation_type}
        found_any = False

        for data_unit, _ in iterator.iterate(desc="Computing total object area"):
            polygons = []
            for obj in data_unit["labels"].get("objects", []):
                if obj["shape"] not in valid_annotation_types:
                    continue

                poly = get_polygon(obj)
                if not poly:  # avoid corrupted objects without vertices ([]) and polygons with less than 3 vertices
                    continue
                polygons.append(poly)

            occupied_area = unary_union(polygons).area
            writer.write(100 * occupied_area)
            found_any = True

        if not found_any:
            logger.info(
                f"<yellow>[Skipping]</yellow> No object labels of types {{{', '.join(valid_annotation_types)}}}."
            )


class AbsoluteObjectAreaMetric(Metric):
    def __init__(self):
        super().__init__(
            title="Object Area - Absolute",
            short_description="Computes object area in amount of pixels",
            long_description=r"""Computes object area in amount of pixels.""",
            metric_type=MetricType.GEOMETRIC,
            data_type=DataType.IMAGE,
            annotation_type=[
                AnnotationType.OBJECT.BOUNDING_BOX,
                AnnotationType.OBJECT.POLYGON,
            ],
        )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        valid_annotation_types = {annotation_type.value for annotation_type in self.metadata.annotation_type}
        found_any = False

        for data_unit, img_pth in iterator.iterate(desc="Computing pixel area"):
            size = get_du_size(data_unit, img_pth)
            if not size:
                continue
            img_h, img_w = size
            img_area = img_h * img_w

            for obj in data_unit["labels"].get("objects", []):
                if obj["shape"] not in valid_annotation_types:
                    continue
                obj_area = get_area(obj)
                pixel_area = int(img_area * obj_area)
                writer.write(pixel_area, obj)
                found_any = True

        if not found_any:
            logger.info(
                f"<yellow>[Skipping]</yellow> No object labels of types {{{', '.join(valid_annotation_types)}}}."
            )


class ObjectAspectRatioMetric(Metric):
    def __init__(self):
        super().__init__(
            title="Object Aspect Ratio",
            short_description="Computes aspect ratios of objects",
            long_description=r"""Computes aspect ratios ($width/height$) of objects.""",
            metric_type=MetricType.GEOMETRIC,
            data_type=DataType.IMAGE,
            annotation_type=[
                AnnotationType.OBJECT.BOUNDING_BOX,
                AnnotationType.OBJECT.POLYGON,
            ],
        )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        valid_annotation_types = {annotation_type.value for annotation_type in self.metadata.annotation_type}
        found_any = False

        for data_unit, img_pth in iterator.iterate(desc="Computing aspect ratio"):
            size = get_du_size(data_unit, img_pth)
            if not size:
                continue
            img_h, img_w = size

            for obj in data_unit["labels"].get("objects", []):
                if obj["shape"] not in valid_annotation_types:
                    continue

                bbox = get_bbox_from_encord_label_object(obj, w=img_w, h=img_h)
                if bbox is None:
                    continue

                x, y, w, h = bbox
                if h == 0:
                    continue
                ar = w / h

                writer.write(ar, obj)
                found_any = True

        if not found_any:
            logger.info(
                f"<yellow>[Skipping]</yellow> No object labels of types {{{', '.join(valid_annotation_types)}}}."
            )
