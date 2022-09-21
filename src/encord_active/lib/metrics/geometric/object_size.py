from loguru import logger
from shapely.ops import unary_union

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.metric import AnnotationType, DataType, Metric, MetricType
from encord_active.lib.common.utils import (
    get_bbox_from_encord_label_object,
    get_du_size,
    get_polygon,
)
from encord_active.lib.common.writer import CSVMetricWriter

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
    TITLE = "Object Area - Relative"
    SHORT_DESCRIPTION = "Computes object area as a percentage of total image area"
    LONG_DESCRIPTION = r"""Computes object area as a percentage of total image area."""
    SCORE_NORMALIZATION = True
    METRIC_TYPE = MetricType.GEOMETRIC
    DATA_TYPE = DataType.IMAGE
    ANNOTATION_TYPE = [
        AnnotationType.OBJECT.BOUNDING_BOX,
        AnnotationType.OBJECT.POLYGON,
    ]

    def test(self, iterator: Iterator, writer: CSVMetricWriter):
        valid_annotation_types = {annotation_type.value for annotation_type in self.ANNOTATION_TYPE}
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
    TITLE = "Frame object density"
    SHORT_DESCRIPTION = "Computes the percentage of image area that's occupied by objects"
    LONG_DESCRIPTION = r"""Computes the percentage of image area that's occupied by objects."""
    SCORE_NORMALIZATION = True
    METRIC_TYPE = MetricType.GEOMETRIC
    DATA_TYPE = DataType.IMAGE
    ANNOTATION_TYPE = [
        AnnotationType.OBJECT.BOUNDING_BOX,
        AnnotationType.OBJECT.POLYGON,
    ]

    def test(self, iterator: Iterator, writer: CSVMetricWriter):
        valid_annotation_types = {annotation_type.value for annotation_type in self.ANNOTATION_TYPE}
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
    TITLE = "Object Area - Absolute"
    SHORT_DESCRIPTION = "Computes object area in amount of pixels"
    LONG_DESCRIPTION = r"""Computes object area in amount of pixels."""
    METRIC_TYPE = MetricType.GEOMETRIC
    DATA_TYPE = DataType.IMAGE
    ANNOTATION_TYPE = [
        AnnotationType.OBJECT.BOUNDING_BOX,
        AnnotationType.OBJECT.POLYGON,
    ]

    def test(self, iterator: Iterator, writer: CSVMetricWriter):
        valid_annotation_types = {annotation_type.value for annotation_type in self.ANNOTATION_TYPE}
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
    TITLE = "Object Aspect Ratio"
    SHORT_DESCRIPTION = "Computes aspect ratios of objects"
    LONG_DESCRIPTION = r"""Computes aspect ratios ($width/height$) of objects."""
    METRIC_TYPE = MetricType.GEOMETRIC
    DATA_TYPE = DataType.IMAGE
    ANNOTATION_TYPE = [
        AnnotationType.OBJECT.BOUNDING_BOX,
        AnnotationType.OBJECT.POLYGON,
    ]

    def test(self, iterator: Iterator, writer: CSVMetricWriter):
        valid_annotation_types = {annotation_type.value for annotation_type in self.ANNOTATION_TYPE}
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
