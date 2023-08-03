from loguru import logger
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.utils import (
    get_bbox_from_encord_label_object,
    get_du_size,
    get_object_coordinates,
    get_polygon,
)
from encord_active.lib.labels.object import BoxShapes, ObjectShape
from encord_active.lib.metrics.metric import Metric
from encord_active.lib.metrics.types import AnnotationType, DataType, MetricType
from encord_active.lib.metrics.writer import CSVMetricWriter

logger = logger.opt(colors=True)


def get_area(obj: dict) -> float:
    if obj["shape"] in {*BoxShapes, ObjectShape.POLYGON}:

        points = get_object_coordinates(obj)
        if points is None or len(points) < 3:
            logger.debug("Less than 3 points")
            return 0.0

        polygon = Polygon(points)
        if polygon.is_simple:
            area = polygon.area
        else:
            tidy_polygon = polygon.buffer(0)
            if isinstance(tidy_polygon, Polygon):
                area = tidy_polygon.area
            elif isinstance(tidy_polygon, MultiPolygon):
                area = 0.0
                for polygon_item in list(polygon.buffer(0)):
                    area += polygon_item.area
            else:
                area = 0.0
                logger.warning(f"Unknown geometry type: {type(tidy_polygon)}")
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
            doc_url="https://docs.encord.com/docs/active-label-quality-metrics#object-area---relative",
            metric_type=MetricType.GEOMETRIC,
            data_type=DataType.IMAGE,
            annotation_type=[
                AnnotationType.OBJECT.BOUNDING_BOX,
                AnnotationType.OBJECT.ROTATABLE_BOUNDING_BOX,
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
            doc_url="https://docs.encord.com/docs/active-label-quality-metrics#frame-object-density",
            metric_type=MetricType.GEOMETRIC,
            data_type=DataType.IMAGE,
            annotation_type=[
                AnnotationType.OBJECT.BOUNDING_BOX,
                AnnotationType.OBJECT.ROTATABLE_BOUNDING_BOX,
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
            doc_url="https://docs.encord.com/docs/active-label-quality-metrics#object-area---absolute",
            metric_type=MetricType.GEOMETRIC,
            data_type=DataType.IMAGE,
            annotation_type=[
                AnnotationType.OBJECT.BOUNDING_BOX,
                AnnotationType.OBJECT.ROTATABLE_BOUNDING_BOX,
                AnnotationType.OBJECT.POLYGON,
            ],
        )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        valid_annotation_types = {annotation_type.value for annotation_type in self.metadata.annotation_type}
        found_any = False

        for data_unit, image in iterator.iterate(desc="Computing pixel area"):
            size = get_du_size(data_unit, image)
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
            doc_url="https://docs.encord.com/docs/active-label-quality-metrics#object-aspect-ratio",
            metric_type=MetricType.GEOMETRIC,
            data_type=DataType.IMAGE,
            annotation_type=[
                AnnotationType.OBJECT.BOUNDING_BOX,
                AnnotationType.OBJECT.ROTATABLE_BOUNDING_BOX,
                AnnotationType.OBJECT.POLYGON,
            ],
        )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        valid_annotation_types = {annotation_type.value for annotation_type in self.metadata.annotation_type}
        found_any = False

        for data_unit, image in iterator.iterate(desc="Computing aspect ratio"):
            size = get_du_size(data_unit, image)
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
