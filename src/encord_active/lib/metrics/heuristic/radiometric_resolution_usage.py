import cv2
import numpy as np

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.metrics.metric import (
    AnnotationType,
    DataType,
    Metric,
    MetricType,
)
from encord_active.lib.metrics.writer import CSVMetricWriter


class RadiometricResolutionUsage(Metric):
    TITLE = "Radiometric Resolution Usage"
    METRIC_TYPE = MetricType.HEURISTIC
    DATA_TYPE = DataType.IMAGE
    ANNOTATION_TYPE = AnnotationType.NONE
    SHORT_DESCRIPTION = "Calculates percentage of radiometric resolution that's actually used in the image."
    LONG_DESCRIPTION = (
        "[Radiometric resolution](https://en.wikipedia.org/wiki/Image_resolution#:~:text=Radiometric%20resolution%20) "
        "determines how finely a system can represent or distinguish differences of intensity. The higher the "
        "radiometric resolution, the better subtle differences of intensity or reflectivity can be represented, at "
        "least in theory.\nThis metric calculates how much (%) of the image's radiometric resolution is actually used."
    )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        for _, img_pth in iterator.iterate(desc="Calculating radiometric resolution usage"):
            if img_pth is None:
                continue
            try:
                image = cv2.imread(img_pth.as_posix())
            except:
                continue
            if image is None:
                continue

            # Calculate the scaled dynamic range of the image
            scaled_dynamic_range = (np.max(image) - np.min(image)) / (2 ** (8 * image.dtype.itemsize) - 1)
            writer.write(100 * scaled_dynamic_range)
