from typing import Callable, Union

import cv2
import numpy as np

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.metrics.metric import (
    AnnotationType,
    DataType,
    MetricType,
    SimpleMetric,
)
from encord_active.lib.metrics.writer import CSVMetricWriter


def execute_helper(
        image,
        writer: CSVMetricWriter,
        rank_fn: Callable,
        color_space: int = cv2.COLOR_BGR2RGB,
):
    try:
        image = cv2.cvtColor(image, color_space)
        writer.write(rank_fn(image))
    except Exception:
        return


class ContrastMetric(SimpleMetric):
    def __init__(self):
        super().__init__(
            title="Contrast",
            short_description="Ranks images by their contrast.",
            long_description=r"""Ranks images by their contrast.

Contrast is computed as the standard deviation of the pixel values.
""",
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.NONE
        )

    @staticmethod
    def rank_by_contrast(image):
        return image.std() / 255

    def execute(self, image, writer: CSVMetricWriter):
        execute_helper(image, writer, self.rank_by_contrast)


class Wrapper:  # we can't have a non-default-constructible Metric implementation at module level
    class ColorMetric(SimpleMetric):
        def __init__(
                self,
                color_name: str,
                hue_filters: Union[list, list[list]],
                saturation_filters=[50, 255],
                value_filters=[20, 255],
        ):

            super().__init__(
                title=f"{color_name} Values".title(),
                short_description=f"Ranks images by how {color_name.lower()} the average value of the image is.",
                long_description=f"""Ranks images by how {color_name.lower()} the average value of the
                    image is.""",
                metric_type=MetricType.HEURISTIC,
                data_type=DataType.IMAGE,
                annotation_type=AnnotationType.NONE
            )

            self.color_name = color_name
            self.hue_filters = hue_filters
            self.saturation_filters = saturation_filters
            self.value_filters = value_filters

            hsv_test = all(0 <= item <= 179 for item in self.__flatten_nested_lists(hue_filters))
            if not hsv_test:
                raise ValueError("Hue parameter should be in [0, 179]")

            saturation_test = all(0 <= item <= 255 for item in saturation_filters)
            if not saturation_test:
                raise ValueError("Saturation parameter should be in [0, 255]")

            value_test = all(0 <= item <= 255 for item in value_filters)
            if not value_test:
                raise ValueError("Value parameter should be in [0, 255]")

        def __flatten_nested_lists(self, nested_list):
            out = []

            for item in nested_list:
                if isinstance(item, list):
                    out.extend(self.__flatten_nested_lists(item))

                else:
                    out.append(item)
            return out

        def rank_by_hsv_filtering(self, image):
            if self.color_name.lower() != "red":
                mask = cv2.inRange(
                    image,
                    np.array([self.hue_filters[0], self.saturation_filters[0], self.value_filters[0]]),
                    np.array([self.hue_filters[1], self.saturation_filters[1], self.value_filters[1]]),
                )
                ratio = np.sum(mask > 0) / (image.shape[0] * image.shape[1])

            else:
                lower_spectrum = [
                    np.array([self.hue_filters[0][0], self.saturation_filters[0], self.value_filters[0]]),
                    np.array([self.hue_filters[0][1], self.saturation_filters[1], self.value_filters[1]]),
                ]
                upper_spectrum = [
                    np.array([self.hue_filters[1][0], self.saturation_filters[0], self.value_filters[0]]),
                    np.array([self.hue_filters[1][1], self.saturation_filters[1], self.value_filters[1]]),
                ]

                lower_mask = cv2.inRange(image, lower_spectrum[0], lower_spectrum[1])
                upper_mask = cv2.inRange(image, upper_spectrum[0], upper_spectrum[1])
                mask = lower_mask + upper_mask
                ratio = np.sum(mask > 0) / (image.shape[0] * image.shape[1])

            return ratio

        def execute(self, image, writer: CSVMetricWriter):
            execute_helper(image, writer, self.rank_by_hsv_filtering, color_space=cv2.COLOR_BGR2HSV)


# Inputs for new color algorithm
class RedMetric(Wrapper.ColorMetric):
    def __init__(self):
        super(RedMetric, self).__init__(
            "Red",
            hue_filters=[[0, 10], [170, 179]],
        )


class GreenMetric(Wrapper.ColorMetric):
    def __init__(self):
        super(GreenMetric, self).__init__("Green", hue_filters=[35, 75])


class BlueMetric(Wrapper.ColorMetric):
    def __init__(self):
        super(BlueMetric, self).__init__("Blue", hue_filters=[90, 130])


class BrightnessMetric(SimpleMetric):
    def __init__(self):
        super().__init__(
            title="Brightness",
            short_description="Ranks images by their brightness.",
            long_description=r"""Ranks images their brightness.

Brightness is computed as the average (normalized) pixel value across each image.
""",
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.NONE
        )

    @staticmethod
    def rank_by_brightness(image):
        return image.mean() / 255

    def execute(self, image, writer: CSVMetricWriter):
        execute_helper(image, writer, self.rank_by_brightness)


class SharpnessMetric(SimpleMetric):
    def __init__(self):
        super().__init__(
            title="Sharpness",
            short_description="Ranks images by their sharpness.",
            long_description=r"""Ranks images by their sharpness.

Sharpness is computed by applying a Laplacian filter to each image and computing the
variance of the output. In short, the score computes "the amount of edges" in each
image.

```python
score = cv2.Laplacian(image, cv2.CV_64F).var()
```
""",
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.NONE
        )

    @staticmethod
    def rank_by_sharpness(image):
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def execute(self, image, writer: CSVMetricWriter):
        execute_helper(image, writer, self.rank_by_sharpness)


class BlurMetric(SimpleMetric):
    def __init__(self):
        super().__init__(
            title="Blur",
            short_description="Ranks images by their blurriness.",
            long_description=r"""Ranks images by their blurriness.

Blurriness is computed by applying a Laplacian filter to each image and computing the
variance of the output. In short, the score computes "the amount of edges" in each
image. Note that this is $1 - \text{sharpness}$.

```python
score = 1 - cv2.Laplacian(image, cv2.CV_64F).var()
```
""",
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.NONE
        )

    @staticmethod
    def rank_by_blur(image):
        return 1 - cv2.Laplacian(image, cv2.CV_64F).var()

    def execute(self, image, writer: CSVMetricWriter):
        execute_helper(image, writer, self.rank_by_blur)


class AspectRatioMetric(SimpleMetric):
    def __init__(self):
        super().__init__(
            title="Aspect Ratio",
            short_description="Ranks images by their aspect ratio (width/height).",
            long_description=r"""Ranks images by their aspect ratio (width/height).

Aspect ratio is computed as the ratio of image width to image height.
    """,
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.NONE
        )
    @staticmethod
    def rank_by_aspect_ratio(image):
        return image.shape[1] / image.shape[0]

    def execute(self, image, writer: CSVMetricWriter):
        execute_helper(image, writer, self.rank_by_aspect_ratio)


class AreaMetric(SimpleMetric):
    def __init__(self):
        super().__init__(
            title="Area",
            short_description="Ranks images by their area (width*height).",
            long_description=r"""Ranks images by their area (width*height).

Area is computed as the product of image width and image height.
        """,
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.NONE
        )

    @staticmethod
    def rank_by_area(image):
        return image.shape[0] * image.shape[1]

    def execute(self, image, writer: CSVMetricWriter):
        execute_helper(image, writer, self.rank_by_area)
