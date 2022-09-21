from typing import Callable, Union

import cv2
import numpy as np

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.metric import AnnotationType, DataType, Metric, MetricType
from encord_active.lib.common.writer import CSVMetricWriter


def iterate_with_rank_fn(
    iterator: Iterator,
    writer: CSVMetricWriter,
    rank_fn: Callable,
    name: str,
    color_space: int = cv2.COLOR_BGR2RGB,
):
    for data_unit, img_pth in iterator.iterate(desc=f"Looking for {name}"):
        if img_pth is None:
            continue
        try:
            image = cv2.imread(img_pth.as_posix())
            image = cv2.cvtColor(image, color_space)
        except cv2.error:
            continue
        writer.write(rank_fn(image))


class ContrastMetric(Metric):
    TITLE = "Contrast"
    SHORT_DESCRIPTION = "Ranks images by their contrast."
    LONG_DESCRIPTION = r"""Ranks images by their contrast. 

Contrast is computed as the standard deviation of the pixel values. 
"""
    SCORE_NORMALIZATION = True
    METRIC_TYPE = MetricType.HEURISTIC
    DATA_TYPE = DataType.IMAGE
    ANNOTATION_TYPE = AnnotationType.NONE

    @staticmethod
    def rank_by_contrast(image):
        return image.std() / 255

    def test(self, iterator: Iterator, writer: CSVMetricWriter):
        return iterate_with_rank_fn(iterator, writer, self.rank_by_contrast, self.TITLE)


class Wrapper:  # we can't have a non-default-constructible Metric implementation at module level
    class ColorMetric(Metric):
        SCORE_NORMALIZATION = True
        METRIC_TYPE = MetricType.HEURISTIC
        DATA_TYPE = DataType.IMAGE
        ANNOTATION_TYPE = AnnotationType.NONE

        def __init__(
            self,
            color_name: str,
            hue_filters: Union[list, list[list]],
            saturation_filters=[50, 255],
            value_filters=[20, 255],
        ):
            self.color_name = color_name
            self._title = f"{color_name} Values".title()
            self._short_description = f"Ranks images by how {color_name.lower()} the average value of the image is."
            self._long_description = f"""Ranks images by how {color_name.lower()} the average value of the
                    image is."""
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

        @property
        def TITLE(self) -> str:
            return self._title

        @property
        def SHORT_DESCRIPTION(self) -> str:
            return self._short_description

        @property
        def LONG_DESCRIPTION(self) -> str:
            return self._long_description

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

        def test(self, iterator: Iterator, writer: CSVMetricWriter):
            return iterate_with_rank_fn(
                iterator, writer, self.rank_by_hsv_filtering, self.TITLE, color_space=cv2.COLOR_BGR2HSV
            )


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


class BrightnessMetric(Metric):
    TITLE = "Brightness"
    SHORT_DESCRIPTION = "Ranks images by their brightness."
    LONG_DESCRIPTION = r"""Ranks images their brightness.

Brightness is computed as the average (normalized) pixel value across each image.
"""
    SCORE_NORMALIZATION = True
    METRIC_TYPE = MetricType.HEURISTIC
    DATA_TYPE = DataType.IMAGE
    ANNOTATION_TYPE = AnnotationType.NONE

    @staticmethod
    def rank_by_brightness(image):
        return image.mean() / 255

    def test(self, iterator: Iterator, writer: CSVMetricWriter):
        return iterate_with_rank_fn(iterator, writer, self.rank_by_brightness, self.TITLE)


class SharpnessMetric(Metric):
    TITLE = "Sharpness"
    SHORT_DESCRIPTION = "Ranks images by their sharpness."
    LONG_DESCRIPTION = r"""Ranks images by their sharpness.

Sharpness is computed by applying a Laplacian filter to each image and computing the
variance of the output. In short, the score computes "the amount of edges" in each 
image.

```python
score = cv2.Laplacian(image, cv2.CV_64F).var()
```
"""
    SCORE_NORMALIZATION = True
    METRIC_TYPE = MetricType.HEURISTIC
    DATA_TYPE = DataType.IMAGE
    ANNOTATION_TYPE = AnnotationType.NONE

    @staticmethod
    def rank_by_sharpness(image):
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def test(self, iterator: Iterator, writer: CSVMetricWriter):
        return iterate_with_rank_fn(iterator, writer, self.rank_by_sharpness, self.TITLE)


class BlurMetric(Metric):
    TITLE = "Blur"
    SHORT_DESCRIPTION = "Ranks images by their blurriness."
    LONG_DESCRIPTION = r"""Ranks images by their blurriness.

Blurriness is computed by applying a Laplacian filter to each image and computing the
variance of the output. In short, the score computes "the amount of edges" in each 
image. Note that this is $1 - \text{sharpness}$.

```python
score = 1 - cv2.Laplacian(image, cv2.CV_64F).var()
```
"""
    SCORE_NORMALIZATION = True
    METRIC_TYPE = MetricType.HEURISTIC
    DATA_TYPE = DataType.IMAGE
    ANNOTATION_TYPE = AnnotationType.NONE

    @staticmethod
    def rank_by_blur(image):
        return 1 - cv2.Laplacian(image, cv2.CV_64F).var()

    def test(self, iterator: Iterator, writer: CSVMetricWriter):
        return iterate_with_rank_fn(iterator, writer, self.rank_by_blur, self.TITLE)


class AspectRatioMetric(Metric):
    TITLE = "Aspect Ratio"
    SHORT_DESCRIPTION = "Ranks images by their aspect ratio (width/height)."
    LONG_DESCRIPTION = r"""Ranks images by their aspect ratio (width/height). 

Aspect ratio is computed as the ratio of image width to image height. 
"""
    METRIC_TYPE = MetricType.HEURISTIC
    DATA_TYPE = DataType.IMAGE
    ANNOTATION_TYPE = AnnotationType.NONE

    @staticmethod
    def rank_by_aspect_ratio(image):
        return image.shape[1] / image.shape[0]

    def test(self, iterator: Iterator, writer: CSVMetricWriter):
        return iterate_with_rank_fn(iterator, writer, self.rank_by_aspect_ratio, self.TITLE)


class AreaMetric(Metric):
    TITLE = "Area"
    SHORT_DESCRIPTION = "Ranks images by their area (width*height)."
    LONG_DESCRIPTION = r"""Ranks images by their area (width*height). 

Area is computed as the product of image width and image height. 
"""
    METRIC_TYPE = MetricType.HEURISTIC
    DATA_TYPE = DataType.IMAGE
    ANNOTATION_TYPE = AnnotationType.NONE

    @staticmethod
    def rank_by_area(image):
        return image.shape[0] * image.shape[1]

    def test(self, iterator: Iterator, writer: CSVMetricWriter):
        return iterate_with_rank_fn(iterator, writer, self.rank_by_area, self.TITLE)
