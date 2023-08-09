from typing import Dict, List

import numpy as np
from loguru import logger
from PIL import Image
from tqdm import tqdm

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.metrics.metric import Metric
from encord_active.lib.metrics.types import AnnotationType, DataType, MetricType
from encord_active.lib.metrics.writer import CSVMetricWriter

logger = logger.opt(colors=True)


class VideoColorMetricWrapper:
    class VideoColorMetric(Metric):
        def __init__(
            self,
            color_name: str,
            sample_rate=30,
        ):
            super().__init__(
                title=f"Video {color_name} Density",
                short_description=f"Rank the videos according to their aggregated {color_name} color density",
                long_description=r"""
    """,
                metric_type=MetricType.HEURISTIC,
                data_type=DataType.SEQUENCE,
                annotation_type=AnnotationType.NONE,
            )
            if color_name not in ["Red", "Green", "Blue"]:
                raise ValueError("color_name parameter should be one of ['Red', 'Green', 'Blue']")
            self._color_name = color_name
            self._sample_rate = sample_rate

        def calculate_average_color_density(self, image: Image):
            np_img = np.array(image.convert("RGB"), dtype=np.float32)
            if self._color_name == "Red":
                color = np_img[:, :, 0] - np.maximum(np_img[:, :, 1], np_img[:, :, 2])
            elif self._color_name == "Green":
                color = np_img[:, :, 1] - np.maximum(np_img[:, :, 0], np_img[:, :, 2])
            elif self._color_name == "Blue":
                color = np_img[:, :, 2] - np.maximum(np_img[:, :, 0], np_img[:, :, 1])
            else:
                raise ValueError("color_name parameter should be one of ['Red', 'Green', 'Blue']")
            color = np.clip(color, 0, 255)

            return np.mean(color) / 255

        def execute(self, iterator: Iterator, writer: CSVMetricWriter):
            videos_to_colors: Dict[str, List] = {}
            for label_row_hash, label_row in tqdm(
                iterator.label_rows.items(), desc="Collecting videos...", leave=False
            ):
                if label_row["data_type"] == "video":
                    videos_to_colors[label_row_hash] = []

            for data_unit, image in iterator.iterate(desc="Calculating average colors"):
                if iterator.label_rows[iterator.label_hash]["data_type"] == "video":
                    if iterator.frame % self._sample_rate == 0:
                        videos_to_colors[iterator.label_hash].append(self.calculate_average_color_density(image))

                    if iterator.frame == int(data_unit["data_fps"] * data_unit["data_duration"]) - 1:
                        writer.write(
                            sum(videos_to_colors[iterator.label_hash]) / len(videos_to_colors[iterator.label_hash]),
                            frame=iterator.frame // 2,
                            key=f"{iterator.label_hash}_{data_unit['data_hash']}_{iterator.frame//2}",
                        )


class VideoRedMetric(VideoColorMetricWrapper.VideoColorMetric):
    def __init__(self):
        super(VideoRedMetric, self).__init__("Red")


class VideoGreenMetric(VideoColorMetricWrapper.VideoColorMetric):
    def __init__(self):
        super(VideoGreenMetric, self).__init__("Green")


class VideoBlueMetric(VideoColorMetricWrapper.VideoColorMetric):
    def __init__(self):
        super(VideoBlueMetric, self).__init__("Blue")
