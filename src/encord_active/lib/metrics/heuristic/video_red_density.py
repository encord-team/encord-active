from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from PIL import Image
from tqdm import tqdm

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.metrics.metric import Metric
from encord_active.lib.metrics.types import AnnotationType, DataType, MetricType
from encord_active.lib.metrics.writer import CSVMetricWriter

logger = logger.opt(colors=True)


class VideoRedDensity(Metric):
    def __init__(self, sample_rate=30):
        super().__init__(
            title="Video Red Density",
            short_description="Rank the videos according to their aggregated Red color density",
            long_description=r"""
Rank the videos according to their aggregated Red color density
""",
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.SEQUENCE,
            annotation_type=AnnotationType.NONE,
        )
        self._sample_rate = sample_rate

    def calculate_average_red_density(self, image: Image):
        np_img = np.array(image)
        redness = np_img[:, :, 0] - np.maximum(np_img[:, :, 1], np_img[:, :, 2])
        redness = np.clip(redness, 0, 255)

        return np.mean(redness) / 255

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        videos_to_colors: Dict[str, List] = {}
        for label_row_hash, label_row in tqdm(iterator.label_rows.items(), desc="Collecting videos...", leave=False):
            if label_row["data_type"] == "video":
                videos_to_colors[label_row_hash] = []

        for data_unit, image in iterator.iterate(desc="Calculating average colors"):
            if iterator.label_rows[iterator.label_hash]["data_type"] == "video":
                if iterator.frame % self._sample_rate == 0:
                    videos_to_colors[iterator.label_hash].append(self.calculate_average_red_density(image))

                if iterator.frame == int(data_unit["data_fps"] * data_unit["data_duration"]) - 1:
                    writer.write(
                        sum(videos_to_colors[iterator.label_hash]) / len(videos_to_colors[iterator.label_hash]),
                        frame=iterator.frame // 2,
                        key=f"{iterator.label_hash}_{data_unit['data_hash']}_{iterator.frame//2}",
                    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from encord_active.lib.metrics.execute import execute_metrics

    path = sys.argv[1]
    execute_metrics([VideoRedDensity()], data_dir=Path(path))
