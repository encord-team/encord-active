from typing import Optional, Union

import numpy as np
from loguru import logger

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.embeddings.hu_moments import get_hu_embeddings
from encord_active.lib.metrics.metric import (
    AnnotationType,
    DataType,
    Metric,
    MetricType,
)
from encord_active.lib.metrics.writer import CSVMetricWriter

logger = logger.opt(colors=True)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class MomentDecayingAverage:
    def __init__(self, decay: Union[float, int] = 0.5):
        decay = max(min(1, decay), 0)  # force 0 <= decay <= 1
        self.decay = decay
        self.vector: Optional[np.ndarray] = None

    def compare(self, vector: np.ndarray):
        if self.vector is None:
            self.vector = vector
            return 1

        val = cosine_similarity(self.vector, vector)
        self.vector = self.decay * self.vector + (1 - self.decay) * vector
        return val


class MomentStore:
    def __init__(self):
        self.moment_averages: dict[str, MomentDecayingAverage] = {}

    def score(self, key: str, moments: np.ndarray):
        decaying_average = self.moment_averages.setdefault(key, MomentDecayingAverage())
        return decaying_average.compare(moments)


class HuMomentsTemporalMetric(Metric):
    def __init__(self):
        super().__init__(
            title="Polygon Shape Similarity",
            short_description="Ranks objects by how similar they are to their instances in previous frames.",
            long_description=r"""Ranks objects by how similar they are to their instances in previous frames
based on [Hu moments](https://en.wikipedia.org/wiki/Image_moment). The more an object's shape changes,
the lower its score will be.""",
            metric_type=MetricType.GEOMETRIC,
            data_type=DataType.SEQUENCE,
            annotation_type=[AnnotationType.OBJECT.POLYGON],
        )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        valid_annotation_types = {annotation_type.value for annotation_type in self.metadata.annotation_type}
        found_any = False

        moment_store = MomentStore()
        hu_moments_df = get_hu_embeddings(iterator)
        hu_moments_identifiers = set(hu_moments_df["identifier"])

        for data_unit, img_pth in iterator.iterate(desc="Computing moment similarity"):
            for obj in data_unit["labels"].get("objects", []):
                if obj["shape"] not in valid_annotation_types:
                    continue

                key = writer.get_identifier(obj)
                if key not in hu_moments_identifiers:  # check if identifier was discarded by get_hu_embeddings
                    continue

                moments = np.array(eval(hu_moments_df.loc[hu_moments_df["identifier"] == key, "embedding"].values[0]))
                score = moment_store.score(obj["objectHash"], moments)
                writer.write(score, obj)
                found_any = True

        if not found_any:
            logger.info(
                f"<yellow>[Skipping]</yellow> No object labels of types {{{', '.join(valid_annotation_types)}}}."
            )
