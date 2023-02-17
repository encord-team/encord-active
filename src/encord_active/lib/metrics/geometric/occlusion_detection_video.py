import numpy as np
from loguru import logger
from sklearn import preprocessing
from tqdm.auto import tqdm

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.metrics.metric import (
    AnnotationType,
    DataType,
    Metric,
    MetricType,
)
from encord_active.lib.metrics.writer import CSVMetricWriter

logger = logger.opt(colors=True)


class OcclusionDetectionOnVideo(Metric):
    def __init__(
        self,
        low_threshold: float = 0.3,
        medium_threshold: float = 0.4,
        high_threshold: float = 0.7,
        min_samples: int = 5,
    ):
        super().__init__(
            title="Detect Occlusion in Video",
            short_description="Tracks objects and detect outliers",
            long_description=r"""This metric collects information related to object size and aspect ratio for each track
 and find outliers among them.""",
            metric_type=MetricType.GEOMETRIC,
            data_type=DataType.SEQUENCE,
            annotation_type=[AnnotationType.OBJECT.BOUNDING_BOX, AnnotationType.OBJECT.ROTATABLE_BOUNDING_BOX],
        )
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.min_samples = min_samples
        self.min_max_scaler = preprocessing.MinMaxScaler()

    def get_description_from_occlusion(self, distance: float) -> str:
        if distance > self.high_threshold:
            return "There is a high probability of occlusion"
        elif distance > self.medium_threshold:
            return "There is a moderate probability of occlusion"
        elif distance > self.low_threshold:
            return "There is a low probability of occlusion"
        else:
            return "There is no occlusion"

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        valid_annotation_types = {annotation_type.value for annotation_type in self.metadata.annotation_type}

        videos: dict[str, dict[str, dict]] = {}
        for label_row_hash, label_row in tqdm(iterator.label_rows.items(), desc="Looking for occlusions", leave=False):
            if label_row["data_type"] == "video":
                videos[label_row_hash] = {}

                data_unit = list(label_row["data_units"].values())[0]
                for label in data_unit["labels"]:
                    for obj in data_unit["labels"][label].get("objects", []):
                        if obj["shape"] not in valid_annotation_types:
                            continue

                        object_hash = obj["objectHash"]
                        if object_hash not in videos[label_row_hash].keys():
                            videos[label_row_hash][object_hash] = {"area": [], "aspect_ratio": [], "frame": []}

                        h = obj["boundingBox"]["h"]
                        w = obj["boundingBox"]["w"]
                        if h == 0:
                            continue
                        videos[label_row_hash][object_hash]["area"].append(h * w)
                        videos[label_row_hash][object_hash]["aspect_ratio"].append(w / h)
                        videos[label_row_hash][object_hash]["frame"].append(label)

                for object_hash in videos[label_row_hash].keys():
                    if len(videos[label_row_hash][object_hash]["frame"]) >= self.min_samples:
                        data_points = np.stack(
                            (
                                videos[label_row_hash][object_hash]["area"],
                                videos[label_row_hash][object_hash]["aspect_ratio"],
                            ),
                            axis=1,
                        )
                        data_points_scaled = self.min_max_scaler.fit_transform(data_points)

                        data_points_mean = np.mean(data_points_scaled, axis=0)
                        distances = np.linalg.norm(data_points_scaled - data_points_mean, axis=1)

                        # We want both distant and low area samples, so filter according to that
                        occlusions = np.logical_and(
                            (distances > self.low_threshold), (data_points_scaled[:, 0] < data_points_mean[0])
                        )

                        distances_adjusted = np.full(data_points.shape[0], 0.0)
                        distances_adjusted[occlusions] = distances[occlusions]

                        descriptions = list(map(self.get_description_from_occlusion, distances_adjusted))

                        scores = np.full(data_points.shape[0], 100.0)
                        scores_full = 1 / distances
                        scores[occlusions] = scores_full[occlusions]

                        videos[label_row_hash][object_hash]["scores"] = scores
                        videos[label_row_hash][object_hash]["descriptions"] = descriptions

        if not videos:
            logger.info("<yellow>[Skipping]</yellow> No videos in dataset. ")

        for data_unit, img_pth in iterator.iterate(desc="Storing occlusion index"):
            label_row_hash = iterator.label_hash
            if label_row_hash not in videos.keys():
                continue

            for obj in data_unit["labels"].get("objects", []):
                object_hash = obj["objectHash"]
                if (
                    obj["shape"] not in valid_annotation_types
                    or "scores" not in videos[label_row_hash][object_hash].keys()
                ):
                    continue

                # todo check if speedup is needed in case a lot of same type's objects are in a label row
                score_index = videos[label_row_hash][object_hash]["frame"].index(str(iterator.frame))

                writer.write(
                    videos[label_row_hash][obj["objectHash"]]["scores"][score_index],
                    obj,
                    description=videos[iterator.label_hash][obj["objectHash"]]["descriptions"][score_index],
                )
