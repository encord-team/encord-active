from dataclasses import dataclass
from typing import List

import faiss
import numpy as np
from loguru import logger
from sklearn.preprocessing import normalize as sk_normalize

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.utils import (
    fix_duplicate_image_orders_in_knn_graph_all_rows,
)
from encord_active.lib.embeddings.cnn import get_cnn_embeddings
from encord_active.lib.embeddings.utils import LabelEmbedding
from encord_active.lib.metrics.metric import DataType, EmbeddingType, Metric, MetricType
from encord_active.lib.metrics.writer import CSVMetricWriter

logger = logger.opt(colors=True)


@dataclass
class DataUnitInfo:
    score: float
    description: str


class ImageSingularity(Metric):
    def __init__(self, near_duplicate_threshold=0.97):
        super(ImageSingularity, self).__init__(
            title="Image Singularity",
            short_description="Finds duplicate and near-duplicate images",
            long_description=r"""
This metric gives each image a score that shows each image's uniqueness.  
- A score of zero means that the image has duplicates in the dataset; on the other hand, a score close to one represents that image is quite unique. Among the duplicate images, we only give a non-zero score to a single image, and the rest will have a score of zero (for example, if there are five identical images, only four will have a score of zero). This way, these duplicate samples can be easily tagged and removed from the project.    
- Images that are near duplicates of each other will be shown side by side. 
### Possible actions
- **To delete duplicate images:** You can set the quality filter to cover only zero values (that ends up with all the duplicate images), then use bulk tagging (e.g., with a tag like `Duplicate`) to tag all images.
- **To mark duplicate images:** Near duplicate images are shown side by side. Navigate through these images and mark whichever is of interest to you.
""",
            metric_type=MetricType.SEMANTIC,
            data_type=DataType.IMAGE,
            embedding_type=EmbeddingType.CLASSIFICATION,
        )
        self.collections: List[LabelEmbedding] = []
        self.scores: dict[str, DataUnitInfo] = {}
        self.near_duplicate_threshold = near_duplicate_threshold

    def convert_to_index(self):
        embeddings_list: List[np.ndarray] = [x["embedding"] for x in self.collections]

        embeddings = np.array(embeddings_list).astype(np.float32)
        embeddings_normalized = sk_normalize(embeddings, axis=1, norm="l2")

        db_index = faiss.index_factory(embeddings_normalized.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)

        db_index.add(embeddings_normalized)  # pylint: disable=no-value-for-parameter
        return embeddings_normalized, db_index

    def scale_cos_score_to_0_1(self, score: float) -> float:
        return (score + 1) / 2

    def score_images(self, project_hash: str, nearest_distances: np.ndarray, nearest_items: np.ndarray):
        previous_duplicates: dict[int, int] = {}

        for i in range(nearest_items.shape[0]):
            if nearest_items[i, 0] in previous_duplicates:
                original_item = self.collections[previous_duplicates[nearest_items[i, 0]]]
                self.scores[self.collections[nearest_items[i, 0]]["data_unit"]] = DataUnitInfo(
                    0.0,
                    f"Duplicate image. To see the original check [here](https://app.encord.com/label_editor/{original_item['data_unit']}&{project_hash}/{original_item['frame']}).",
                )
                continue
            for j in range(1, nearest_items.shape[1]):
                if abs(1.0 - nearest_distances[i, j]) < 1e-5:
                    previous_duplicates[nearest_items[i, j]] = nearest_items[i, 0]
                else:
                    if self.scale_cos_score_to_0_1(nearest_distances[i, j]) >= self.near_duplicate_threshold:
                        self.scores[self.collections[nearest_items[i, 0]]["data_unit"]] = DataUnitInfo(
                            1 - self.scale_cos_score_to_0_1(nearest_distances[i, j]), "Near duplicate image"
                        )
                    else:
                        self.scores[self.collections[nearest_items[i, 0]]["data_unit"]] = DataUnitInfo(
                            1 - self.scale_cos_score_to_0_1(nearest_distances[i, j]), ""
                        )
                    break

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        self.collections = get_cnn_embeddings(iterator, embedding_type=EmbeddingType.IMAGE)

        if len(self.collections) > 0:

            embeddings, db_index = self.convert_to_index()
            # For more information why we set the below threshold
            # see here: https://github.com/facebookresearch/faiss/wiki/Implementation-notes#matrix-multiplication-to-do-many-l2-distance-computations
            # If the 2nd approach is used, identical images have distance more than zero
            # which affects this metric because we want identical images to have a zero value
            faiss.cvar.distance_compute_blas_threshold = embeddings.shape[0] + 1
            nearest_distances, nearest_items = db_index.search(
                embeddings, embeddings.shape[0]
            )  # pylint: disable=no-value-for-parameter
            nearest_items = fix_duplicate_image_orders_in_knn_graph_all_rows(nearest_items)

            self.score_images(iterator.project.project_hash, nearest_distances, nearest_items)

        else:
            logger.info("<yellow>[Skipping]</yellow> The embedding file is empty.")

        for data_unit, img_pth in iterator.iterate(desc="Writing scores to a file"):

            data_unit_info = self.scores[data_unit["data_hash"]]
            writer.write(
                score=float(data_unit_info.score),
                description=data_unit_info.description,
            )
