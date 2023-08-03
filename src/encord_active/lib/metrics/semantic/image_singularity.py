from dataclasses import dataclass

import numpy as np
from loguru import logger

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.utils import (
    fix_duplicate_image_orders_in_knn_graph_all_rows,
)
from encord_active.lib.embeddings.embedding_index import (
    EmbeddingIndex,
    EmbeddingSearchResult,
)
from encord_active.lib.embeddings.types import LabelEmbedding
from encord_active.lib.metrics.metric import Metric
from encord_active.lib.metrics.types import DataType, EmbeddingType, MetricType
from encord_active.lib.metrics.writer import CSVMetricWriter
from encord_active.lib.project.project_file_structure import ProjectFileStructure

logger = logger.opt(colors=True)


@dataclass
class DataUnitInfo:
    score: float
    description: str


class ImageSingularity(Metric):
    def __init__(self, near_duplicate_threshold=0.03):
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
            doc_url="https://docs.encord.com/docs/active-data-quality-metrics#image-singularity",
            metric_type=MetricType.SEMANTIC,
            data_type=DataType.IMAGE,
            embedding_type=EmbeddingType.IMAGE,
        )
        self.near_duplicate_threshold = near_duplicate_threshold

    def score_images(
        self, embedding_info: list[LabelEmbedding], search_result: EmbeddingSearchResult, project_hash: str
    ) -> dict[str, DataUnitInfo]:
        scores: dict[str, DataUnitInfo] = {}
        previous_duplicates: dict[int, int] = {}

        for item_idx, (neighbors, distances) in enumerate(zip(*search_result)):
            data_hash = embedding_info[item_idx]["data_unit"]
            if item_idx in previous_duplicates:
                original_item = embedding_info[previous_duplicates[item_idx]]
                scores[data_hash] = DataUnitInfo(
                    0.0,
                    f"Duplicate image. To see the original check [here](https://app.encord.com/label_editor/{original_item['data_unit']}&{project_hash}/{original_item['frame']}).",
                )
                continue

            for neighbor_idx, neighbor_dist in zip(neighbors, distances):
                if neighbor_dist < 1e-5:
                    previous_duplicates[neighbor_idx] = item_idx
                else:
                    if neighbor_dist <= self.near_duplicate_threshold:
                        scores[data_hash] = DataUnitInfo(neighbor_dist, "Near duplicate image")
                    else:
                        scores[data_hash] = DataUnitInfo(neighbor_dist, "")
                    break
        return scores

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        if not self.metadata.embedding_type:
            logger.error(
                f"<yellow>[Skipping]</yellow> No `embedding_type` provided for the {self.metadata.title} metric!"
            )
            return

        pfs = ProjectFileStructure(iterator.cache_dir)
        embedding_index, embedding_info = EmbeddingIndex.from_project(pfs, self.metadata.embedding_type)

        if embedding_index is None or len(embedding_info) == 0:
            logger.info("<yellow>[Skipping]</yellow> The embedding file is empty.")
            return

        embeddings = np.stack([e["embedding"] for e in embedding_info])
        query_res = embedding_index.query(embeddings, k=30)
        fix_duplicate_image_orders_in_knn_graph_all_rows(query_res.indices)
        scores = self.score_images(embedding_info, query_res, iterator.project.project_hash)

        for data_unit, _ in iterator.iterate(desc="Writing scores to a file"):
            data_unit_info = scores.get(data_unit["data_hash"])
            if data_unit_info is not None:
                writer.write(
                    score=float(data_unit_info.score),
                    description=data_unit_info.description,
                )
