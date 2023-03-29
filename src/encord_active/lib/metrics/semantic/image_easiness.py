import json
from typing import Dict, List

import numpy as np
from loguru import logger
from sklearn.cluster import KMeans

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.embeddings.cnn import get_cnn_embeddings
from encord_active.lib.embeddings.utils import LabelEmbedding
from encord_active.lib.metrics.metric import DataType, EmbeddingType, Metric, MetricType
from encord_active.lib.metrics.writer import CSVMetricWriter

logger = logger.opt(colors=True)


class ImageEasiness(Metric):
    def __init__(self):
        super(ImageEasiness, self).__init__(
            title="Image Easiness",
            short_description="Ranks images according to their proximity to class prototypes (lower values = closer to "
            "class prototypes = easy samples)",
            long_description=r"""
This metric gives each image a ranking value that shows the image's easiness.  
It clusters images according to the number of classes in the ontology (if there are both object and frame level 
classifications in the ontology, the number of object classes is taken into account) 
and rank images inside each cluster by assigning lower 
score to the ones which are closer to the cluster center. Finally, ranked images in different clusters are 
merged by keeping the samples of classes the same for the first _N_ samples.
""",
            doc_url="https://docs.encord.com/active/docs/metrics/semantic#image-easiness",
            metric_type=MetricType.SEMANTIC,
            data_type=DataType.IMAGE,
            embedding_type=EmbeddingType.IMAGE,
        )

        self.collections: List[LabelEmbedding] = []

    def _get_cluster_size(self, iterator: Iterator) -> int:
        default_k_size = 10
        if iterator.project_file_structure.ontology.is_file():
            ontology_dict = json.loads(iterator.project_file_structure.ontology.read_text(encoding="utf-8"))
            if len(ontology_dict.get("objects", [])) > 0:
                return len(ontology_dict["objects"])
            elif len(ontology_dict.get("classifications", [])) > 0:
                return len(ontology_dict["classifications"])
            else:
                return default_k_size
        else:
            return default_k_size

    def _get_easiness_ranking(self, cluster_size: int) -> Dict[str, int]:
        id_to_data_hash: Dict[int, str] = {i: item["data_unit"] for i, item in enumerate(self.collections)}
        embeddings = np.array([item["embedding"] for item in self.collections]).astype(np.float32)
        kmeans: KMeans = KMeans(n_clusters=cluster_size).fit(embeddings)  # type: ignore

        cluster_ids_all = []

        for i in range(cluster_size):
            cluster_values = embeddings[kmeans.labels_ == i]
            cluster_ids = np.where(kmeans.labels_ == i)[0]

            distances_to_center = np.linalg.norm(cluster_values - kmeans.cluster_centers_[i], axis=1)
            cluster_ids_all.append(cluster_ids[distances_to_center.argsort()])

        common_array_indices = []
        sample_exist = True
        counter = 0
        while sample_exist:
            sample_exist = False
            for i in range(cluster_size):
                if counter < len(cluster_ids_all[i]):
                    sample_exist = True
                    common_array_indices.append(cluster_ids_all[i][counter])

            counter += 1

        data_hash_to_score: Dict[str, int] = {}
        for counter, item in enumerate(common_array_indices):
            data_hash_to_score[id_to_data_hash[item]] = counter + 1

        return data_hash_to_score

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        if self.metadata.embedding_type:
            self.collections = get_cnn_embeddings(iterator, embedding_type=self.metadata.embedding_type)
        else:
            logger.error(
                f"<yellow>[Skipping]</yellow> No `embedding_type` provided for the {self.metadata.title} metric!"
            )
            return

        if len(self.collections) > 0:
            cluster_size = self._get_cluster_size(iterator)
            data_hash_to_score = self._get_easiness_ranking(cluster_size)

            for data_unit, img_pth in iterator.iterate(desc="Writing scores to a file"):
                writer.write(score=data_hash_to_score[data_unit["data_hash"]])
        else:
            logger.info("<yellow>[Skipping]</yellow> The embedding file is empty.")
