import json
from typing import Dict, List

import numpy as np
from loguru import logger
from sklearn.cluster import KMeans

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.embeddings.embeddings import get_embeddings
from encord_active.lib.embeddings.types import LabelEmbedding
from encord_active.lib.labels.classification import ClassificationType
from encord_active.lib.metrics.metric import Metric
from encord_active.lib.metrics.types import DataType, EmbeddingType, MetricType
from encord_active.lib.metrics.writer import CSVMetricWriter

logger = logger.opt(colors=True)


class ImageDiversity(Metric):
    def __init__(self):
        super(ImageDiversity, self).__init__(
            title="Image Diversity",
            short_description="Ranks images according to their proximity to class prototypes (lower values = closer to "
            "class prototypes = easy samples)",
            long_description=r"""  
It clusters images according to the number of classes in the ontology (if there are both object and frame level 
classifications in the ontology, the number of object classes is taken into account) 
and rank images inside each cluster by assigning lower 
score to the ones which are closer to the cluster center. Finally, ranked images in different clusters are 
merged by keeping the samples of classes the same for the first _N_ samples.
""",
            doc_url="https://docs.encord.com/docs/active-data-quality-metrics#image-diversity",
            metric_type=MetricType.SEMANTIC,
            data_type=DataType.IMAGE,
            embedding_type=EmbeddingType.IMAGE,
        )

        self.label_embeddings: List[LabelEmbedding] = []

    def _get_cluster_size(self, iterator: Iterator) -> int:
        default_k_size = 10
        if iterator.project_file_structure.ontology.is_file():
            ontology_dict = json.loads(iterator.project_file_structure.ontology.read_text(encoding="utf-8"))
            if len(ontology_dict.get("objects", [])) > 0:
                return len(ontology_dict["objects"])
            elif len(ontology_dict.get("classifications", [])) > 0:
                if ontology_dict["classifications"][0]["attributes"][0]["type"] in [
                    ClassificationType.RADIO.value,
                    ClassificationType.CHECKLIST.value,
                ]:
                    return len(ontology_dict["classifications"][0]["attributes"][0]["options"])
                else:
                    return default_k_size
            else:
                return default_k_size
        else:
            return default_k_size

    def _get_difficulty_ranking(self, cluster_size: int) -> Dict[str, int]:
        id_to_data_hash: Dict[int, str] = {i: emb["data_unit"] for i, emb in enumerate(self.label_embeddings)}
        embeddings = np.array([emb["embedding"] for emb in self.label_embeddings]).astype(np.float32)
        kmeans: KMeans = KMeans(n_clusters=cluster_size, n_init="auto").fit(embeddings)  # type: ignore

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

        data_hash_to_score: Dict[str, int] = {
            id_to_data_hash[item]: counter + 1 for counter, item in enumerate(common_array_indices)
        }

        return data_hash_to_score

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        if self.metadata.embedding_type:
            self.label_embeddings = get_embeddings(iterator, embedding_type=self.metadata.embedding_type)
        else:
            logger.error(
                f"<yellow>[Skipping]</yellow> No `embedding_type` provided for the {self.metadata.title} metric!"
            )
            return

        if len(self.label_embeddings) == 0:
            logger.info("<yellow>[Skipping]</yellow> The embedding file is empty.")
            return

        cluster_size = self._get_cluster_size(iterator)
        if len(self.label_embeddings) < cluster_size:
            logger.info("<yellow>[Skipping]</yellow> There are very few samples compared to the number of classes.")
            return

        data_hash_to_score = self._get_difficulty_ranking(cluster_size)

        for data_unit, _ in iterator.iterate(desc="Writing scores to a file"):
            score = data_hash_to_score.get(data_unit["data_hash"])
            if score is not None:
                writer.write(score=score)
