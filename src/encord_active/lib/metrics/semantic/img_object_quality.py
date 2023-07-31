from collections import Counter

import numpy as np
from loguru import logger
from tqdm.auto import tqdm

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.utils import (
    fix_duplicate_image_orders_in_knn_graph_all_rows,
)
from encord_active.lib.embeddings.embedding_index import EmbeddingIndex
from encord_active.lib.embeddings.types import LabelEmbedding
from encord_active.lib.metrics.metric import Metric
from encord_active.lib.metrics.types import (
    AnnotationType,
    DataType,
    EmbeddingType,
    MetricType,
)
from encord_active.lib.metrics.writer import CSVMetricWriter
from encord_active.lib.project.project_file_structure import ProjectFileStructure

logger = logger.opt(colors=True)


class ObjectEmbeddingSimilarityTest(Metric):
    def __init__(self, num_nearest_neighbors: int = 10, certainty_ratio: float = 0.6, project_dataset_name: str = ""):
        """
        :param num_nearest_neighbors: determines how many nearest neighbors' labels should be checked for the quality.
         This parameter should be +1 than the actual intended number because in the nearest neighbor graph queried
         embedding already exists
        :param project_dataset_name: if QM is wanted to be run on specific dataset, name should be given here. If it is
         empty, it means evaluate all datasets in the project
        """
        super(ObjectEmbeddingSimilarityTest, self).__init__(
            title="Object Annotation Quality",
            short_description="Compares object annotations against similar image crops",
            long_description=r"""This metric transforms polygons into bounding boxes
    and an embedding for each bounding box is extracted. Then, these embeddings are compared
    with their neighbors. If the neighbors are annotated differently, a low score is given to it.
    """,
            doc_url="https://docs.encord.com/docs/active-label-quality-metrics#object-annotation-quality",
            metric_type=MetricType.GEOMETRIC,
            data_type=DataType.IMAGE,
            annotation_type=[
                AnnotationType.OBJECT.BOUNDING_BOX,
                AnnotationType.OBJECT.ROTATABLE_BOUNDING_BOX,
                AnnotationType.OBJECT.POLYGON,
            ],
            embedding_type=EmbeddingType.OBJECT,
        )
        self.label_embeddings: dict[str, LabelEmbedding] = {}
        self.featureNodeHash_to_index: dict[str, int] = {}
        self.index_to_object_name: dict[int, str] = {}
        self.object_name_to_index: dict[str, int] = {}
        self.num_nearest_neighbors = num_nearest_neighbors
        self.certainty_ratio = certainty_ratio
        self.project_dataset_name = project_dataset_name

    def setup(self, iterator) -> bool:
        found_any = False
        for i, object_label in enumerate(iterator.project.ontology.objects):
            found_any = True
            self.featureNodeHash_to_index[object_label.feature_node_hash] = i
            self.index_to_object_name[i] = object_label.name
            self.object_name_to_index[object_label.name] = i
        return found_any

    def convert_to_np(self, label_embeddings):
        embeddings = np.stack([l["embedding"] for l in label_embeddings])
        noisy_labels = np.array([self.featureNodeHash_to_index[l["featureHash"]] for l in label_embeddings]).astype(
            np.int32
        )
        return embeddings, noisy_labels

    def get_description_info(self, nearest_labels: np.ndarray, noisy_label: int):
        threshold = int(len(nearest_labels) * self.certainty_ratio)
        counter = Counter(nearest_labels)
        target_label, target_label_frequency = counter.most_common(1)[0]

        if noisy_label == target_label and target_label_frequency > threshold:
            description = (
                f":heavy_check_mark: The object is correctly annotated as `{self.index_to_object_name[noisy_label]}`"
            )
        elif noisy_label != target_label and target_label_frequency > threshold:
            description = f":x: The object is annotated as `{self.index_to_object_name[noisy_label]}`. Similar \
             objects were annotated as `{self.index_to_object_name[target_label]}`."
        else:  # covers cases for  target_label_frequency <= threshold:
            description = f":question: The object is annotated as `{self.index_to_object_name[noisy_label]}`. \
            The annotated class may be wrong, as the most similar objects have different classes."
        return description

    def unpack_label_embeddings(self, collections: list) -> None:
        for item in tqdm(collections, desc="Unpacking embeddings"):
            identifier = self.label_embedding_to_identifier(item)
            if item["dataset_title"] == self.project_dataset_name or self.project_dataset_name == "":
                self.label_embeddings[identifier] = item

    def label_embedding_to_identifier(self, item):
        return f'{item["label_row"]}_{item["data_unit"]}_{item["frame"]:05d}_{item["labelHash"]}'

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        ontology_contains_objects = self.setup(iterator)
        if not ontology_contains_objects:
            logger.info("<yellow>[Skipping]</yellow> No objects in the project ontology.")
            return

        project_file_structure = ProjectFileStructure(iterator.cache_dir)
        if self.metadata.embedding_type:
            index, label_embeddings = EmbeddingIndex.from_project(
                project_file_structure, self.metadata.embedding_type, iterator
            )
        else:
            logger.error(
                f"<yellow>[Skipping]</yellow> No `embedding_type` provided for the {self.metadata.title} metric!"
            )
            return

        if index is None or len(label_embeddings) == 0:
            logger.info("<yellow>[Skipping]</yellow> The object embedding file is empty.")
            return

        embedding_identifiers = [self.label_embedding_to_identifier(emb) for emb in label_embeddings]
        self.unpack_label_embeddings(label_embeddings)

        embeddings, noisy_labels = self.convert_to_np(label_embeddings)
        query_result = index.query(embeddings, k=self.num_nearest_neighbors)
        fix_duplicate_image_orders_in_knn_graph_all_rows(query_result.indices)

        nearest_labels = np.take(noisy_labels, query_result.indices)
        noisy_labels_tmp, nearest_labels_except_self = np.split(nearest_labels, [1], axis=-1)
        assert np.all(noisy_labels == noisy_labels_tmp.squeeze()), "Failed class index extraction"

        label_matches = np.equal(nearest_labels_except_self, np.expand_dims(noisy_labels, axis=-1))
        label_scores = label_matches.mean(axis=-1)

        valid_annotation_types = {annotation_type.value for annotation_type in self.metadata.annotation_type}
        for data_unit, _ in iterator.iterate(desc="Storing index"):
            for obj in data_unit["labels"].get("objects", []):
                if obj["shape"] not in valid_annotation_types:
                    continue

                key = iterator.get_identifier(object=obj)
                if key in self.label_embeddings:
                    assert obj["name"] == self.label_embeddings[key]["name"], "Indexing inconsistencies"
                    idx = embedding_identifiers.index(key)
                    description = self.get_description_info(nearest_labels_except_self[idx], noisy_labels[idx])
                    writer.write(label_scores[idx], obj, description=description)
