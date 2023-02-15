from collections import Counter
from typing import List

import faiss
import numpy as np
from loguru import logger
from tqdm.auto import tqdm

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.utils import (
    fix_duplicate_image_orders_in_knn_graph_all_rows,
)
from encord_active.lib.embeddings.cnn import get_cnn_embeddings
from encord_active.lib.embeddings.dimensionality_reduction import (
    generate_2d_embedding_data,
)
from encord_active.lib.embeddings.utils import LabelEmbedding
from encord_active.lib.metrics.metric import (
    AnnotationType,
    DataType,
    EmbeddingType,
    Metric,
    MetricType,
)
from encord_active.lib.metrics.writer import CSVMetricWriter

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
            metric_type=MetricType.GEOMETRIC,
            data_type=DataType.IMAGE,
            annotation_type=[
                AnnotationType.OBJECT.BOUNDING_BOX,
                AnnotationType.OBJECT.ROTATABLE_BOUNDING_BOX,
                AnnotationType.OBJECT.POLYGON,
            ],
            embedding_type=EmbeddingType.OBJECT,
        )
        self.collections: dict[str, LabelEmbedding] = {}
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

    def convert_to_index(self):
        embeddings_list: List[np.ndarray] = []
        noisy_labels_list: List[int] = []
        for x in self.collections.values():
            embeddings_list.append(x["embedding"])
            noisy_labels_list.append(self.object_name_to_index[x["name"]])

        embeddings = np.array(embeddings_list).astype(np.float32)
        noisy_labels = np.array(noisy_labels_list).astype(np.int32)

        db_index = faiss.IndexFlatL2(embeddings.shape[1])
        db_index.add(embeddings)  # pylint: disable=no-value-for-parameter
        return embeddings, db_index, noisy_labels

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

    def unpack_collections(self, collections: list) -> None:
        for item in tqdm(collections, desc="Unpacking embeddings"):
            identifier = self.get_identifier_from_collection_item(item)
            if item["dataset_title"] == self.project_dataset_name or self.project_dataset_name == "":
                self.collections[identifier] = item

    def get_identifier_from_collection_item(self, item):
        return f'{item["label_row"]}_{item["data_unit"]}_{item["frame"]:05d}_{item["labelHash"]}'

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        ontology_contains_objects = self.setup(iterator)
        if not ontology_contains_objects:
            logger.info("<yellow>[Skipping]</yellow> No objects in the project ontology.")
            return

        collections = get_cnn_embeddings(iterator, embedding_type=EmbeddingType.OBJECT)

        if len(collections) > 0:
            generate_2d_embedding_data(EmbeddingType.OBJECT, iterator.cache_dir)
            embedding_identifiers = [self.get_identifier_from_collection_item(item) for item in collections]

            self.unpack_collections(collections)

            embedding_database, index, noisy_labels = self.convert_to_index()
            nearest_distances, nearest_metrics = index.search(  # pylint: disable=no-value-for-parameter
                embedding_database, self.num_nearest_neighbors
            )
            nearest_metrics = fix_duplicate_image_orders_in_knn_graph_all_rows(nearest_metrics)

            nearest_labels = np.take(noisy_labels, nearest_metrics)
            noisy_labels_tmp, nearest_labels_except_self = np.split(nearest_labels, [1], axis=-1)
            assert np.all(noisy_labels == noisy_labels_tmp.squeeze()), "Failed class index extraction"

            label_matches = np.equal(nearest_labels_except_self, np.expand_dims(noisy_labels, axis=-1))
            collections_scores = label_matches.mean(axis=-1)

            valid_annotation_types = {annotation_type.value for annotation_type in self.metadata.annotation_type}
            for data_unit, img_pth in iterator.iterate(desc="Storing index"):
                for obj in data_unit["labels"].get("objects", []):
                    if obj["shape"] not in valid_annotation_types:
                        continue

                    key = iterator.get_identifier(object=obj)
                    if key in self.collections:
                        assert obj["name"] == self.collections[key]["name"], "Indexing inconsistencies"
                        idx = embedding_identifiers.index(key)
                        description = self.get_description_info(nearest_labels_except_self[idx], noisy_labels[idx])
                        writer.write(collections_scores[idx], obj, description=description)
        else:
            logger.info("<yellow>[Skipping]</yellow> The object embedding file is empty.")
