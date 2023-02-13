from collections import Counter
from pathlib import Path
from typing import Dict, List, TypedDict

import faiss
import numpy as np
import torch
from encord.objects.common import PropertyType
from loguru import logger

from encord_active.lib.common.iterator import Iterator
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
from encord_active.lib.metrics.utils import is_multiclass_ontology
from encord_active.lib.metrics.writer import CSVMetricWriter

logger = logger.opt(colors=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DescriptionInfo(TypedDict):
    question: str
    answer: str
    target_answer: str
    above_threshold: bool


class ClassificationInfo(DescriptionInfo):
    score: float
    description: str
    annotator: str


class ImageLevelQualityTest(Metric):
    def __init__(self, num_nearest_neighbors: int = 10, certainty_ratio: float = 0.60):
        """

        :param num_nearest_neighbors: determines how many nearest neighbors' labels should be checked for the quality.
        This parameter should be +1 than the actual intended number because in the nearest neighbor graph queried
        embedding already exists
        :param certainty_ratio: ratio to determine if the annotation is mislabeled or there is a different problem in
        the annotation
        """
        super(ImageLevelQualityTest, self).__init__(
            title="Image-level Annotation Quality",
            short_description="Compares image classifications against similar images",
            long_description=r"""This metric creates embeddings from images. Then, these embeddings are used to build
    nearest neighbor graph. Similar embeddings' classifications are compared against each other.
        """,
            metric_type=MetricType.SEMANTIC,
            data_type=DataType.IMAGE,
            annotation_type=[AnnotationType.CLASSIFICATION.RADIO],
            embedding_type=EmbeddingType.CLASSIFICATION,
        )
        self.collections: list[LabelEmbedding] = []
        self.featureNodeHash_to_index: dict[str, dict] = {}
        self.featureNodeHash_to_name: dict[str, dict] = {}
        self.featureNodeHash_to_question_name: dict[str, str] = {}
        self.index_to_answer_name: dict[str, dict] = {}
        self.identifier_to_embedding: dict[str, np.ndarray] = {}
        self.cache_dir: Path = Path()
        self.num_nearest_neighbors = num_nearest_neighbors
        self.certainty_ratio = certainty_ratio

    def extract_description_info(
        self, question_hash: str, nearest_labels: dict[str, np.ndarray], index: int
    ) -> DescriptionInfo:
        gt_label = nearest_labels["gt_label"][index]
        neighbor_labels = nearest_labels["neighbor_labels"][index].tolist()

        threshold = int(len(neighbor_labels) * self.certainty_ratio)
        counter = Counter(neighbor_labels)
        target_label, target_label_frequency = counter.most_common(1)[0]

        return DescriptionInfo(
            question=self.featureNodeHash_to_question_name[question_hash],
            answer=self.index_to_answer_name[question_hash][gt_label],
            target_answer=self.index_to_answer_name[question_hash][target_label],
            above_threshold=target_label_frequency > threshold,
        )

    def build_description_string(self, info: DescriptionInfo) -> str:
        if not info["above_threshold"]:
            return f":question: For the question `{info['question']}`, the image is annotated as `{info['answer']}`. \
            The annotated class may be wrong, as the most similar objects have different classes."

        if info["answer"] == info["target_answer"]:
            return f":heavy_check_mark: For the question `{info['question']}`, the image is correctly annotated as `{info['answer']}`"
        else:
            return f":x: For the question `{info['question']}`, the image is annotated as `{info['answer']}`. Similar \
                    images were annotated as `{info['target_answer']}`."

    def convert_to_indexes(self):
        embedding_databases, indexes = {}, {}
        question_hashes = {c["featureHash"] for c in self.collections}

        for question_hash in question_hashes:
            selected_collections = list(filter(lambda c: c["featureHash"] == question_hash, self.collections))

            if len(selected_collections) > self.num_nearest_neighbors:
                embedding_database = np.stack(list(map(lambda x: x["embedding"], selected_collections)))

                index = faiss.IndexFlatL2(embedding_database.shape[1])
                index.add(embedding_database)  # pylint: disable=no-value-for-parameter

                embedding_databases[question_hash] = embedding_database
                indexes[question_hash] = index

        return embedding_databases, indexes

    def get_nearest_indexes(self):
        # from collections build faiss index
        embedding_databases, indexes = self.convert_to_indexes()

        nearest_metrics = {}
        for question_hash in indexes:
            nearest_distance, nearest_index = indexes[question_hash].search(  # pylint: disable=no-value-for-parameter
                embedding_databases[question_hash], self.num_nearest_neighbors
            )
            nearest_metrics[question_hash] = nearest_index

        return nearest_metrics

    def transform_neighbors_to_labels_for_all_questions(self, nearest_indexes: dict[str, np.ndarray]):
        nearest_labels_all_questions = {}

        for question in nearest_indexes:
            noisy_labels_list = []
            for i in range(nearest_indexes[question].shape[0]):
                answers = self.collections[i]["classification_answers"]

                if not answers:
                    gt_label = "Unclassified"
                else:
                    gt_label = self.featureNodeHash_to_index[question][answers["answer_featureHash"]]

                noisy_labels_list.append(gt_label)

            noisy_labels = np.array(noisy_labels_list).astype(np.int32)
            nearest_labels = np.take(noisy_labels, nearest_indexes[question])
            noisy_labels_tmp, nearest_labels_except_self = np.split(nearest_labels, [1], axis=-1)
            assert np.all(noisy_labels == noisy_labels_tmp.squeeze()), "Failed class index extraction"

            nearest_labels_all_questions[question] = {
                "gt_label": noisy_labels,
                "neighbor_labels": nearest_labels_except_self,
            }

        return nearest_labels_all_questions

    def convert_nearest_labels_to_scores(self, nearest_labels: dict[str, np.ndarray]) -> List[float]:
        """
        Higher scores mean more reliable label
        :param nearest_labels: output of the index.search()
        :return: score for each row
        """

        label_matches = np.equal(nearest_labels["neighbor_labels"], np.expand_dims(nearest_labels["gt_label"], axis=-1))
        collections_scores = label_matches.mean(axis=-1)

        return collections_scores

    def convert_nearest_labels_to_scores_for_all_questions(
        self, nearest_labels_all_questions: Dict[str, dict]
    ) -> Dict[str, List[float]]:
        collections_scores_all_questions: Dict[str, List[float]] = {}
        for question in nearest_labels_all_questions:
            scores = self.convert_nearest_labels_to_scores(nearest_labels_all_questions[question])
            collections_scores_all_questions[question] = scores
        return collections_scores_all_questions

    def create_key_score_pairs(self, nearest_indexes: dict[str, np.ndarray]):

        nearest_labels_all_questions = self.transform_neighbors_to_labels_for_all_questions(nearest_indexes)
        collections_scores_all_questions = self.convert_nearest_labels_to_scores_for_all_questions(
            nearest_labels_all_questions
        )

        key_score_pairs: Dict[str, Dict[str, ClassificationInfo]] = {}
        for i, collection in enumerate(self.collections):
            label_hash = collection["label_row"]
            du_hash = collection["data_unit"]
            frame_idx = int(collection["frame"])

            key = f"{label_hash}_{du_hash}_{frame_idx:05d}"

            temp_entry = {}
            question_hash = collection["featureHash"]
            if question_hash in collections_scores_all_questions:
                # sub_collection_index = self.question_hash_to_collection_indexes[question].index(i)
                score = collections_scores_all_questions[question_hash][i]
                answers = collection["classification_answers"]
                if not answers:
                    raise Exception("Missing classification answers")

                description_info = self.extract_description_info(
                    question_hash, nearest_labels_all_questions[question_hash], i
                )

                temp_entry[question_hash] = ClassificationInfo(
                    score=score,
                    description=self.build_description_string(description_info),
                    annotator=answers["annotator"],
                    **description_info,  # type: ignore
                )

            key_score_pairs[key] = temp_entry
        return key_score_pairs

    def fix_nearest_indexes(self, nearest_indexes: dict[str, np.ndarray]) -> None:
        """
        In the presence of duplicate images, ordering of the nearest metrics might be confused. This function ensures
        that first column value is always equal to row number.
        :param nearest_indexes: dictionary to keep the nearest metrics for different classification questions
        :return: None
        """
        for question in nearest_indexes:
            question_nearest_metrics = nearest_indexes[question]
            for i in range(question_nearest_metrics.shape[0]):
                if i != question_nearest_metrics[i, 0]:
                    item_index = np.where(question_nearest_metrics[i] == i)[0][0]
                    question_nearest_metrics[i, 0], question_nearest_metrics[i, item_index] = (
                        question_nearest_metrics[i, item_index],
                        question_nearest_metrics[i, 0],
                    )

    def setup(self, iterator: Iterator) -> bool:
        """
        Looks for classes to use.
        :param iterator:
        :return: True if there is anything to do based on the ontology. Otherwise, False.
        """
        self.cache_dir = iterator.cache_dir
        # TODO: This only evaluates immediate classes, not nested ones
        # TODO: Look into other types of classifications apart from radio buttons
        found_any = False
        for class_label in iterator.project.ontology.classifications:
            class_question = class_label.attributes[0]
            if class_question.get_property_type() == PropertyType.RADIO:
                found_any = True
                self.featureNodeHash_to_index[class_label.feature_node_hash] = {}
                self.featureNodeHash_to_name[class_label.feature_node_hash] = {}
                self.featureNodeHash_to_question_name[class_label.feature_node_hash] = class_label.attributes[0].name
                self.index_to_answer_name[class_label.feature_node_hash] = {}

                for counter, option in enumerate(class_question.options):
                    self.featureNodeHash_to_index[class_label.feature_node_hash][option.feature_node_hash] = counter
                    self.featureNodeHash_to_name[class_label.feature_node_hash][option.feature_node_hash] = option.label
                    self.index_to_answer_name[class_label.feature_node_hash][counter] = option.label

        return found_any

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        project_has_classifications = self.setup(iterator)
        if not project_has_classifications:
            logger.info("<yellow>[Skipping]</yellow> No frame level classifications in the project ontology.")

        # TODO: move me somewhere else, this is here to ensure the generation of image embeddings
        get_cnn_embeddings(iterator, embedding_type=EmbeddingType.IMAGE)
        generate_2d_embedding_data(EmbeddingType.IMAGE, iterator.cache_dir)

        self.collections = get_cnn_embeddings(iterator, embedding_type=EmbeddingType.CLASSIFICATION)

        if len(self.collections) > 0:
            generate_2d_embedding_data(EmbeddingType.CLASSIFICATION, iterator.cache_dir)
            nearest_indexes = self.get_nearest_indexes()
            self.fix_nearest_indexes(nearest_indexes)
            key_score_pairs = self.create_key_score_pairs(nearest_indexes)
            for data_unit, img_pth in iterator.iterate(desc="Storing index"):
                key = iterator.get_identifier()
                is_multiclass = is_multiclass_ontology(iterator.project.ontology)

                if key in key_score_pairs:
                    for classification in data_unit["labels"].get("classifications", []):
                        question_featureHash = classification["featureHash"]
                        if question_featureHash not in self.featureNodeHash_to_question_name:
                            continue

                        classification_hash = classification["classificationHash"]
                        classification_info = key_score_pairs[key][question_featureHash]

                        label_class = f"{classification_info['answer']}"
                        if is_multiclass:
                            label_class = f"{classification_info['question']}:{label_class}"

                        if question_featureHash in key_score_pairs[key]:
                            writer.write(
                                key_score_pairs[key][question_featureHash]["score"],
                                key=f"{key}_{classification_hash}",
                                description=classification_info["description"],
                                label_class=label_class,
                                labels=classification,
                            )
        else:
            logger.info("<yellow>[Skipping]</yellow> The embedding file is empty.")
