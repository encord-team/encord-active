import json
import logging
from dataclasses import dataclass, field

import numpy as np
from encord.constants.enums import DataType
from encord.objects.ontology_structure import OntologyStructure
from pandera.typing import DataFrame

from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.project import ProjectFileStructure


@dataclass
class AnnotationStatistics:
    objects: dict = field(default_factory=dict)
    classifications: dict = field(default_factory=dict)
    total_object_labels: int = 0
    total_classification_labels: int = 0


from encord_active.lib.dataset.outliers import (
    MetricOutlierInfo,
    MetricsSeverity,
    MetricWithDistanceSchema,
    Severity,
    get_iqr_outliers,
)
from encord_active.lib.metrics.utils import MetricData, load_metric_dataframe

_COLUMNS = MetricWithDistanceSchema


def get_all_image_sizes(project_file_structure: ProjectFileStructure) -> np.ndarray:
    image_sizes = []

    with PrismaConnection(project_file_structure) as conn:
        results = conn.query_raw("SELECT width, height FROM DataUnit")
        image_sizes = [[data_unit["width"], data_unit["height"]] for data_unit in results]

    # HACK - return some result for video only datasets
    if len(image_sizes) == 0:
        image_sizes.append([-1, -1])
    return np.array(image_sizes)


def get_median_value_of_2d_array(array: np.ndarray) -> np.ndarray:
    """
    This function calculates the median value based on the product of the two dimension.
    For example, if they are image width and height, median dimensions corresponds to median average
    """
    product = array[:, 0] * array[:, 1]
    product_sorted = np.sort(product)
    median_value = product_sorted[product_sorted.size // 2]
    item_index = np.where(product == median_value)
    return array[item_index[0][0], :]


def get_all_annotation_numbers(project_file_structure: ProjectFileStructure) -> AnnotationStatistics:
    """
    Returns label statistics for both objects and classifications. Does not count nested
    labels, only counts the immediate labels.
    """

    labels: AnnotationStatistics = AnnotationStatistics()
    classification_label_counter = 0
    object_label_counter = 0

    project_ontology = json.loads(project_file_structure.ontology.read_text(encoding="utf-8"))
    ontology = OntologyStructure.from_dict(project_ontology)

    for object_item in ontology.objects:
        labels.objects[object_item.name] = 0
    for classification_item in ontology.classifications:
        labels.classifications[classification_item.attributes[0].name] = {}

        # For radio and checkbox types
        if hasattr(classification_item.attributes[0], "options"):
            for option in classification_item.attributes[0].options:
                labels.classifications[classification_item.attributes[0].name][option.label] = 0

    with PrismaConnection(project_file_structure) as conn:
        for label_row_structure in project_file_structure.iter_labels(cache_db=conn):
            label_row_meta = label_row_structure.get_label_row_json(cache_db=conn)
            if label_row_meta["data_type"] in [DataType.IMAGE.value, DataType.IMG_GROUP.value]:
                for data_unit in label_row_meta["data_units"].values():

                    object_label_counter += len(data_unit["labels"].get("objects", []))
                    classification_label_counter += len(data_unit["labels"].get("classifications", []))

                    for object_ in data_unit["labels"].get("objects", []):
                        if object_["name"] not in labels.objects:
                            print(f'Object name "{object_["name"]}" is not exist in project ontology')
                        labels.objects[object_["name"]] += 1

                    for classification in data_unit["labels"].get("classifications", []):
                        classificationHash = classification["classificationHash"]
                        if classificationHash not in label_row_meta["classification_answers"]:
                            logging.warning(
                                "Couldn't find the classification answer. Something is probably wrong with the label row"
                            )
                            continue

                        classification_answer_item = label_row_meta["classification_answers"][classificationHash][
                            "classifications"
                        ][0]
                        classification_question_name = classification_answer_item["name"]
                        if classification_question_name in labels.classifications:

                            if isinstance(classification_answer_item["answers"], list):
                                for answer_item in classification_answer_item["answers"]:
                                    if answer_item["name"] in labels.classifications[classification_question_name]:
                                        labels.classifications[classification_question_name][answer_item["name"]] += 1
                            elif isinstance(classification_answer_item["answers"], str):
                                labels.classifications[classification_question_name].setdefault(
                                    classification_answer_item["answers"], 0
                                )
                                labels.classifications[classification_question_name][
                                    classification_answer_item["answers"]
                                ] += 1

    labels.total_object_labels = object_label_counter
    labels.total_classification_labels = classification_label_counter

    return labels


def get_metric_summary(metrics: list[MetricData]) -> MetricsSeverity:
    metric_severity = MetricsSeverity()
    total_unique_severe_outliers = set()
    total_unique_moderate_outliers = set()

    for metric in metrics:
        original_df = load_metric_dataframe(metric, normalize=False)
        res = get_iqr_outliers(original_df)

        if not res:
            continue

        df, iqr_outliers = res
        values = df[[_COLUMNS.identifier, _COLUMNS.outliers_status]].values

        for identifier, outlier_status in values:
            if outlier_status == Severity.severe:
                total_unique_severe_outliers.add(identifier)
            elif outlier_status == Severity.moderate:
                total_unique_moderate_outliers.add(identifier)

        metric_severity.metrics[metric.name] = MetricOutlierInfo(
            metric=metric, df=DataFrame[MetricWithDistanceSchema](df), iqr_outliers=iqr_outliers
        )

    metric_severity.total_unique_severe_outliers = len(total_unique_severe_outliers)
    metric_severity.total_unique_moderate_outliers = len(total_unique_moderate_outliers)

    return metric_severity
