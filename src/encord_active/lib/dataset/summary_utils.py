import json
from pathlib import Path

import numpy as np
from encord.constants.enums import DataType
from dataclasses import dataclass, field

@dataclass
class AnnotationStatistics:
    objects: dict = field(default_factory=dict)
    classifications: dict = field(default_factory=dict)
    total_object_labels: int = 0
    total_classification_labels: int = 0

def get_all_image_sizes(project_folder: Path) -> np.ndarray:
    image_sizes = []
    for label_row in (project_folder / "data").iterdir():
        if (label_row / "label_row.json").exists():
            label_row_meta = json.loads((label_row / "label_row.json").read_text(encoding="utf-8"))
            if label_row_meta["data_type"] in [DataType.IMAGE.value, DataType.IMG_GROUP.value]:
                for data_unit in label_row_meta["data_units"].values():
                    image_sizes.append([data_unit["width"], data_unit["height"]])

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


def get_all_annotation_numbers(project_folder: Path) -> AnnotationStatistics:
    """
    returns label statistics for both objects and classifications. Does not count nested
    labels, only counts the immediate labels.
    """

    labels: AnnotationStatistics = AnnotationStatistics()
    classification_label_counter = 0
    object_label_counter = 0

    project_ontology = json.loads((project_folder / "ontology.json").read_text(encoding="utf-8"))
    for object_item in project_ontology["objects"]:
        labels.objects[object_item["name"]] = 0
    for classification_item in project_ontology["classifications"]:
        labels.classifications[classification_item["attributes"][0]["name"]] = {}

        # For radio and checkbox types
        for option in classification_item["attributes"][0].get("options", []):
            labels.classifications[classification_item["attributes"][0]["name"]][option["label"]] = 0

    for label_row in (project_folder / "data").iterdir():
        if (label_row / "label_row.json").exists():
            label_row_meta = json.loads((label_row / "label_row.json").read_text(encoding="utf-8"))
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
                        classification_answer_item = label_row_meta["classification_answers"][classificationHash][
                            "classifications"
                        ][0]
                        classification_question_name = classification_answer_item["name"]
                        if classification_question_name in labels.classifications:

                            if isinstance(classification_answer_item["answers"], list):
                                for answer_item in classification_answer_item["answers"]:
                                    if answer_item["name"] in labels.classifications[classification_question_name]:
                                        labels.classifications[classification_question_name][
                                            answer_item["name"]
                                        ] += 1
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
