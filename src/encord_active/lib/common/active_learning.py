# Temporary place for utility functions used in active learning examples.
# They should be integrated as features of the corresponding classes.
import json

import pandas as pd

from encord_active.lib.common.iterator import DatasetIterator
from encord_active.lib.metrics.acquisition_functions import BaseModelWrapper
from encord_active.lib.metrics.execute import execute_metrics
from encord_active.lib.project.project_file_structure import ProjectFileStructure


def get_data(project_fs: ProjectFileStructure, model: BaseModelWrapper, subset_size=None):
    data_hashes = get_data_hashes_from_project(project_fs, subset_size)
    image_paths, y = get_data_from_data_hashes(project_fs, data_hashes)
    X = model.prepare_data(image_paths)
    return X, y


def get_data_hashes_from_project(project_fs: ProjectFileStructure, subset_size: int = None):
    iterator = DatasetIterator(project_fs.project_dir, subset_size)
    data_hashes = [(iterator.label_hash, iterator.du_hash) for _ in iterator.iterate()]
    return data_hashes


def get_data_from_data_hashes(project_fs: ProjectFileStructure, data_hashes: list[tuple[str, str]]):
    image_paths, class_labels = zip(*(get_data_sample(project_fs, data_hash) for data_hash in data_hashes))
    return list(image_paths), list(class_labels)


def get_data_sample(project_fs: ProjectFileStructure, data_hash: tuple[str, str], class_name: str):
    label_hash, du_hash = data_hash
    lr_struct = project_fs.label_row_structure(label_hash)

    # get classification label
    label_row = json.loads(lr_struct.label_row_file.read_text())
    class_label = get_classification_label(label_row, du_hash, class_name=class_name)

    # get image path
    image_path = lr_struct.images_dir / f"{du_hash}.{label_row['data_units'][du_hash]['data_type'].split('/')[-1]}"

    return image_path, class_label


def get_classification_label(label_row, du_hash: str, class_name: str):
    # only works for text classifications, extend if necessary
    data_unit = label_row["data_units"][du_hash]
    filtered_class = [_class for _class in data_unit["labels"]["classifications"] if _class["name"] == class_name]
    if len(filtered_class) == 0:
        return None
    class_hash = filtered_class[0]["classificationHash"]
    class_label = label_row["classification_answers"][class_hash]["classifications"][0]["answers"]
    return class_label


def get_n_best_ranked_data_samples(project_fs: ProjectFileStructure, acq_func_instance, n, data_hashes, rank_by: str):
    execute_metrics([acq_func_instance], data_dir=project_fs.project_dir)
    unique_acq_func_name = acq_func_instance.metadata.get_unique_name()
    acq_func_results = pd.read_csv(project_fs.metrics / f"{unique_acq_func_name}.csv")

    # filter acquisition function results to only contain data samples specified in data_hashes
    str_data_hashes = tuple(f"{label_hash}_{du_hash}" for label_hash, du_hash in data_hashes)
    filtered_results = acq_func_results[acq_func_results["identifier"].str.startswith(str_data_hashes, na=False)]

    if rank_by == "asc":  # get the first n data samples if they were sorted by ascending score order
        best_n = filtered_results[["identifier", "score"]].nsmallest(n, "score", keep="first")["identifier"]
    elif rank_by == "desc":  # get the first n data samples if they were sorted by descending score order
        best_n = filtered_results[["identifier", "score"]].nlargest(n, "score", keep="first")["identifier"]
    else:
        raise ValueError
    return [get_data_hash_from_identifier(identifier) for identifier in best_n]


def get_data_hash_from_identifier(identifier: str):
    return tuple(identifier.split("_", maxsplit=2)[:2])
