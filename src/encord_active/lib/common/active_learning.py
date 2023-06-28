# Temporary place for utility functions used in active learning examples.
# They should be integrated as features of the corresponding classes.
from typing import List, Optional, Tuple

import pandas as pd
from PIL import Image

from encord_active.lib.common.iterator import DatasetIterator
from encord_active.lib.project.project_file_structure import ProjectFileStructure


def get_data(
    project_fs: ProjectFileStructure,
    data_hashes: list[tuple[str, str]],
    class_name: str,
):
    image_paths, y = get_data_from_data_hashes(project_fs, data_hashes, class_name)
    images = [Image.open(image_path) for image_path in image_paths]
    return images, y


def get_data_hashes_from_project(project_fs: ProjectFileStructure, subset_size: int = None) -> list[tuple[str, str]]:
    iterator = DatasetIterator(project_fs.project_dir, subset_size)
    data_hashes = [(iterator.label_hash, iterator.du_hash) for _ in iterator.iterate()]
    return data_hashes


def get_data_from_data_hashes(
    project_fs: ProjectFileStructure, data_hashes: list[tuple[str, str]], class_name: str
) -> Tuple[List[str], List[str]]:
    image_urls, class_labels = zip(*(get_data_sample(project_fs, data_hash, class_name) for data_hash in data_hashes))
    return list(image_urls), list(class_labels)


def get_data_sample(
    project_fs: ProjectFileStructure, data_hash: tuple[str, str], class_name: str
) -> Tuple[str, Optional[str]]:
    label_hash, du_hash = data_hash
    lr_struct = project_fs.label_row_structure(label_hash)

    # get classification label
    label_row = lr_struct.label_row_json
    class_label = get_classification_label(label_row, du_hash, class_name=class_name)

    # get image path
    image_url = next(du_struct.signed_url for du_struct in lr_struct.iter_data_unit(du_hash))

    return image_url, class_label


def get_classification_label(label_row, du_hash: str, class_name: str):
    data_unit = label_row["data_units"][du_hash]
    filtered_class = [_class for _class in data_unit["labels"]["classifications"] if _class["name"] == class_name]
    if len(filtered_class) == 0:
        return None
    class_hash = filtered_class[0]["classificationHash"]
    answers = label_row["classification_answers"].get(class_hash, {}).get("classifications", [{}])[0].get("answers")
    if isinstance(answers, str):
        return answers
    elif isinstance(answers, list):
        return answers[0]["name"]
    else:
        return None


def get_metric_results(project_fs: ProjectFileStructure, acq_func):
    unique_acq_func_name = acq_func.metadata.get_unique_name()
    acq_func_results = pd.read_csv(project_fs.metrics / f"{unique_acq_func_name}.csv")
    return acq_func_results


def get_n_best_ranked_data_samples(
    acq_func_results,
    n,
    rank_by: str,
    filter_by_data_hashes: list[tuple[str, str]] = None,
    exclude_data_hashes: list[tuple[str, str]] = None,
):
    # filter acquisition function results to include/exclude data hashes stated by the user
    if filter_by_data_hashes is not None:
        str_data_hashes = tuple(f"{label_hash}_{du_hash}" for label_hash, du_hash in filter_by_data_hashes)
        acq_func_results = acq_func_results[acq_func_results["identifier"].str.startswith(str_data_hashes, na=False)]

    if exclude_data_hashes is not None:
        str_data_hashes = tuple(f"{label_hash}_{du_hash}" for label_hash, du_hash in exclude_data_hashes)
        acq_func_results = acq_func_results[~acq_func_results["identifier"].str.startswith(str_data_hashes, na=False)]

    if rank_by == "asc":  # get the first n data samples if they were sorted by ascending score order
        best_n = acq_func_results[["identifier", "score"]].nsmallest(n, "score", keep="first")
    elif rank_by == "desc":  # get the first n data samples if they were sorted by descending score order
        best_n = acq_func_results[["identifier", "score"]].nlargest(n, "score", keep="first")
    else:
        raise ValueError

    identifiers, scores = zip(*best_n.itertuples(index=False))
    data_hashes = tuple(get_data_hash_from_identifier(identifier) for identifier in identifiers)
    return data_hashes, scores


def get_data_hash_from_identifier(identifier: str):
    return tuple(identifier.split("_", maxsplit=2)[:2])
