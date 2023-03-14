import json
import pickle
import subprocess
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import yaml

from encord_active.lib.common.utils import iterate_in_batches
from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.embeddings.utils import (
    EMBEDDING_REDUCED_TO_FILENAME,
    LabelEmbedding,
    load_collections,
    save_collections,
)
from encord_active.lib.metrics.metric import EmbeddingType, MetricMetadata
from encord_active.lib.project import ProjectFileStructure
from encord_active.lib.project.metadata import fetch_project_meta


class LabelRowDataUnit(NamedTuple):
    label_row: str
    data_unit: str


def rename_files(project_file_structure: ProjectFileStructure, file_mappings: dict[LabelRowDataUnit, LabelRowDataUnit]):
    folder_maps = {old_lr: new_lr for (old_lr, old_du), (new_lr, new_du) in file_mappings.items()}
    file_maps = {old_du: new_du for (old_lr, old_du), (new_lr, new_du) in file_mappings.items()}

    for old_lr, new_lr in folder_maps.items():
        old_lr_path = project_file_structure.data / old_lr
        new_lr_path = project_file_structure.data / new_lr
        old_lr_path.rename(new_lr_path)

    for _, new_lr in folder_maps.items():
        new_lr_path = project_file_structure.data / new_lr
        for old_du_f in new_lr_path.glob("images/*.*"):
            old_du_name_parts = old_du_f.stem.split("_")
            old_du_name_parts[0] = file_maps.get(old_du_name_parts[0], old_du_name_parts[0])
            new_f_name = "_".join(old_du_name_parts) + old_du_f.suffix
            old_du_f.rename(new_lr_path / "images" / new_f_name)


def update_embedding_identifiers(
    project_file_structure: ProjectFileStructure, embedding_type: EmbeddingType, renaming_map: dict[str, str]
):
    def _update_identifiers(embedding: LabelEmbedding, renaming_map: dict[str, str]):
        old_lr, old_du = embedding["label_row"], embedding["data_unit"]
        new_lr, new_du = renaming_map.get(old_lr, old_lr), renaming_map.get(old_du, old_du)
        embedding["label_row"] = new_lr
        embedding["data_unit"] = new_du
        embedding["url"] = embedding["url"].replace(old_du, new_du).replace(old_lr, new_lr)
        return embedding

    collection = load_collections(embedding_type, project_file_structure.embeddings)
    updated_collection = [_update_identifiers(up, renaming_map) for up in collection]
    save_collections(embedding_type, project_file_structure.embeddings, updated_collection)


def update_2d_embedding_identifiers(
    project_file_structure: ProjectFileStructure, embedding_type: EmbeddingType, renaming_map: dict[str, str]
):
    def _update_identifiers(identifier: str):
        old_lr, old_du, *_ = identifier.split("_", 3)
        new_lr, new_du = renaming_map.get(old_lr, old_lr), renaming_map.get(old_du, old_du)
        return identifier.replace(old_du, new_du).replace(old_lr, new_lr)

    embedding_file = project_file_structure.embeddings / EMBEDDING_REDUCED_TO_FILENAME[embedding_type]
    if not embedding_file.is_file():
        return

    embeddings = pickle.loads(embedding_file.read_bytes())
    embeddings["identifier"] = [_update_identifiers(id) for id in embeddings["identifier"]]
    embedding_file.write_bytes(pickle.dumps(embeddings))


def replace_in_files(project_file_structure: ProjectFileStructure, renaming_map):
    for subs in iterate_in_batches(list(renaming_map.items()), 100):
        substitutions = " -e  " + " -e ".join(f"'s/{old}/{new}/g'" for old, new in subs)
        cmd = f" find . -type f \( -iname \*.json -o -iname \*.yaml -o -iname \*.csv \) -exec sed -i '' {substitutions} {{}} +"
        subprocess.run(cmd, shell=True, cwd=project_file_structure.project_dir)


def replace_uids(
    project_file_structure: ProjectFileStructure,
    file_mappings: dict[LabelRowDataUnit, LabelRowDataUnit],
    old_project_hash: str,
    new_project_hash: str,
    dataset_hash: str,
):
    label_row_meta = json.loads(project_file_structure.label_row_meta.read_text(encoding="utf-8"))

    renaming_map = {old_project_hash: new_project_hash}
    for x in label_row_meta.values():
        renaming_map[x["dataset_hash"]] = dataset_hash

    for (old_lr, old_du), (new_lr, new_du) in file_mappings.items():
        renaming_map[old_lr], renaming_map[old_du] = new_lr, new_du

    try:
        _replace_uids(project_file_structure, file_mappings, renaming_map)
    except Exception as e:
        rev_renaming_map = {v: k for k, v in renaming_map.items()}
        _replace_uids(project_file_structure, {v: k for k, v in file_mappings.items()}, rev_renaming_map)
        raise Exception("UID replacement failed")


def _replace_uids(project_file_structure: ProjectFileStructure, file_mappings: dict[LabelRowDataUnit, LabelRowDataUnit], renaming_map: dict[str, str]):
    rename_files(project_file_structure, file_mappings)
    replace_in_files(project_file_structure, renaming_map)
    original_project_dir = DBConnection.set_project_path(project_file_structure.project_dir)
    MergedMetrics().replace_identifiers(renaming_map)
    DBConnection.set_project_path(original_project_dir)
    for embedding_type in [EmbeddingType.IMAGE, EmbeddingType.CLASSIFICATION, EmbeddingType.OBJECT]:
        update_embedding_identifiers(project_file_structure, embedding_type, renaming_map)
        update_2d_embedding_identifiers(project_file_structure, embedding_type, renaming_map)


def create_filtered_embeddings(
    curr_project_structure: ProjectFileStructure,
    target_project_structure: ProjectFileStructure,
    filtered_label_rows: set[str],
    filtered_data_hashes: set[str],
    filtered_df: pd.DataFrame,
):
    target_project_structure.embeddings.mkdir(parents=True, exist_ok=True)
    for csv_embedding_file in curr_project_structure.embeddings.glob("*.csv"):
        csv_df = pd.read_csv(csv_embedding_file, index_col=0)
        filtered_csv_df = csv_df[csv_df.index.isin(filtered_df.identifier)]
        filtered_csv_df.to_csv(target_project_structure.embeddings / csv_embedding_file.name)
    for embedding_type in [EmbeddingType.IMAGE, EmbeddingType.CLASSIFICATION, EmbeddingType.OBJECT]:
        collection = load_collections(embedding_type, curr_project_structure.embeddings)
        collection = [
            c for c in collection if c["label_row"] in filtered_label_rows and c["data_unit"] in filtered_data_hashes
        ]
        save_collections(embedding_type, target_project_structure.embeddings, collection)

    for embedding_type in [EmbeddingType.IMAGE, EmbeddingType.CLASSIFICATION, EmbeddingType.OBJECT]:
        embedding_file_name = EMBEDDING_REDUCED_TO_FILENAME[embedding_type]
        if not Path(curr_project_structure.embeddings / embedding_file_name).exists():
            continue
        embeddings = pickle.loads(Path(curr_project_structure.embeddings / embedding_file_name).read_bytes())
        embeddings_df = pd.DataFrame.from_dict(embeddings)
        embeddings_df = embeddings_df[embeddings_df["identifier"].isin(filtered_df.identifier)]
        filtered_embeddings = embeddings_df.to_dict(orient="list")
        (target_project_structure.embeddings / embedding_file_name).write_bytes(pickle.dumps(filtered_embeddings))


def get_filtered_objects(filtered_labels, label_row_hash, data_unit_hash, objects):
    return [obj for obj in objects if (label_row_hash, data_unit_hash, obj["objectHash"]) in filtered_labels]


def get_filtered_classifications(filtered_labels, label_row_hash, data_unit_hash, classifications):
    return [
        obj for obj in classifications if (label_row_hash, data_unit_hash, obj["classificationHash"]) in filtered_labels
    ]


def copy_filtered_data(
    curr_project_structure: ProjectFileStructure,
    target_project_structure: ProjectFileStructure,
    filtered_label_rows: set[str],
    filtered_data_hashes: set[str],
    filtered_labels: set[tuple[str, str, str]],
):
    target_project_structure.data.mkdir(parents=True, exist_ok=True)
    for label_row_hash in filtered_label_rows:
        if not (curr_project_structure.data / label_row_hash).is_dir():
            continue
        (target_project_structure.data / label_row_hash / "images").mkdir(parents=True, exist_ok=True)
        for curr_file in (curr_project_structure.data / label_row_hash / "images").glob("*.*"):
            curr_data_unit = curr_file.stem.split("_")[0]
            if (
                curr_data_unit in filtered_data_hashes
                and not (target_project_structure.data / label_row_hash / "images" / curr_file.name).exists()
            ):
                (target_project_structure.data / label_row_hash / "images" / curr_file.name).symlink_to(curr_file)

        label_row = json.loads((curr_project_structure.data / label_row_hash / "label_row.json").read_text())
        label_row["data_units"] = {k: v for k, v in label_row["data_units"].items() if k in filtered_data_hashes}

        for data_unit_hash, v in label_row["data_units"].items():
            if "objects" in label_row["data_units"][data_unit_hash]["labels"]:
                label_row["data_units"][data_unit_hash]["labels"]["objects"] = get_filtered_objects(
                    filtered_labels, label_row_hash, data_unit_hash, v["labels"]["objects"]
                )
                label_row["data_units"][data_unit_hash]["labels"]["classifications"] = get_filtered_classifications(
                    filtered_labels, label_row_hash, data_unit_hash, v["labels"]["classifications"]
                )
                continue
            for label_no, label_item in label_row["data_units"][data_unit_hash]["labels"].items():
                label_row["data_units"][data_unit_hash]["labels"][label_no]["objects"] = get_filtered_objects(
                    filtered_labels, label_row_hash, data_unit_hash, v["labels"][label_no]["objects"]
                )
                label_row["data_units"][data_unit_hash]["labels"][label_no][
                    "classifications"
                ] = get_filtered_classifications(
                    filtered_labels, label_row_hash, data_unit_hash, v["labels"][label_no]["classifications"]
                )

        filtered_label_hashes = {f[2] for f in filtered_labels}
        label_row["object_answers"] = {
            k: v for k, v in label_row["object_answers"].items() if k in filtered_label_hashes
        }
        label_row["classification_answers"] = {
            k: v for k, v in label_row["classification_answers"].items() if k in filtered_label_hashes
        }
        (target_project_structure.data / label_row_hash / "label_row.json").write_text(json.dumps(label_row))


def create_filtered_db(target_project_dir: Path, filtered_df: pd.DataFrame):
    to_save_df = filtered_df.set_index("identifier")
    curr_project_dir = DBConnection.project_file_structure().project_dir
    DBConnection.set_project_path(target_project_dir)
    MergedMetrics().replace_all(to_save_df)
    DBConnection.set_project_path(curr_project_dir)


def create_filtered_metrics(
    curr_project_structure: ProjectFileStructure,
    target_project_structure: ProjectFileStructure,
    filtered_df: pd.DataFrame,
):
    target_project_structure.metrics.mkdir(parents=True, exist_ok=True)
    for csv_metric_file in curr_project_structure.metrics.glob("*.csv"):
        csv_df = pd.read_csv(csv_metric_file, index_col=0)
        filtered_csv_df = csv_df[csv_df.index.isin(filtered_df.identifier)]
        filtered_csv_df.to_csv(target_project_structure.metrics / csv_metric_file.name)

        metric_json_path = Path(csv_metric_file.as_posix().rsplit(".", 1)[0] + ".meta.json")
        metric_meta = MetricMetadata.parse_file(metric_json_path)
        metric_meta.stats.num_rows = filtered_csv_df.shape[0]
        metric_meta.stats.min_value = float(filtered_csv_df["score"].min())
        metric_meta.stats.max_value = float(filtered_csv_df["score"].max())
        mm_js = metric_meta.json()
        (target_project_structure.metrics / metric_json_path.name).write_text(mm_js)


def copy_project_meta(
    curr_project_structure: ProjectFileStructure,
    target_project_structure: ProjectFileStructure,
    project_title: str,
    project_description: str,
):
    project_meta = fetch_project_meta(curr_project_structure.project_dir)
    project_meta["project_title"] = project_title
    project_meta["project_description"] = project_description
    project_meta["has_remote"] = False
    project_meta["project_hash"] = ""
    target_project_structure.project_meta.write_text(yaml.safe_dump(project_meta))


def copy_image_data_unit_json(
    curr_project_structure: ProjectFileStructure,
    target_project_structure: ProjectFileStructure,
    filtered_data_hashes: set[str],
):
    image_data_unit = json.loads(curr_project_structure.image_data_unit.read_text())
    filtered_image_data_unit = {k: v for k, v in image_data_unit.items() if v["data_hash"] in filtered_data_hashes}
    target_project_structure.image_data_unit.write_text(json.dumps(filtered_image_data_unit))


def copy_label_row_meta_json(
    curr_project_structure: ProjectFileStructure,
    target_project_structure: ProjectFileStructure,
    filtered_label_rows: set[str],
) -> dict:
    label_row_meta = json.loads(curr_project_structure.label_row_meta.read_text())
    filtered_label_row_meta = {k: v for k, v in label_row_meta.items() if k in filtered_label_rows}
    target_project_structure.label_row_meta.write_text(json.dumps(filtered_label_row_meta))
    return filtered_label_row_meta
