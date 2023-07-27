import json
import logging
import pickle
import subprocess
import uuid
from pathlib import Path
from typing import NamedTuple, Optional

import pandas as pd
import yaml

from encord_active.lib.common.data_utils import iterate_in_batches, url_to_file_path
from encord_active.lib.common.utils import DataHashMapping
from encord_active.lib.db.connection import DBConnection, PrismaConnection
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.embeddings.dimensionality_reduction import (
    generate_2d_embedding_data,
)
from encord_active.lib.embeddings.embedding_index import EmbeddingIndex
from encord_active.lib.embeddings.types import LabelEmbedding
from encord_active.lib.embeddings.utils import (
    load_label_embeddings,
    save_label_embeddings,
)
from encord_active.lib.metrics.metric import MetricMetadata
from encord_active.lib.metrics.types import EmbeddingType
from encord_active.lib.metrics.utils import load_metric_dataframe
from encord_active.lib.model_predictions.reader import (
    get_model_predictions,
    read_prediction_files,
)
from encord_active.lib.project import ProjectFileStructure
from encord_active.lib.project.metadata import fetch_project_meta
from encord_active.server.utils import (
    filtered_merged_metrics,
    get_similarity_finder,
    load_project_metrics,
    read_class_idx,
)


class LabelRowDataUnit(NamedTuple):
    label_row: str
    data_unit: str


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

    label_embeddings = load_label_embeddings(embedding_type, project_file_structure)
    updated_label_embeddings = [_update_identifiers(up, renaming_map) for up in label_embeddings]
    save_label_embeddings(embedding_type, project_file_structure, updated_label_embeddings)


def update_2d_embedding_identifiers(
    project_file_structure: ProjectFileStructure, embedding_type: EmbeddingType, renaming_map: dict[str, str]
):
    def _update_identifiers(identifier: str):
        old_lr, old_du, *_ = identifier.split("_", 3)
        new_lr, new_du = renaming_map.get(old_lr, old_lr), renaming_map.get(old_du, old_du)
        return identifier.replace(old_du, new_du).replace(old_lr, new_lr)

    embedding_file = project_file_structure.get_embeddings_file(embedding_type, reduced=True)
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
    data_hash_mapping: DataHashMapping,
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

    for old_dh, new_dh in data_hash_mapping.items():
        renaming_map[old_dh] = new_dh

    logging.debug(f"renaming map for execution: {renaming_map}")
    try:
        _replace_uids(project_file_structure, renaming_map)
    except Exception as e:
        rev_renaming_map = {v: k for k, v in renaming_map.items()}
        _replace_uids(project_file_structure, rev_renaming_map)
        raise Exception("UID replacement failed")


def _replace_uids(
    project_file_structure: ProjectFileStructure,
    renaming_map: dict[str, str],
):
    original_mappings = {}
    if project_file_structure.mappings.is_file():
        original_mappings = json.loads(project_file_structure.mappings.read_text())

    replace_in_files(project_file_structure, renaming_map)
    with DBConnection(project_file_structure) as conn:
        MergedMetrics(conn).replace_identifiers(renaming_map)
    for embedding_type in [EmbeddingType.IMAGE, EmbeddingType.CLASSIFICATION, EmbeddingType.OBJECT]:
        update_embedding_identifiers(project_file_structure, embedding_type, renaming_map)
        update_2d_embedding_identifiers(project_file_structure, embedding_type, renaming_map)

    if original_mappings:
        new_mappings = {renaming_map[k]: v for k, v in original_mappings.items()}
    else:
        new_mappings = {v: k for k, v in renaming_map.items()}
    project_file_structure.mappings.write_text(json.dumps(new_mappings))

    # Invalidate caches - all caches to do with this project are out of date!
    caches = [
        load_metric_dataframe,
        load_project_metrics,
        get_similarity_finder,
        filtered_merged_metrics,
        read_class_idx,
        get_model_predictions,
        read_prediction_files,
    ]
    for cache in caches:
        cache.cache_clear()  # type: ignore
    project_file_structure.cache_clear()


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
        label_embeddings = load_label_embeddings(embedding_type, curr_project_structure)
        label_embeddings = [
            le
            for le in label_embeddings
            if le["label_row"] in filtered_label_rows and le["data_unit"] in filtered_data_hashes
        ]
        save_label_embeddings(embedding_type, target_project_structure, label_embeddings)

    for embedding_type in [EmbeddingType.IMAGE, EmbeddingType.CLASSIFICATION, EmbeddingType.OBJECT]:
        curr_embedding_file = curr_project_structure.get_embeddings_file(embedding_type)
        if not curr_embedding_file.exists():
            continue
        embeddings = pickle.loads(curr_embedding_file.read_bytes())
        if embeddings:
            embeddings_df = pd.DataFrame.from_dict(embeddings)
            if embedding_type == EmbeddingType.IMAGE:
                label_row_du_hashes = filtered_df.identifier.str.slice(stop=73)
                embeddings_df = embeddings_df[
                    embeddings_df[["label_row", "data_unit"]].agg("_".join, axis=1).isin(label_row_du_hashes)
                ]
            else:
                label_hashes = filtered_df.identifier.str.split("_").str.get(3).dropna()
                embeddings_df = embeddings_df[embeddings_df["labelHash"].isin(label_hashes)]
            filtered_embeddings = embeddings_df.to_dict(orient="records")
            target_project_structure.get_embeddings_file(embedding_type).write_bytes(pickle.dumps(filtered_embeddings))
            EmbeddingIndex.from_project(target_project_structure, embedding_type)
            generate_2d_embedding_data(embedding_type, target_project_structure, embeddings)


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
) -> None:
    if curr_project_structure.mappings.is_file():
        hash_mappings = json.loads(curr_project_structure.mappings.read_text())
        filtered_hash_mappings = {
            k: v for k, v in hash_mappings.items() if k in filtered_label_rows or k in filtered_data_hashes
        }
        target_project_structure.mappings.write_text(json.dumps(filtered_hash_mappings))
        target_project_structure = ProjectFileStructure(
            target_project_structure.project_dir
        )  # Recreate object to reload mappings

    local_data_mapping = {}
    label_row_mapping = {}
    for label_row_hash in filtered_label_rows:
        current_label_row_structure = curr_project_structure.label_row_structure(label_row_hash)
        for data_unit in current_label_row_structure.iter_data_unit():
            if data_unit.du_hash in filtered_data_hashes:
                data_unit_file_path = url_to_file_path(data_unit.signed_url, curr_project_structure.project_dir)
                if data_unit_file_path is not None:
                    old_data = data_unit_file_path
                    if not target_project_structure.local_data_store.exists():
                        target_project_structure.local_data_store.mkdir(parents=True, exist_ok=True)

                    # Create local clone if needed.
                    new_file = target_project_structure.local_data_store / old_data.name
                    local_data_mapping[data_unit.signed_url] = new_file.as_uri()
                    new_file.symlink_to(old_data, target_is_directory=False)

        label_row = current_label_row_structure.label_row_json
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
        label_row_mapping[label_row_hash] = label_row

    with PrismaConnection(curr_project_structure) as conn:
        all_label_rows = conn.labelrow.find_many(
            where={
                "label_hash": {"in": list(filtered_label_rows)},
            },
            include={"data_units": {"where": {"data_hash": {"in": list(filtered_data_hashes)}}}},
        )
    with PrismaConnection(target_project_structure) as conn:
        for label_row_db in all_label_rows:
            conn.labelrow.create(
                data={
                    "label_hash": label_row_db.label_hash,
                    "data_units": {
                        "create": [
                            {
                                "data_hash": data_unit.data_hash,
                                "data_title": data_unit.data_title,
                                "frame": data_unit.frame,
                                "data_uri": local_data_mapping.get(data_unit.data_uri, data_unit.data_uri)
                                if data_unit.data_uri is not None
                                else None,
                                "width": data_unit.width,
                                "height": data_unit.height,
                                "fps": data_unit.fps,
                            }
                            for data_unit in (label_row_db.data_units or [])
                        ]
                    },
                    "label_row_json": json.dumps(label_row_mapping[label_row_db.label_hash or ""]),
                    "data_hash": label_row_db.data_hash,
                    "data_title": label_row_db.data_title,
                    "data_type": label_row_db.data_type,
                    "created_at": label_row_db.created_at,
                    "last_edited_at": label_row_db.last_edited_at,
                },
                include={"data_units": True},
            )


def create_filtered_db(target_project_dir: Path, filtered_df: pd.DataFrame):
    to_save_df = filtered_df.set_index("identifier")
    with DBConnection(ProjectFileStructure(target_project_dir)) as conn:
        MergedMetrics(conn).replace_all(to_save_df)


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
    final_data_version: Optional[int] = None,
):
    project_meta = fetch_project_meta(curr_project_structure.project_dir)
    project_meta["project_title"] = project_title
    project_meta["project_description"] = project_description
    project_meta["has_remote"] = False
    project_meta["project_hash"] = str(uuid.uuid4())
    if final_data_version:
        project_meta["data_version"] = final_data_version
    target_project_structure.project_meta.write_text(yaml.safe_dump(project_meta), encoding="utf-8")


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
