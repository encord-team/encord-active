import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import pandas as pd
import yaml
from encord import Dataset, EncordUserClient, Project
from encord.constants.enums import DataType
from encord.exceptions import AuthorisationError
from encord.objects.ontology_structure import OntologyStructure
from encord.orm.dataset import StorageLocation
from encord.orm.label_row import LabelRow
from encord.orm.project import (
    CopyDatasetAction,
    CopyDatasetOptions,
    CopyLabelsOptions,
    ReviewApprovalState,
)
from encord.utilities.label_utilities import construct_answer_dictionaries
from tqdm import tqdm

from encord_active.app.common.state import get_state
from encord_active.lib.common.utils import (
    fetch_project_meta,
    iterate_in_batches,
    update_project_meta,
)
from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.embeddings.utils import (
    LabelEmbedding,
    load_collections,
    save_collections,
)
from encord_active.lib.encord.utils import get_client
from encord_active.lib.metrics.metric import EmbeddingType, MetricMetadata
from encord_active.lib.project import ProjectFileStructure
from encord_active.lib.project.metadata import fetch_project_meta, update_project_meta


class LabelRowDataUnit(NamedTuple):
    label_row: str
    data_unit: str


class DatasetCreationResult(NamedTuple):
    hash: str
    du_original_mapping: dict[str, LabelRowDataUnit]
    lr_du_mapping: dict[LabelRowDataUnit, LabelRowDataUnit]


class ProjectCreationResult(NamedTuple):
    hash: str


def create_filtered_embeddings(
    curr_project_structure: ProjectFileStructure,
    target_project_structure: ProjectFileStructure,
    filtered_label_rows: set[str],
    filtered_data_hashes: set[str],
    filtered_df: pd.DataFrame,
):
    target_project_structure.embeddings.mkdir()
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


def copy_filtered_data(
    curr_project_structure: ProjectFileStructure,
    target_project_structure: ProjectFileStructure,
    filtered_label_rows: set[str],
    filtered_data_hashes: set[str],
    filtered_labels: set[tuple[str, str, str]],
):
    target_project_structure.data.mkdir(parents=True, exist_ok=True)
    for label_row_hash, data_unit in zip(filtered_label_rows, filtered_data_hashes):
        if not (curr_project_structure.data / label_row_hash).is_dir():
            continue
        (target_project_structure.data / label_row_hash / "images").mkdir(parents=True, exist_ok=True)
        for curr_file in (curr_project_structure.data / label_row_hash / "images").glob(f"{data_unit}.*"):
            if not (target_project_structure.data / label_row_hash / "images" / curr_file.name).exists():
                (target_project_structure.data / label_row_hash / "images" / curr_file.name).symlink_to(curr_file)

        label_row = json.loads((curr_project_structure.data / label_row_hash / "label_row.json").read_text())
        label_row["data_units"] = {k: v for k, v in label_row["data_units"].items() if k in filtered_data_hashes}
        for data_unit_hash, v in label_row["data_units"].items():
            label_row["data_units"][data_unit_hash]["labels"]["objects"] = [
                obj
                for obj in v["labels"]["objects"]
                if (label_row_hash, data_unit_hash, obj["objectHash"]) in filtered_labels
            ]

            label_row["data_units"][data_unit_hash]["labels"]["classifications"] = [
                obj
                for obj in v["labels"]["classifications"]
                if (label_row_hash, data_unit_hash, obj["classificationHash"]) in filtered_labels
            ]

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
    return curr_project_dir


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


class EncordActions:
    def __init__(self, project_dir: Path, fallback_ssh_key_path: Optional[Path] = None):
        self._original_project = None
        self.project_meta = fetch_project_meta(project_dir)
        self.project_file_structure = ProjectFileStructure(project_dir)

        try:
            original_project_hash = self.project_meta["project_hash"]
        except Exception as e:
            raise MissingProjectMetaAttribute(e.args[0], self.project_file_structure.project_meta)

        try:
            ssh_key_path = Path(self.project_meta["ssh_key_path"]).resolve()
        except Exception as e:
            if not fallback_ssh_key_path:
                raise MissingProjectMetaAttribute(e.args[0], self.project_file_structure.project_meta)
            ssh_key_path = fallback_ssh_key_path

        if not ssh_key_path.is_file():
            raise FileNotFoundError(f"No SSH file in location: {ssh_key_path}")

        self.user_client = get_client(ssh_key_path.expanduser())

    @property
    def original_project(self):
        if self._original_project:
            return self._original_project

        try:
            project = self.user_client.get_project(self.project_meta["project_hash"])
            if project.project_hash == self.project_meta["project_hash"]:
                self._original_project = project
        except AuthorisationError:
            pass

        return self._original_project

    def _upload_item(
        self,
        dataset: Dataset,
        label_row_hash: str,
        data_unit_hash: str,
        data_unit_hashes: set[str],
        new_du_to_original: dict[str, LabelRowDataUnit],
    ) -> Optional[str]:
        label_row_structure = self.project_file_structure.label_row_structure(label_row_hash)
        label_row = json.loads(label_row_structure.label_row_file.expanduser().read_text())
        dataset_hash = dataset.dataset_hash

        if label_row["data_type"] == DataType.IMAGE.value:
            image_path = list(label_row_structure.images_dir.glob(f"{data_unit_hash}.*"))[0]
            return dataset.upload_image(
                file_path=image_path, title=label_row["data_units"][data_unit_hash]["data_title"]
            )["data_hash"]

        elif label_row["data_type"] == DataType.IMG_GROUP.value:
            image_paths = []
            image_names = []
            if len(data_unit_hashes) > 0:
                for data_unit_hash in data_unit_hashes:
                    img_path = list(label_row_structure.images_dir.glob(f"{data_unit_hash}.*"))[0]
                    image_paths.append(img_path.as_posix())
                    image_names.append(img_path.name)
                # Unfortunately the following function does not return metadata related to the uploaded items
                dataset.create_image_group(file_paths=image_paths, title=label_row["data_title"])
            return _find_new_row_hash(self.user_client, dataset_hash, new_du_to_original)

        elif label_row["data_type"] == DataType.VIDEO.value:
            video_path = list(label_row_structure.images_dir.glob(f"{data_unit_hash}.*"))[0].as_posix()

            # Unfortunately the following function does not return metadata related to the uploaded items
            dataset.upload_video(file_path=video_path, title=label_row["data_units"][data_unit_hash]["data_title"])
            return _find_new_row_hash(self.user_client, dataset_hash, new_du_to_original)

        else:
            raise Exception(f'Undefined data type {label_row["data_type"]} for label_row={label_row["label_hash"]}')

    def create_dataset(
        self,
        dataset_title: str,
        dataset_description: str,
        dataset_df: pd.DataFrame,
        progress_callback: Optional[Callable] = None,
    ):
        datasets_with_same_title = self.user_client.get_datasets(title_eq=dataset_title)
        if len(datasets_with_same_title) > 0:
            raise DatasetUniquenessError(dataset_title)

        new_du_to_original: dict[str, LabelRowDataUnit] = {}
        lrdu_mapping: dict[LabelRowDataUnit, LabelRowDataUnit] = {}

        self.user_client.create_dataset(
            dataset_title=dataset_title,
            dataset_type=StorageLocation.CORD_STORAGE,
            dataset_description=dataset_description,
        )
        dataset_hash: str = self.user_client.get_datasets(title_eq=dataset_title)[0]["dataset"].dataset_hash
        dataset: Dataset = self.user_client.get_dataset(dataset_hash)

        # The following operation is for image groups (to upload them efficiently)
        label_hash_to_data_units: dict[str, set] = {}
        for identifier, item in tqdm(dataset_df.iterrows(), total=dataset_df.shape[0]):
            label_row_hash, data_unit_hash, *_ = str(identifier).split("_")
            label_hash_to_data_units.setdefault(label_row_hash, set()).add(data_unit_hash)

        uploaded_data_units: set[str] = set()
        for label_row_hash, data_hashes in label_hash_to_data_units.items():
            for data_unit_hash in data_hashes:
                if data_unit_hash not in uploaded_data_units:
                    # Since create_image_group does not return info related to the uploaded images, we should find its
                    # data_hash in a hacky way
                    new_data_unit_hash = self._upload_item(
                        dataset, label_row_hash, data_unit_hash, data_hashes, new_du_to_original
                    )
                    uploaded_data_units.add(data_unit_hash)
                    if not new_data_unit_hash:
                        raise Exception("Data unit upload failed")

                    new_du_to_original[new_data_unit_hash] = LabelRowDataUnit(label_row_hash, data_unit_hash)
                    lrdu_mapping[LabelRowDataUnit(label_row_hash, data_unit_hash)] = LabelRowDataUnit(
                        "", new_data_unit_hash
                    )
                    uploaded_data_units.add(data_unit_hash)

                if progress_callback:
                    progress_callback(len(uploaded_data_units) / dataset_df.shape[0])
        return DatasetCreationResult(dataset_hash, new_du_to_original, lrdu_mapping)

    def create_ontology(self, title: str, description: str):
        ontology_dict = json.loads(self.project_file_structure.ontology.read_text(encoding="utf-8"))
        ontology_structure = OntologyStructure.from_dict(ontology_dict)
        return self.user_client.create_ontology(title, structure=ontology_structure, description=description)

    @staticmethod
    def prepare_label_row(
        original_label_row: LabelRow, new_label_row: LabelRow, new_label_row_data_unit_hash: str, original_du: str
    ) -> LabelRow:
        if new_label_row["data_type"] in [DataType.IMAGE.value, DataType.VIDEO.value]:
            original_labels = original_label_row["data_units"][original_du]["labels"]
            new_label_row["data_units"][new_label_row_data_unit_hash]["labels"] = original_labels
            new_label_row["object_answers"] = original_label_row["object_answers"]
            new_label_row["classification_answers"] = original_label_row["classification_answers"]

        elif new_label_row["data_type"] == DataType.IMG_GROUP.value:
            object_hashes: set[str] = set()
            classification_hashes: set[str] = set()

            # Currently img_groups are matched using data_title, it should be fixed after SDK update
            for data_unit in new_label_row["data_units"].values():
                for original_data in original_label_row["data_units"].values():
                    if original_data["data_hash"] == data_unit["data_title"].split(".")[0]:
                        data_unit["labels"] = original_data["labels"]
                        for obj in data_unit["labels"].get("objects", []):
                            object_hashes.add(obj["objectHash"])
                        for classification in data_unit["labels"].get("classifications", []):
                            classification_hashes.add(classification["classificationHash"])

            new_label_row["object_answers"] = original_label_row["object_answers"]
            new_label_row["classification_answers"] = original_label_row["classification_answers"]

            # Remove unused object/classification answers
            for object_hash in object_hashes:
                new_label_row["object_answers"].pop(object_hash)

            for classification_hash in classification_hashes:
                new_label_row["classification_answers"].pop(classification_hash)
        return construct_answer_dictionaries(new_label_row)

    def create_project(
        self,
        dataset_creation_result: DatasetCreationResult,
        project_title: str,
        project_description: str,
        ontology_hash: str,
        progress_callback: Optional[Callable] = None,
    ):
        new_project_hash: str = self.user_client.create_project(
            project_title=project_title,
            dataset_hashes=[dataset_creation_result.hash],
            project_description=project_description,
            ontology_hash=ontology_hash,
        )

        new_project: Project = self.user_client.get_project(new_project_hash)

        # Copy labels from old project to new project
        # Three things to copy: labels, object_answers, classification_answers

        for counter, new_label_row_metadata in enumerate(new_project.label_rows):
            new_label_data_unit_hash = new_label_row_metadata["data_hash"]
            new_label_row = new_project.create_label_row(new_label_data_unit_hash)
            original_lr_du = dataset_creation_result.du_original_mapping[new_label_data_unit_hash]

            dataset_creation_result.lr_du_mapping[original_lr_du] = LabelRowDataUnit(
                new_label_row["label_hash"], new_label_data_unit_hash
            )
            original_label_row = json.loads(
                self.project_file_structure.label_row_structure(original_lr_du.label_row).label_row_file.read_text(
                    encoding="utf-8",
                )
            )
            label_row = self.prepare_label_row(
                original_label_row, new_label_row, new_label_data_unit_hash, original_lr_du.data_unit
            )
            if any(
                data_unit["labels"].get("objects", []) or data_unit["labels"].get("classifications", [])
                for data_unit in label_row["data_units"].values()
            ):
                new_project.save_label_row(label_row["label_hash"], label_row)

            if progress_callback:
                progress_callback((counter + 1) / len(new_project.label_rows))
        self.project_meta["has_remote"] = True
        update_project_meta(self.project_file_structure.project_dir, self.project_meta)
        return new_project

    def update_embedding_identifiers(self, embedding_type: EmbeddingType, renaming_map: dict[str, str]):
        def _update_identifiers(embedding: LabelEmbedding, renaming_map: dict[str, str]):
            old_lr, old_du = embedding["label_row"], embedding["data_unit"]
            new_lr, new_du = renaming_map.get(old_lr, old_lr), renaming_map.get(old_du, old_du)
            embedding["label_row"] = new_lr
            embedding["data_unit"] = new_du
            embedding["url"] = embedding["url"].replace(old_du, new_du).replace(old_lr, new_lr)
            return embedding

        collection = load_collections(embedding_type, self.project_file_structure.embeddings)
        updated_collection = [_update_identifiers(up, renaming_map) for up in collection]
        save_collections(embedding_type, self.project_file_structure.embeddings, updated_collection)

    def _rename_files(self, file_mappings: dict[LabelRowDataUnit, LabelRowDataUnit]):
        for (old_lr, old_du), (new_lr, new_du) in file_mappings.items():
            old_lr_path = self.project_file_structure.data / old_lr
            new_lr_path = self.project_file_structure.data / new_lr
            if old_lr_path.exists() and not new_lr_path.exists():
                old_lr_path.rename(new_lr_path)
            for old_du_f in new_lr_path.glob(f"**/{old_du}.*"):
                old_du_f.rename(new_lr_path / "images" / f"{new_du}.{old_du_f.suffix}")

    def _replace_in_files(self, renaming_map):
        for subs in iterate_in_batches(list(renaming_map.items()), 100):
            substitutions = ";".join(f"s/{old}/{new}/g" for old, new in subs)
            cmd = f" find . -type f \( -iname \*.json -o -iname \*.yaml -o -iname \*.csv \) -exec sed -i '' '{substitutions}' {{}} +"
            subprocess.run(cmd, shell=True, cwd=self.project_file_structure.project_dir)

    def replace_uids(
        self, file_mappings: dict[LabelRowDataUnit, LabelRowDataUnit], project_hash: str, dataset_hash: str
    ):
        label_row_meta = json.loads(self.project_file_structure.label_row_meta.read_text(encoding="utf-8"))
        original_dataset_hash = next(iter(label_row_meta.values()))["dataset_hash"]
        renaming_map = {self.project_meta["project_hash"]: project_hash, original_dataset_hash: dataset_hash}
        for (old_lr, old_du), (new_lr, new_du) in file_mappings.items():
            renaming_map[old_lr], renaming_map[old_du] = new_lr, new_du

        try:
            self._rename_files(file_mappings)
            self._replace_in_files(renaming_map)
            MergedMetrics().replace_identifiers(renaming_map)
            for embedding_type in [EmbeddingType.IMAGE, EmbeddingType.CLASSIFICATION, EmbeddingType.OBJECT]:
                self.update_embedding_identifiers(embedding_type, renaming_map)
        except Exception as e:
            rev_renaming_map = {v: k for k, v in renaming_map.items()}
            self._rename_files({v: k for k, v in file_mappings.items()})
            self._replace_in_files(rev_renaming_map)
            MergedMetrics().replace_identifiers(rev_renaming_map)
            for embedding_type in [EmbeddingType.IMAGE, EmbeddingType.CLASSIFICATION, EmbeddingType.OBJECT]:
                self.update_embedding_identifiers(embedding_type, rev_renaming_map)
            raise Exception("UID replacement failed")

    def create_subset(
        self,
        filtered_df: pd.DataFrame,
        project_title: str,
        project_description: str,
        dataset_title: Optional[str] = None,
        dataset_description: Optional[str] = None,
        remote_copy: bool = False,
    ) -> Path:
        curr_project_structure = get_state().project_paths
        target_project_dir = curr_project_structure.project_dir.parent / project_title
        target_project_structure = ProjectFileStructure(target_project_dir)

        if target_project_dir.exists():
            raise Exception("Subset with the same title already exists")
        target_project_dir.mkdir()

        ids_df = filtered_df["identifier"].str.split("_", n=4, expand=True)
        filtered_lr_du = {LabelRowDataUnit(label_row, data_unit) for label_row, data_unit in zip(ids_df[0], ids_df[1])}
        filtered_label_rows = {lr_du.label_row for lr_du in filtered_lr_du}
        filtered_data_hashes = {lr_du.data_unit for lr_du in filtered_lr_du}
        filtered_labels = {(ids[1][0], ids[1][1], ids[1][3]) for ids in ids_df.iterrows()}

        curr_project_dir = create_filtered_db(target_project_dir, filtered_df)

        if curr_project_structure.image_data_unit.exists():
            image_data_unit = json.loads(curr_project_structure.image_data_unit.read_text())
            filtered_image_data_unit = {
                k: v for k, v in image_data_unit.items() if v["data_hash"] in filtered_data_hashes
            }
            target_project_structure.image_data_unit.write_text(json.dumps(filtered_image_data_unit))

        label_row_meta = json.loads(curr_project_structure.label_row_meta.read_text())
        filtered_label_row_meta = {k: v for k, v in label_row_meta.items() if k in filtered_label_rows}
        target_project_structure.label_row_meta.write_text(json.dumps(filtered_label_row_meta))

        dataset_hash_map: dict[str, set[str]] = {}
        label_rows = set()
        for k, v in filtered_label_row_meta.items():
            dataset_hash_map.setdefault(v["dataset_hash"], set()).add(v["data_hash"])
            label_rows.add(k)

        shutil.copy2(curr_project_structure.ontology, target_project_structure.ontology)

        project_meta = fetch_project_meta(curr_project_structure.project_dir)
        project_meta["project_title"] = project_title
        project_meta["project_description"] = project_description
        project_meta["has_remote"] = False
        project_meta["project_hash"] = ""
        target_project_structure.project_meta.write_text(yaml.safe_dump(project_meta))

        create_filtered_metrics(curr_project_structure, target_project_structure, filtered_df)

        copy_filtered_data(
            curr_project_structure,
            target_project_structure,
            filtered_label_rows,
            filtered_data_hashes,
            filtered_labels,
        )

        create_filtered_embeddings(
            curr_project_structure, target_project_structure, filtered_label_rows, filtered_data_hashes, filtered_df
        )

        if remote_copy:
            self.original_project.copy_project(
                new_title=project_title,
                new_description=project_description,
                copy_collaborators=True,
                copy_datasets=CopyDatasetOptions(
                    action=CopyDatasetAction.CLONE,
                    dataset_title=dataset_title,
                    dataset_description=dataset_description,
                    datasets_to_data_hashes_map={k: list(v) for k, v in dataset_hash_map.items()},
                ),
                copy_labels=CopyLabelsOptions(
                    accepted_label_statuses=[state for state in ReviewApprovalState],
                    accepted_label_hashes=list(label_rows),
                ),
            )
        return target_project_structure.project_dir


def _find_new_row_hash(user_client: EncordUserClient, new_dataset_hash: str, out_mapping: dict) -> Optional[str]:
    updated_dataset = user_client.get_dataset(new_dataset_hash)
    for new_data_row in updated_dataset.data_rows:
        if new_data_row["data_hash"] not in out_mapping:
            return new_data_row["data_hash"]
    return None


class MissingProjectMetaAttribute(Exception):
    """Exception raised when project metadata doesn't contain an attribute

    Attributes:
        project_dir -- path to a project directory
    """

    def __init__(self, attribute: str, project_meta_file: Path):
        self.attribute = attribute
        self.project_meta_file = project_meta_file
        super().__init__(
            f"`{attribute}` not specified in the project meta data file `{project_meta_file.resolve().as_posix()}`"
        )


class DatasetUniquenessError(Exception):
    """Exception raised when a dataset with the same title already exists"""

    def __init__(self, dataset_title: str):
        self.dataset_title = dataset_title
        super().__init__(
            f"Dataset title '{dataset_title}' already exists in your list of datasets at Encord. Please use a different title."
        )
