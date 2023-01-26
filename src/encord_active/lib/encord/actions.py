import json
import os
import subprocess
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import pandas as pd
from encord import Dataset, EncordUserClient, Project
from encord.constants.enums import DataType
from encord.exceptions import AuthorisationError
from encord.objects.ontology_structure import OntologyStructure
from encord.orm.dataset import StorageLocation
from encord.utilities.label_utilities import construct_answer_dictionaries
from tqdm import tqdm

from encord_active.lib.common.utils import fetch_project_meta
from encord_active.lib.db.merged_metrics import rename_identifiers
from encord_active.lib.embeddings.utils import load_collections, save_collections
from encord_active.lib.metrics.metric import EmbeddingType
from encord_active.lib.project import ProjectFileStructure


class DatasetCreationResult(NamedTuple):
    hash: str
    du_original_mapping: dict[str, dict]
    lr_du_mapping: dict[tuple[str, str], tuple[str, str]]


class ProjectCreationResult(NamedTuple):
    hash: str


class EncordActions:
    def __init__(self, project_dir: Path):
        self.original_project = None
        self.project_meta = fetch_project_meta(project_dir)
        self.project_file_structure = ProjectFileStructure(project_dir)

        try:
            ssh_key_path = Path("/Users/encord/.ssh/id_ed25519")
            self.original_project_hash = self.project_meta["project_hash"]
        except Exception as e:
            raise MissingProjectMetaAttribure(e.args[0], self.project_file_structure.project_meta)

        if not ssh_key_path.is_file():
            raise FileNotFoundError(f"No SSH file in location: {ssh_key_path}")

        self.user_client = EncordUserClient.create_with_ssh_private_key(
            Path(ssh_key_path).expanduser().read_text(encoding="utf-8"),
        )

    def init_original_project(self):
        try:
            self.original_project = self.user_client.get_project(self.original_project_hash)
            if self.original_project.project_hash == self.original_project_hash:
                return False
        except AuthorisationError:
            return False
        return True

    def _upload_item(
        self, dataset, label_row_hash, data_unit_hash, data_unit_hashes, new_du_to_original, uploaded_data
    ) -> Optional[str]:
        label_row_structure = self.project_file_structure.label_row_structure(label_row_hash)
        label_row = json.loads(label_row_structure.label_row_file.expanduser().read_text())
        dataset_hash = dataset.dataset_hash
        uploaded_data.add(data_unit_hash)

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
        filtered_dataset: pd.DataFrame,
        progress_callback: Optional[Callable] = None,
    ):
        datasets_with_same_title = self.user_client.get_datasets(title_eq=dataset_title)
        if len(datasets_with_same_title) > 0:
            raise DatasetUniquenessError(dataset_title)

        new_du_to_original: dict[str, dict] = {}
        lrdu_mapping: dict[tuple[str, str], tuple[str, str]] = {}

        self.user_client.create_dataset(
            dataset_title=dataset_title,
            dataset_type=StorageLocation.CORD_STORAGE,
            dataset_description=dataset_description,
        )
        dataset_hash: str = self.user_client.get_datasets(title_eq=dataset_title)[0]["dataset"].dataset_hash
        dataset: Dataset = self.user_client.get_dataset(dataset_hash)

        # The following operation is for image groups (to upload them efficiently)
        label_hash_to_data_units: dict[str, set] = {}
        for _, item in tqdm(filtered_dataset.iterrows(), total=filtered_dataset.shape[0]):
            label_row_hash, data_unit_hash, *_ = str(item["identifier"]).split("_")
            label_hash_to_data_units.setdefault(label_row_hash, set()).add(data_unit_hash)

        uploaded_data_units: set = set()
        for label_row_hash, data_hashes in label_hash_to_data_units.items():
            for data_unit_hash in data_hashes:
                if data_unit_hash not in uploaded_data_units:
                    # Since create_image_group does not return info related to the uploaded images, we should find its
                    # data_hash in a hacky way
                    new_data_unit_hash = self._upload_item(
                        dataset, label_row_hash, data_unit_hash, data_hashes, new_du_to_original, uploaded_data_units
                    )
                    if not new_data_unit_hash:
                        raise Exception("Data unit upload failed")

                    _update_mapping(new_data_unit_hash, label_row_hash, data_unit_hash, new_du_to_original)
                    uploaded_data_units.add(data_unit_hash)
                    lrdu_mapping[(label_row_hash, data_unit_hash)] = ("", new_data_unit_hash)

                if progress_callback:
                    progress_callback(len(uploaded_data_units) / filtered_dataset.shape[0])
        return DatasetCreationResult(dataset_hash, new_du_to_original, lrdu_mapping)

    def create_ontology(self, title):
        ontology_d = json.loads(self.project_file_structure.ontology.read_text(encoding="utf-8"))
        ontology_structure = OntologyStructure.from_dict(ontology_d)
        return self.user_client.create_ontology(title, structure=ontology_structure)

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

        all_new_label_rows = new_project.label_rows
        for counter, new_label_row in enumerate(all_new_label_rows):
            initiated_label_row: dict = new_project.create_label_row(new_label_row["data_hash"])
            original_data = dataset_creation_result.du_original_mapping[new_label_row["data_hash"]]

            new_label_row_hash = initiated_label_row["label_hash"]
            new_data_unit_hash = dataset_creation_result.lr_du_mapping[
                (original_data["label_row_hash"], original_data["data_unit_hash"])
            ][1]
            dataset_creation_result.lr_du_mapping[
                (original_data["label_row_hash"], original_data["data_unit_hash"])
            ] = (new_label_row_hash, new_data_unit_hash)
            original_label_row = json.loads(
                self.project_file_structure.label_row_structure(
                    original_data["label_row_hash"]
                ).label_row_file.read_text(
                    encoding="utf-8",
                )
            )

            if initiated_label_row["data_type"] in [DataType.IMAGE.value, DataType.VIDEO.value]:
                original_labels = original_label_row["data_units"][original_data["data_unit_hash"]]["labels"]
                initiated_label_row["data_units"][new_label_row["data_hash"]]["labels"] = original_labels
                initiated_label_row["object_answers"] = original_label_row["object_answers"]
                initiated_label_row["classification_answers"] = original_label_row["classification_answers"]

                if original_labels != {}:
                    initiated_label_row = construct_answer_dictionaries(initiated_label_row)
                    new_project.save_label_row(initiated_label_row["label_hash"], initiated_label_row)

            elif initiated_label_row["data_type"] == DataType.IMG_GROUP.value:
                object_hashes: set = set()
                classification_hashes: set = set()

                # Currently img_groups are matched using data_title, it should be fixed after SDK update
                for data_unit in initiated_label_row["data_units"].values():
                    for original_data in original_label_row["data_units"].values():
                        if original_data["data_hash"] == data_unit["data_title"].split(".")[0]:
                            data_unit["labels"] = original_data["labels"]
                            for obj in data_unit["labels"].get("objects", []):
                                object_hashes.add(obj["objectHash"])
                            for classification in data_unit["labels"].get("classifications", []):
                                classification_hashes.add(classification["classificationHash"])

                initiated_label_row["object_answers"] = original_label_row["object_answers"]
                initiated_label_row["classification_answers"] = original_label_row["classification_answers"]

                # Remove unused object/classification answers
                for object_hash in object_hashes:
                    initiated_label_row["object_answers"].pop(object_hash)

                for classification_hash in classification_hashes:
                    initiated_label_row["classification_answers"].pop(classification_hash)

                initiated_label_row = construct_answer_dictionaries(initiated_label_row)
                new_project.save_label_row(initiated_label_row["label_hash"], initiated_label_row)

                # remove unused object and classification answers

            if progress_callback:
                progress_callback((counter + 1) / len(all_new_label_rows))

        return new_project

    def fix_pickle_files(self, embedding_type: EmbeddingType, renaming_map):
        def fix_pickle_file(up, renaming_map):
            up["label_row"] = renaming_map[up["label_row"]]
            up["data_unit"] = renaming_map[up["data_unit"]]
            url_without_extension, extension = up["url"].split(".")
            changed_parts = [renaming_map[x] if x in renaming_map else x for x in url_without_extension.split("/")]
            up["url"] = "/".join(changed_parts) + "." + extension
            return up

        collection = load_collections(embedding_type, self.project_file_structure.embeddings)
        updated_collection = [fix_pickle_file(up, renaming_map) for up in collection]
        save_collections(embedding_type, self.project_file_structure.embeddings, updated_collection)

    def replace_uids(self, file_mappings, project_hash):
        renaming_map = {self.original_project_hash: project_hash}

        for (old_lr, old_du), (new_lr, new_du) in file_mappings.items():
            os.rename(
                (self.project_file_structure.data / old_lr / "images" / old_du).as_posix() + ".jpg",
                (self.project_file_structure.data / old_lr / "images" / new_du).as_posix() + ".jpg",
            )
            renaming_map[old_lr], renaming_map[old_du] = new_lr, new_du

        dir_renames = {old_lr: new_lr for (old_lr, old_du), (new_lr, new_du) in file_mappings.items()}

        for (old_lr, new_lr) in dir_renames.items():
            os.rename(self.project_file_structure.data / old_lr, self.project_file_structure.data / new_lr)

        rename_identifiers(renaming_map)
        for o, n in renaming_map.items():
            cmd = f" find . -type f \( -iname \*.json -o -iname \*.yaml -o -iname \*.csv \) -exec sed -i '' 's/{o}/{n}/g' {{}} +"
            subprocess.run(cmd, shell=True, cwd=self.project_file_structure.project_dir)

        for embedding_type in [EmbeddingType.IMAGE, EmbeddingType.CLASSIFICATION, EmbeddingType.OBJECT]:
            self.fix_pickle_files(embedding_type, renaming_map)


def _find_new_row_hash(user_client: EncordUserClient, new_dataset_hash: str, out_mapping: dict) -> Optional[str]:
    updated_dataset = user_client.get_dataset(new_dataset_hash)
    for new_data_row in updated_dataset.data_rows:
        if new_data_row["data_hash"] not in out_mapping:
            return new_data_row["data_hash"]
    return None


def _update_mapping(new_data_hash: str, label_row_hash: str, data_unit_hash: str, out_mapping: dict):
    out_mapping[new_data_hash] = {
        "label_row_hash": label_row_hash,
        "data_unit_hash": data_unit_hash,
    }


class MissingProjectMetaAttribure(Exception):
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
