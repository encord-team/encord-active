import json
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import pandas as pd
from encord import Dataset, EncordUserClient, Project
from encord.constants.enums import DataType
from encord.exceptions import AuthorisationError
from encord.orm.dataset import Image, StorageLocation
from encord.utilities.label_utilities import construct_answer_dictionaries
from tqdm import tqdm

from encord_active.lib.common.utils import fetch_project_meta
from encord_active.lib.project import ProjectFileStructure


class DatasetCreationResult(NamedTuple):
    hash: str
    du_original_mapping: dict[str, dict]


class ProjectCreationResult(NamedTuple):
    hash: str


class EncordActions:
    def __init__(self, project_dir: Path):
        self.project_meta = fetch_project_meta(project_dir)
        self.project_file_structure = ProjectFileStructure(project_dir)

        try:
            ssh_key_path = Path(self.project_meta["ssh_key_path"]).resolve()
            original_project_hash = self.project_meta["project_hash"]
        except Exception as e:
            raise MissingProjectMetaAttribure(e.args[0], self.project_file_structure.project_meta)

        if not ssh_key_path.is_file():
            raise FileNotFoundError(f"No SSH file in location: {ssh_key_path}")

        self.user_client = EncordUserClient.create_with_ssh_private_key(
            Path(ssh_key_path).expanduser().read_text(encoding="utf-8"),
        )

        self.original_project = self.user_client.get_project(original_project_hash)
        try:
            if self.original_project.project_hash == original_project_hash:
                pass
        except AuthorisationError:
            raise AuthorisationError(
                f'The user associated to the ssh key `{ssh_key_path}` does not have access to the project with project hash `{original_project_hash}`. Run "encord-active config set ssh_key_path /path/to/your/key_file" to set it.'
            )

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
        self.user_client.create_dataset(
            dataset_title=dataset_title,
            dataset_type=StorageLocation.CORD_STORAGE,
            dataset_description=dataset_description,
        )
        dataset_hash: str = self.user_client.get_datasets(title_eq=dataset_title)[0]["dataset"].dataset_hash
        dataset: Dataset = self.user_client.get_dataset(dataset_hash)

        # The following operation is for image groups (to upload them efficiently)
        label_hash_to_data_units: dict[str, list] = {}
        for _, item in tqdm(filtered_dataset.iterrows(), total=filtered_dataset.shape[0]):
            label_row_hash, data_unit_hash, *_ = str(item["identifier"]).split("_")
            label_hash_to_data_units.setdefault(label_row_hash, []).append(data_unit_hash)

        uploaded_label_rows: set = set()
        for counter, (_, item) in enumerate(filtered_dataset.iterrows()):
            label_row_hash, data_unit_hash, *_ = str(item["identifier"]).split("_")
            label_row_structure = self.project_file_structure.label_row_structure(label_row_hash)
            label_row = json.loads(label_row_structure.label_row_file.expanduser().read_text())

            if label_row_hash not in uploaded_label_rows:
                if label_row["data_type"] == DataType.IMAGE.value:
                    image_path = list(label_row_structure.images_dir.glob(f"{data_unit_hash}.*"))[0]
                    uploaded_image: Image = dataset.upload_image(
                        file_path=image_path, title=label_row["data_units"][data_unit_hash]["data_title"]
                    )

                    new_du_to_original[uploaded_image["data_hash"]] = {
                        "label_row_hash": label_row_hash,
                        "data_unit_hash": data_unit_hash,
                    }

                elif label_row["data_type"] == DataType.IMG_GROUP.value:
                    image_paths = []
                    image_names = []
                    if len(label_hash_to_data_units[label_row_hash]) > 0:
                        for data_unit in label_hash_to_data_units[label_row_hash]:
                            img_path = list(label_row_structure.images_dir.glob(f"{data_unit}.*"))[0]
                            image_paths.append(img_path.as_posix())
                            image_names.append(img_path.name)

                        # Unfortunately the following function does not return metadata related to the uploaded items
                        dataset.create_image_group(file_paths=image_paths, title=label_row["data_title"])

                        # Since create_image_group does not return info related to the uploaded images, we should find its
                        # data_hash in a hacky way
                        _update_mapping(
                            self.user_client, dataset_hash, label_row_hash, data_unit_hash, new_du_to_original
                        )

                elif label_row["data_type"] == DataType.VIDEO.value:
                    video_path = list(label_row_structure.images_dir.glob(f"{data_unit_hash}.*"))[0].as_posix()

                    # Unfortunately the following function does not return metadata related to the uploaded items
                    dataset.upload_video(
                        file_path=video_path, title=label_row["data_units"][data_unit_hash]["data_title"]
                    )

                    # Since upload_video does not return info related to the uploaded video, we should find its data_hash
                    # in a hacky way
                    _update_mapping(self.user_client, dataset_hash, label_row_hash, data_unit_hash, new_du_to_original)

                else:
                    raise Exception(
                        f'Undefined data type {label_row["data_type"]} for label_row={label_row["label_hash"]}'
                    )

                uploaded_label_rows.add(label_row_hash)

            if progress_callback:
                progress_callback((counter + 1) / filtered_dataset.shape[0])

        return DatasetCreationResult(dataset_hash, new_du_to_original)

    def create_project(
        self,
        dataset_creation_result: DatasetCreationResult,
        project_title: str,
        project_description: str,
        progress_callback: Optional[Callable] = None,
    ):
        new_project_hash: str = self.user_client.create_project(
            project_title=project_title,
            dataset_hashes=[dataset_creation_result.hash],
            project_description=project_description,
            ontology_hash=self.original_project.get_project().ontology_hash,
        )

        new_project: Project = self.user_client.get_project(new_project_hash)

        # Copy labels from old project to new project
        # Three things to copy: labels, object_answers, classification_answers

        all_new_label_rows = new_project.label_rows
        for counter, new_label_row in enumerate(all_new_label_rows):
            initiated_label_row: dict = new_project.create_label_row(new_label_row["data_hash"])
            original_data = dataset_creation_result.du_original_mapping[new_label_row["data_hash"]]
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


def _update_mapping(
    user_client: EncordUserClient, new_dataset_hash: str, label_row_hash: str, data_unit_hash: str, out_mapping: dict
):
    updated_dataset = user_client.get_dataset(new_dataset_hash)
    for new_data_row in updated_dataset.data_rows:
        if new_data_row["data_hash"] not in out_mapping:
            out_mapping[new_data_row["data_hash"]] = {
                "label_row_hash": label_row_hash,
                "data_unit_hash": data_unit_hash,
            }
            return


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
