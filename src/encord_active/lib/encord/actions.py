import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Dict, NamedTuple, Optional

import pandas as pd
from encord import Dataset, Project
from encord.constants.enums import DataType
from encord.exceptions import AuthorisationError
from encord.objects.ontology_structure import OntologyStructure
from encord.orm.dataset import DataRow, ImagesDataFetchOptions, StorageLocation
from encord.orm.label_row import LabelRow
from encord.orm.project import (
    CopyDatasetAction,
    CopyDatasetOptions,
    CopyLabelsOptions,
    ReviewApprovalState,
)
from encord.utilities.label_utilities import construct_answer_dictionaries
from tqdm.auto import tqdm

from encord_active.cli.utils.server import ensure_safe_project
from encord_active.lib.common.data_utils import download_file, try_execute
from encord_active.lib.common.utils import DataHashMapping
from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.encord.project_sync import (
    LabelRowDataUnit,
    copy_filtered_data,
    copy_image_data_unit_json,
    copy_label_row_meta_json,
    copy_project_meta,
    create_filtered_db,
    create_filtered_embeddings,
    create_filtered_metrics,
    replace_uids,
)
from encord_active.lib.encord.utils import get_client
from encord_active.lib.project import Project, ProjectFileStructure
from encord_active.lib.project.metadata import fetch_project_meta, update_project_meta


class DatasetCreationResult(NamedTuple):
    hash: str
    du_original_mapping: dict[str, LabelRowDataUnit]
    lr_du_mapping: dict[LabelRowDataUnit, LabelRowDataUnit]
    new_du_hash_to_original_mapping: DataHashMapping
    new_lr_data_hash_to_original_mapping: DataHashMapping


class ProjectCreationResult(NamedTuple):
    hash: str


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
        self.ssh_key_path = ssh_key_path

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

    def _upload_label_row(
        self,
        dataset: Dataset,
        label_row_hash: str,
        data_unit_hashes: set[str],
        new_lr_data_hash_to_original_mapping: DataHashMapping,
    ) -> Optional[DataHashMapping]:
        label_row_structure = self.project_file_structure.label_row_structure(label_row_hash)
        label_row_entry = label_row_structure.get_label_row_from_db()
        label_row = label_row_structure.label_row_json

        if label_row["data_type"] == DataType.IMAGE.value:
            data_unit_hash = next(iter(label_row["data_units"]), None)  # There is only one data unit in an image (type)
            data_unit = next(label_row_structure.iter_data_unit(data_unit_hash=data_unit_hash), None)
            if data_unit_hash is not None and data_unit_hash in data_unit_hashes and data_unit is not None:
                with tempfile.TemporaryDirectory() as td:
                    tf_path = Path(td) / data_unit_hash
                    download_file(
                        data_unit.signed_url, project_dir=self.project_file_structure.project_dir, destination=tf_path
                    )
                    new_lr_data_hash = dataset.upload_image(
                        file_path=tf_path, title=label_row["data_units"][data_unit_hash]["data_title"]
                    )["data_hash"]
                new_lr_data_hash_to_original_mapping.set(
                    new_lr_data_hash, label_row.get("data_hash", label_row_entry.data_hash)
                )
                # The data unit hash and label row data hash of an image (type) are the same
                new_du_hash_to_original_mapping = DataHashMapping()
                new_du_hash_to_original_mapping.set(new_lr_data_hash, data_unit_hash)
                return new_du_hash_to_original_mapping

        elif label_row["data_type"] == DataType.IMG_GROUP.value:
            sorted_data_units: list[dict] = sorted(
                (_du for _du in label_row["data_units"].values() if _du["data_hash"] in data_unit_hashes),
                key=lambda _du: int(_du["data_sequence"]),
            )
            dus_with_signed_url = [
                (_du, _data_unit.signed_url)
                for _du in sorted_data_units
                for _data_unit in label_row_structure.iter_data_unit(data_unit_hash=_du["data_hash"])
            ]

            if len(dus_with_signed_url) > 0:
                # create_image_group() method doesn't allow to send data unit names, so their hashes are used instead
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir)
                    image_paths = [
                        download_file(
                            signed_url,
                            project_dir=self.project_file_structure.project_dir,
                            destination=tmp_path / f"{_du['data_title'].replace('/', ':')}",
                            # replace the '/' for ':' in the data unit title to avoid confusing the destination path
                        ).as_posix()
                        for _du, signed_url in dus_with_signed_url
                    ]

                    # Reverse the image paths because of the undocumented behaviour of `dataset.create_image_group()`
                    # which creates the image group with the images in the reversed order
                    image_paths = list(reversed(image_paths))
                    dataset.create_image_group(file_paths=image_paths, title=label_row["data_title"])
                # Since `create_image_group()` does not return info related to the uploaded images,
                # we need to find the data hash of the image group in a hacky way
                new_data_row: DataRow = _find_new_data_row(dataset, new_lr_data_hash_to_original_mapping)
                new_lr_data_hash_to_original_mapping.set(new_data_row.uid, label_row["data_hash"])

                # Obtain the data unit hashes of the images contained in the image group
                new_data_row.refetch_data(images_data_fetch_options=ImagesDataFetchOptions(fetch_signed_urls=False))
                new_du_hash_to_original_mapping = DataHashMapping()
                for i, du in enumerate(sorted_data_units):
                    image_data = new_data_row.images_data[i]
                    new_du_hash_to_original_mapping.set(image_data.image_hash, du["data_hash"])
                return new_du_hash_to_original_mapping

        elif label_row["data_type"] == DataType.VIDEO.value:
            data_unit_hash = next(iter(label_row["data_units"]), None)  # There is only one data unit in a video (type)
            data_unit = next(label_row_structure.iter_data_unit(data_unit_hash=data_unit_hash), None)
            if data_unit_hash is not None and data_unit_hash in data_unit_hashes and data_unit is not None:
                with tempfile.TemporaryDirectory() as td:
                    tf_path = Path(td) / data_unit_hash
                    download_file(
                        data_unit.signed_url, project_dir=self.project_file_structure.project_dir, destination=tf_path
                    )
                    dataset.upload_video(
                        file_path=str(tf_path), title=label_row["data_units"][data_unit_hash]["data_title"]
                    )
                # Since `upload_video()` does not return info related to the uploaded video,
                # we need to find the data hash of the video in a hacky way
                new_data_row = _find_new_data_row(dataset, new_lr_data_hash_to_original_mapping)
                new_lr_data_hash_to_original_mapping.set(new_data_row.uid, label_row["data_hash"])

                # The data unit hash and label row data hash of a video (type) are the same
                new_du_hash_to_original_mapping = DataHashMapping()
                new_du_hash_to_original_mapping.set(new_data_row.uid, data_unit_hash)
                return new_du_hash_to_original_mapping

        else:
            raise Exception(f'Undefined data type {label_row["data_type"]} for label_row={label_row["label_hash"]}')

        return None  # No data unit in the label row matched those in `data_unit_hashes`

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

        total_amount_of_data_units: int = sum(map(len, label_hash_to_data_units.values()))
        current_number_of_uploaded_data_units: int = 0
        new_du_hash_to_original_mapping = DataHashMapping()
        new_lr_data_hash_to_original_mapping = DataHashMapping()
        for label_row_hash, data_hashes in label_hash_to_data_units.items():
            output_new_du_hash_to_original_mapping: Optional[DataHashMapping] = try_execute(
                self._upload_label_row,
                5,
                {
                    "dataset": dataset,
                    "label_row_hash": label_row_hash,
                    "data_unit_hashes": data_hashes,
                    "new_lr_data_hash_to_original_mapping": new_lr_data_hash_to_original_mapping,
                },
            )
            if output_new_du_hash_to_original_mapping is None:
                raise Exception("Data upload failed")

            for new_data_unit_hash, data_unit_hash in output_new_du_hash_to_original_mapping.items():
                new_du_to_original[new_data_unit_hash] = LabelRowDataUnit(label_row_hash, data_unit_hash)
                # TODO: check if lrdu_mapping without label row's data hash works ok for image groups (old behaviour)
                lrdu_mapping[LabelRowDataUnit(label_row_hash, data_unit_hash)] = LabelRowDataUnit(
                    "", new_data_unit_hash
                )

            # Update global data unit mapping and advance progress bar
            new_du_hash_to_original_mapping.update(output_new_du_hash_to_original_mapping)
            if progress_callback:
                progress_callback(len(new_du_hash_to_original_mapping) / total_amount_of_data_units)

        return DatasetCreationResult(
            dataset_hash,
            new_du_to_original,
            lrdu_mapping,
            new_du_hash_to_original_mapping,
            new_lr_data_hash_to_original_mapping,
        )

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

        label_row_json_map: Dict[str, str] = {}

        for counter, new_label_row_metadata in enumerate(new_project.label_rows):
            new_lr_data_hash = new_label_row_metadata["data_hash"]
            new_label_row = new_project.create_label_row(new_lr_data_hash)
            original_lr_du = dataset_creation_result.du_original_mapping[new_lr_data_hash]

            dataset_creation_result.lr_du_mapping[original_lr_du] = LabelRowDataUnit(
                new_label_row["label_hash"], new_lr_data_hash
            )
            original_label_row = self.project_file_structure.label_row_structure(
                original_lr_du.label_row
            ).label_row_json
            label_row = self.prepare_label_row(
                original_label_row, new_label_row, new_lr_data_hash, original_lr_du.data_unit
            )
            if any(
                data_unit["labels"].get("objects", []) or data_unit["labels"].get("classifications", [])
                for data_unit in label_row["data_units"].values()
            ):
                try_execute(new_project.save_label_row, 5, {"uid": label_row["label_hash"], "label": label_row})

            # Unconditionally store locally the remapped label hash (needed for correct database migration)
            label_row_json_map[label_row["label_hash"]] = json.dumps(label_row)

            if progress_callback:
                progress_callback((counter + 1) / len(new_project.label_rows))

        old_project_hash = self.project_meta["project_hash"]
        self.project_meta["has_remote"] = True
        self.project_meta["ssh_key_path"] = self.ssh_key_path.absolute().as_posix()
        self.project_meta["project_title"] = project_title
        self.project_meta["project_hash"] = new_project.project_hash
        update_project_meta(self.project_file_structure.project_dir, self.project_meta)

        # Reverse ordering : correct data unit hash mapping data
        data_hash_mapping = DataHashMapping()
        for new_du, old_du in dataset_creation_result.new_du_hash_to_original_mapping.items():
            data_hash_mapping.set(old_du, new_du)

        replace_uids(
            self.project_file_structure,
            dataset_creation_result.lr_du_mapping,  # FIXME: needs to be updated to work correctly for non-images
            data_hash_mapping,
            old_project_hash,
            new_project.project_hash,
            dataset_creation_result.hash,
        )

        replace_db_uids(
            self.project_file_structure,
            data_hash_mapping,
            dataset_creation_result.lr_du_mapping,  # FIXME: needs to be updated to work correctly for non-images
            label_row_json_map,
        )

        return new_project

    def create_subset(
        self,
        curr_project_structure: ProjectFileStructure,
        filtered_df: pd.DataFrame,
        project_title: str,
        project_description: str,
        dataset_title: Optional[str] = None,
        dataset_description: Optional[str] = None,
        remote_copy: bool = False,
    ) -> Path:
        target_project_dir = curr_project_structure.project_dir.parent / project_title
        target_project_structure = ProjectFileStructure(target_project_dir)

        if target_project_dir.exists():
            raise Exception("Subset with the same title already exists")
        target_project_dir.mkdir()

        try:
            ids_df = filtered_df["identifier"].str.split("_", n=4, expand=True)
            filtered_lr_du = {
                LabelRowDataUnit(label_row, data_unit) for label_row, data_unit in zip(ids_df[0], ids_df[1])
            }
            filtered_label_rows = {lr_du.label_row for lr_du in filtered_lr_du}
            filtered_data_hashes = {lr_du.data_unit for lr_du in filtered_lr_du}
            filtered_labels = {
                (ids[1][0], ids[1][1], ids[1][3] if len(ids[1]) > 3 else None) for ids in ids_df.iterrows()
            }

            create_filtered_db(target_project_dir, filtered_df)

            if curr_project_structure.image_data_unit.exists():
                copy_image_data_unit_json(curr_project_structure, target_project_structure, filtered_data_hashes)

            filtered_label_row_meta = copy_label_row_meta_json(
                curr_project_structure, target_project_structure, filtered_label_rows
            )

            label_rows = {label_row for label_row in filtered_label_row_meta.keys()}

            shutil.copy2(curr_project_structure.ontology, target_project_structure.ontology)

            copy_project_meta(curr_project_structure, target_project_structure, project_title, project_description)

            create_filtered_metrics(curr_project_structure, target_project_structure, filtered_df)

            ensure_safe_project(target_project_structure.project_dir)
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
                cloned_project_hash = self._create_and_sync_subset_clone(
                    target_project_structure,
                    project_title,
                    project_description,
                    dataset_title,
                    dataset_description,
                    label_rows,
                    filtered_lr_du,
                    filtered_label_row_meta,
                )

        except Exception as e:
            shutil.rmtree(target_project_dir.as_posix())
            raise e
        return target_project_structure.project_dir

    def _create_and_sync_subset_clone(
        self,
        target_project_structure: ProjectFileStructure,
        project_title: str,
        project_description: str,
        dataset_title: Optional[str],
        dataset_description: Optional[str],
        label_rows: set[str],
        filtered_lr_du: set[LabelRowDataUnit],
        filtered_label_row_meta: dict,
    ):
        dataset_hash_map: dict[str, set[str]] = {}
        for k, v in filtered_label_row_meta.items():
            dataset_hash_map.setdefault(v["dataset_hash"], set()).add(v["data_hash"])

        cloned_project_hash = self.original_project.copy_project(
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
        cloned_project = self.user_client.get_project(cloned_project_hash)
        cloned_project_label_rows = [
            cloned_project.get_label_row(src_row.label_hash) for src_row in cloned_project.list_label_rows_v2()
        ]
        filtered_du_lr_mapping = {lrdu.data_unit: lrdu.label_row for lrdu in filtered_lr_du}

        def _get_one_data_unit(lr: dict, valid_data_units: dict) -> str:
            data_units = lr["data_units"]
            for data_unit_key in data_units.keys():
                if data_unit_key in valid_data_units:
                    return data_unit_key
            raise StopIteration(
                f"Cannot find data unit to lookup: {list(data_units.keys())}, {list(valid_data_units.keys())}"
            )

        lr_du_mapping = {
            # We only use the label hash as the key for database migration. The data hashes are preserved anyway.
            LabelRowDataUnit(
                filtered_du_lr_mapping[_get_one_data_unit(lr, filtered_du_lr_mapping)],
                lr["data_hash"],  # This value is the same
            ): LabelRowDataUnit(lr["label_hash"], lr["data_hash"])
            for lr in cloned_project_label_rows
        }

        with PrismaConnection(target_project_structure) as conn:
            original_label_rows = conn.labelrow.find_many()
        original_label_row_map = {
            original_label_row.label_hash: json.loads(original_label_row.label_row_json or "")
            for original_label_row in original_label_rows
        }

        new_label_row_map = {label_row["label_hash"]: label_row for label_row in cloned_project_label_rows}

        label_row_json_map = {}
        for (old_lr, old_du), (new_lr, new_du) in lr_du_mapping.items():
            lr = dict(original_label_row_map[old_lr])
            lr["label_hash"] = new_label_row_map[new_lr]["label_hash"]
            lr["dataset_hash"] = new_label_row_map[new_lr]["dataset_hash"]
            label_row_json_map[new_lr] = json.dumps(lr)

        project_meta = fetch_project_meta(target_project_structure.project_dir)
        project_meta["has_remote"] = True
        project_meta["project_hash"] = cloned_project_hash
        update_project_meta(target_project_structure.project_dir, project_meta)

        du_hash_map = DataHashMapping()

        replace_uids(
            target_project_structure,
            lr_du_mapping,
            du_hash_map,
            self.original_project.project_hash,
            cloned_project_hash,
            cloned_project.datasets[0]["dataset_hash"],
        )

        # Sync database identifiers
        replace_db_uids(
            target_project_structure,
            du_hash_map=DataHashMapping(),  # Preserved and used as migration key
            lr_du_mapping=lr_du_mapping,  # Update label hash and lr_dr hashes ( label hash)
            label_row_json_map=label_row_json_map,  # Update label row jsons to correct value.
        )

        return cloned_project_hash


def _find_new_data_row(dataset: Dataset, mapping: DataHashMapping) -> Optional[DataRow]:
    dataset.refetch_data()  # ensure the dataset instance include the latest changes
    for data_row in dataset.data_rows:
        if data_row["data_hash"] not in mapping:
            return data_row
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


def replace_db_uids(
    project_file_structure: ProjectFileStructure,
    du_hash_map: DataHashMapping,
    lr_du_mapping: dict[LabelRowDataUnit, LabelRowDataUnit],
    label_row_json_map: dict[str, str],
    refresh: bool = True,
):
    # Lazy import to support prisma reload
    from prisma.types import (
        DataUnitUpdateManyMutationInput,
        DataUnitWhereInput,
        LabelRowUpdateInput,
    )

    # Update the data hash changes in the DataUnit and LabelRow db tables
    with PrismaConnection(project_file_structure) as conn:
        with conn.batch_() as batcher:
            for old_du_hash, new_du_hash in du_hash_map.items():
                batcher.dataunit.update_many(
                    where=DataUnitWhereInput(data_hash=old_du_hash),
                    data=DataUnitUpdateManyMutationInput(data_hash=new_du_hash),
                )
            for (old_lr, old_du), (new_lr, new_du) in lr_du_mapping.items():
                batcher.labelrow.update(
                    where={"label_hash": old_lr},
                    data=LabelRowUpdateInput(
                        data_hash=new_du, label_hash=new_lr, label_row_json=label_row_json_map[new_lr]
                    ),
                )
    if refresh:
        Project(project_file_structure.project_dir).refresh()
