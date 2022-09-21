import json
from pathlib import Path

import pandas as pd
import streamlit as st
from encord import Dataset, EncordUserClient, Project
from encord.constants.enums import DataType
from encord.exceptions import AuthorisationError
from encord.orm.dataset import Image, StorageLocation
from encord.utilities.label_utilities import construct_answer_dictionaries
from tqdm import tqdm

from encord_active.lib.common.utils import fetch_project_meta


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


def create_a_new_dataset(
    user_client: EncordUserClient, dataset_title: str, dataset_description: str, filtered_dataset: pd.DataFrame
) -> tuple[str, dict[str, dict[str, str]]]:
    new_du_to_original: dict[str, dict] = {}
    user_client.create_dataset(
        dataset_title=dataset_title, dataset_type=StorageLocation.CORD_STORAGE, dataset_description=dataset_description
    )
    dataset_hash: str = user_client.get_datasets(title_eq=dataset_title)[0]["dataset"].dataset_hash
    dataset: Dataset = user_client.get_dataset(dataset_hash)

    # The following operation is for image groups (to upload them efficiently)
    label_hash_to_data_units: dict[str, list] = {}
    for index, item in tqdm(filtered_dataset.iterrows(), total=filtered_dataset.shape[0]):
        label_row_hash, data_unit_hash, *rest = item["identifier"].split("_")
        label_hash_to_data_units.setdefault(label_row_hash, []).append(data_unit_hash)

    temp_progress_bar = st.empty()
    temp_progress_bar.progress(0.0)
    uploaded_label_rows: set = set()
    for counter, (index, item) in enumerate(filtered_dataset.iterrows()):
        label_row_hash, data_unit_hash, frame_id, *rest = item["identifier"].split("_")
        json_txt = (st.session_state.project_dir / "data" / label_row_hash / "label_row.json").expanduser().read_text()
        label_row = json.loads(json_txt)

        if label_row_hash not in uploaded_label_rows:
            if label_row["data_type"] == DataType.IMAGE.value:
                image_path = list(
                    Path(st.session_state.project_dir / "data" / label_row_hash / "images").glob(f"{data_unit_hash}.*")
                )[0]
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
                        img_path: Path = list(
                            Path(st.session_state.project_dir / "data" / label_row_hash / "images").glob(
                                f"{data_unit}.*"
                            )
                        )[0]
                        image_paths.append(img_path.as_posix())
                        image_names.append(img_path.name)

                    # Unfortunately the following function does not return metadata related to the uploaded items
                    dataset.create_image_group(file_paths=image_paths, title=label_row["data_title"])

                    # Since create_image_group does not return info related to the uploaded images, we should find its
                    # data_hash in a hacky way
                    _update_mapping(user_client, dataset_hash, label_row_hash, data_unit_hash, new_du_to_original)

            elif label_row["data_type"] == DataType.VIDEO.value:
                video_path = list(
                    Path(st.session_state.project_dir / "data" / label_row_hash / "images").glob(f"{data_unit_hash}.*")
                )[0].as_posix()

                # Unfortunately the following function does not return metadata related to the uploaded items
                dataset.upload_video(file_path=video_path, title=label_row["data_units"][data_unit_hash]["data_title"])

                # Since upload_video does not return info related to the uploaded video, we should find its data_hash
                # in a hacky way
                _update_mapping(user_client, dataset_hash, label_row_hash, data_unit_hash, new_du_to_original)

            else:
                st.error(f'Undefined data type {label_row["data_type"]} for label_row={label_row["label_hash"]}')

            uploaded_label_rows.add(label_row_hash)

        temp_progress_bar.progress(counter / filtered_dataset.shape[0])

    temp_progress_bar.empty()
    return dataset_hash, new_du_to_original


def create_new_project_on_encord_platform(
    dataset_title: str,
    dataset_description: str,
    project_title: str,
    project_description: str,
    filtered_dataset: pd.DataFrame,
):
    try:
        project_meta = fetch_project_meta(st.session_state.project_dir)
    except (KeyError, FileNotFoundError) as _:
        st.markdown(
            f"""
        ❌ No `project_meta.yaml` file in the project folder.
        Please create `project_meta.yaml` file in **{st.session_state.project_dir}** folder with the following content
        and try again:
        ``` yaml
        project_hash: <project_hash>
        ssh_key_path: /path/to/your/encord/ssh_key
        ```
        """
        )
        return

    meta_file = st.session_state.project_dir / "project_meta.yaml"
    ssh_key_path = project_meta.get("ssh_key_path")
    if not ssh_key_path:
        st.error(f"`ssh_key_path` not specified in the project meta data file `{meta_file}`.")
        return

    ssh_key_path = Path(ssh_key_path).expanduser()
    if not ssh_key_path.is_file():
        st.error(f"No SSH file in location:{ssh_key_path}")
        return

    user_client = EncordUserClient.create_with_ssh_private_key(
        Path(ssh_key_path).expanduser().read_text(encoding="utf-8"),
    )

    original_project_hash = project_meta.get("project_hash")
    if not original_project_hash:
        st.error(f"`project_hash` not specified in the project meta data file `{meta_file}`.")
        return

    original_project: Project = user_client.get_project(original_project_hash)
    try:
        if original_project.project_hash == original_project_hash:
            pass
    except AuthorisationError:
        st.error(
            f'The user associated to the ssh key `{ssh_key_path}` does not have access to the project with project hash `{original_project_hash}`. Run "encord-active config set ssh_key_path /path/to/your/key_file" to set it.'
        )
        return

    datasets_with_same_title = user_client.get_datasets(title_eq=dataset_title)
    if len(datasets_with_same_title) > 0:
        st.error(
            f"Dataset title '{dataset_title}' already exists in your list of datasets at Encord. Please use a different title."
        )
        return None

    label = st.empty()
    label.text("Step 1/2: Uploading data...")
    new_dataset_hash, new_du_to_original = create_a_new_dataset(
        user_client, dataset_title, dataset_description, filtered_dataset
    )

    new_project_hash: str = user_client.create_project(
        project_title=project_title,
        dataset_hashes=[new_dataset_hash],
        project_description=project_description,
        ontology_hash=original_project.get_project().ontology_hash,
    )

    new_project: Project = user_client.get_project(new_project_hash)

    # Copy labels from old project to new project
    # Three things to copy: labels, object_answers, classification_answers

    all_new_label_rows = new_project.label_rows
    label.text("Step 2/2: Uploading labels...")
    temp_progress_bar = st.empty()
    temp_progress_bar.progress(0.0)
    for counter, new_label_row in enumerate(all_new_label_rows):
        initiated_label_row: dict = new_project.create_label_row(new_label_row["data_hash"])

        with open(
            (
                st.session_state.project_dir
                / "data"
                / new_du_to_original[new_label_row["data_hash"]]["label_row_hash"]
                / "label_row.json"
            ).expanduser(),
            "r",
            encoding="utf-8",
        ) as file:
            original_label_row = json.load(file)

        if initiated_label_row["data_type"] in [DataType.IMAGE.value, DataType.VIDEO.value]:

            original_labels = original_label_row["data_units"][
                new_du_to_original[new_label_row["data_hash"]]["data_unit_hash"]
            ]["labels"]
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
                for original_data_unit in original_label_row["data_units"].values():
                    if original_data_unit["data_hash"] == data_unit["data_title"].split(".")[0]:
                        data_unit["labels"] = original_data_unit["labels"]
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

        temp_progress_bar.progress(counter / len(all_new_label_rows))

    temp_progress_bar.empty()
    label.info("🎉 New project is created!")
    new_project_link = f"https://app.encord.com/projects/view/{new_project_hash}/summary"
    new_dataset_link = f"https://app.encord.com/datasets/view/{new_dataset_hash}"
    st.markdown(f"[Go to new project]({new_project_link})")
    st.markdown(f"[Go to new dataset]({new_dataset_link})")
