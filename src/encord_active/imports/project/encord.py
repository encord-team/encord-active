import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import encord
from tqdm import tqdm

from encord_active.db.models import (
    Project,
    ProjectDataMetadata,
    ProjectDataUnitMetadata,
)

from ..local_files import get_data_uri
from .op import ProjectImportSpec


def import_encord(
    encord_project: encord.Project,
    database_dir: Path,
    store_data_locally: bool,
) -> ProjectImportSpec:
    project_hash = uuid.UUID(encord_project.project_hash)

    label_rows = encord_project.list_label_rows_v2()
    tqdm_iter = tqdm(total=1, desc="Listing Label Rows")
    tqdm_iter.update(1)

    # Batch label row initialization.
    tqdm_iter = tqdm(total=len(label_rows), desc="Downloading Labels")
    for i in range(0, len(label_rows), 100):
        label_row_group = label_rows[i : i + 100]
        bundle = encord_project.create_bundle()
        for label_row in label_row_group:
            label_row.initialise_labels(bundle=bundle)
        bundle.execute()
        tqdm_iter.update(len(label_row_group))
    encord_project.refetch_data()

    # Iter label rows
    project_data_list: List[ProjectDataMetadata] = []
    project_du_list: List[ProjectDataUnitMetadata] = []
    du_hash_local_storage: Dict[uuid.UUID, str] = {}
    for label_row in tqdm(
        label_rows, desc="Importing & Downloading Label Rows" if store_data_locally else "Importing Label Rows"
    ):
        label_row_json = label_row.to_encord_dict()
        data_hash = uuid.UUID(label_row_json.pop("data_hash"))
        label_hash = uuid.UUID(label_row_json.pop("label_hash"))
        dataset_hash = uuid.UUID(label_row_json.pop("dataset_hash"))
        dataset_title = str(label_row_json.pop("dataset_title"))
        data_title = str(label_row_json.pop("data_title"))
        data_unit_list_json = label_row_json.pop("data_units")
        data_type = str(label_row_json.pop("data_type"))
        created_at = datetime.fromisoformat(label_row_json.pop("created_at"))
        last_edited_at = datetime.fromisoformat(label_row_json.pop("last_edited_at"))
        object_answers = label_row_json.pop("object_answers")
        classification_answers = label_row_json.pop("classification_answers")
        is_video = data_type == "video"
        project_data_list.append(
            ProjectDataMetadata(
                project_hash=project_hash,
                data_hash=data_hash,
                label_hash=label_hash,
                dataset_hash=dataset_hash,
                num_frames=len(data_unit_list_json),  # Updated later for video
                frames_per_second=None,  # Assigned later
                dataset_title=dataset_title,
                data_title=data_title,
                data_type=data_type,
                created_at=created_at,
                last_edited_at=last_edited_at,
                object_answers=object_answers,
                classification_answers=classification_answers,
            )
        )

        video_width = None
        video_height = None
        video_labels_json: Optional[Dict[str, dict]] = None
        for du_key, data_unit_json in data_unit_list_json.items():
            labels_json = data_unit_json.pop("labels")
            du_hash = uuid.UUID(data_unit_json.pop("data_hash"))
            data_uri = None
            if store_data_locally and du_hash in du_hash_local_storage:
                data_uri = du_hash_local_storage[du_hash]
            elif store_data_locally:
                video, images = encord_project.get_data(str(data_hash), get_signed_url=True)
                for image in images or []:
                    du_hash_local_storage[uuid.UUID(image["image_hash"])] = get_data_uri(
                        url_or_path=str(image["file_link"]),
                        store_data_locally=True,
                        store_symlinks=False,
                        database_dir=database_dir,
                    )
                if video and video.get("file_link"):
                    du_hash_local_storage[data_hash] = get_data_uri(
                        url_or_path=str(video["file_link"]),
                        store_data_locally=True,
                        store_symlinks=False,
                        database_dir=database_dir,
                    )
                data_uri = du_hash_local_storage[du_hash]
            if is_video:
                video_width = int(data_unit_json.pop("width"))
                video_height = int(data_unit_json.pop("height"))
                video_labels_json = labels_json
                for frame, frame_labels_json in labels_json.items():
                    project_du_list.append(
                        ProjectDataUnitMetadata(
                            project_hash=project_hash,
                            du_hash=du_hash,
                            frame=int(frame),
                            data_hash=data_hash,
                            width=video_width,
                            height=video_height,
                            data_uri=data_uri,
                            data_uri_is_video=True,
                            data_title=data_title,
                            data_type=data_type,
                            objects=frame_labels_json.get("objects", []),
                            classifications=frame_labels_json.get("classifications", []),
                        )
                    )
            else:
                project_du_list.append(
                    ProjectDataUnitMetadata(
                        project_hash=project_hash,
                        du_hash=du_hash,
                        frame=0,
                        data_hash=data_hash,
                        width=int(data_unit_json.pop("width")),
                        height=int(data_unit_json.pop("height")),
                        data_uri=data_uri,
                        data_uri_is_video=False,
                        data_title=str(data_unit_json.pop("data_title")),
                        data_type=str(data_unit_json.pop("data_type")),
                        objects=labels_json.get("objects", []),
                        classifications=labels_json.get("classifications", []),
                    )
                )

        # Video needs special case handling to populate all frames that were missed.
        if is_video:
            video_json = data_unit_list_json[str(data_hash)]
            video_fps = video_json["data_fps"]
            video_duration = video_json["data_duration"]
            video_frames = max(int(round(video_fps * video_duration)), 0)
            project_data_list[-1].frames_per_second = video_fps
            project_data_list[-1].num_frames = video_frames
            if video_width is None or video_height is None:
                raise ValueError(f"Video import failure, missing width or height: {label_row_json}")
            for i in range(0, video_frames):
                if video_labels_json is None or str(i) not in video_labels_json:
                    project_du_list.append(
                        ProjectDataUnitMetadata(
                            project_hash=project_hash,
                            du_hash=data_hash,
                            frame=i,
                            data_hash=data_hash,
                            width=video_width,
                            height=video_height,
                            data_uri=None,
                            data_uri_is_video=is_video,
                            data_title=data_title,
                            data_type=data_type,
                            objects=[],
                            classifications=[],
                        )
                    )

    # Result
    return ProjectImportSpec(
        project=Project(
            project_hash=uuid.UUID(encord_project.project_hash),
            name=encord_project.title,
            description=encord_project.description,
            remote=True,
            ontology=encord_project.ontology_structure.to_dict(),
        ),
        project_import_meta=None,
        project_data_list=project_data_list,
        project_du_list=project_du_list,
    )
