import uuid
from pathlib import Path
from typing import List

import encord
from tqdm import tqdm

from encord_active.db.models import Project, ProjectDataUnitMetadata, ProjectDataMetadata
from .op import ProjectImportSpec


def import_encord(
    encord_project: encord.Project,
    ssh_key_path: Path,
    database_dir: Path,
    store_data_locally: bool
) -> ProjectImportSpec:
    project_hash = uuid.UUID(encord_project.project_hash)

    # Iter label rows
    label_rows = encord_project.list_label_rows_v2()
    project_data_list: List[ProjectDataMetadata] = []
    project_du_list: List[ProjectDataUnitMetadata] = []
    for label_row in tqdm(label_rows, desc="Importing Label Rows"):
        label_row.initialise_labels()
        label_row_json = label_row.to_encord_dict()
        data_hash = uuid.UUID(label_row_json["data_hash"])
        data_type = str(label_row_json["data_type"])
        is_video = data_type == "video"
        width = label_row_json.get("width", 0)
        height = label_row_json.get("height", 0)
        project_data_list.append(ProjectDataMetadata(
            project_hash=project_hash,
            data_hash=data_hash,
            label_hash=uuid.UUID(label_row_json["label_hash"]),
            dataset_hash=uuid.UUID(label_row_json["dataset_hash"]),
            num_frames=len(label_row_json["data_units"]),  # Updated later for video
            frames_per_second=None,  # Assigned later
            dataset_title=str(label_row_json["dataset_title"]),
            data_title=str(label_row_json["data_title"]),
            data_type=str(label_row_json["data_type"]),
            label_row_json=label_row_json,
        ))
        data_unit_list_json = label_row_json["data_units"]
        for du_key, data_unit_json in data_unit_list_json.items():
            labels_json = data_unit_json["labels"]
            project_du_list.append(ProjectDataUnitMetadata(
                project_hash=project_hash,
                du_hash=uuid.UUID(data_unit_json["data_hash"]),
                frame=int(du_key) if is_video else 0,
                data_hash=data_hash,
                width=int(data_unit_json.get("width", 0) or width),
                height=int(data_unit_json.get("height", 0) or height),
                data_uri=None,
                data_uri_is_video=is_video,
                objects=labels_json.get("objects", []),
                classifications=labels_json.get("classifications", []),
            ))
        # Video needs special case handling to populate all frames that were missed.
        if data_type == "video":
            video_json = data_unit_list_json[str(data_hash)]
            video_fps = video_json["video_fps"]
            video_duration = video_json["data_duration"]
            video_frames = max(int(round(video_fps * video_duration)), 0)
            project_data_list[-1].frames_per_second = video_fps
            project_data_list[-1].num_frames = video_frames
            if width is None or height is None:
                raise ValueError(f"Video import failure, missing width & height")
            for i in range(0, video_frames):
                if str(i) not in data_unit_list_json:
                    project_du_list.append(ProjectDataUnitMetadata(
                        project_hash=project_hash,
                        du_hash=data_hash,
                        frame=i,
                        data_hash=data_hash,
                        width=int(width),
                        height=int(height),
                        data_uri=None,
                        data_uri_is_video=is_video,
                        objects=[],
                        classifications=[],
                    ))

    # Result
    return ProjectImportSpec(
        project=Project(
            project_hash=uuid.UUID(encord_project.project_hash),
            project_name=encord_project.title,
            project_description=encord_project.description,
            project_remote_ssh_key_path=str(ssh_key_path),
            project_ontology=encord_project.get_project_ontology().to_dict(),
        ),
        project_data_list=project_data_list,
        project_du_list=project_du_list,
    )
