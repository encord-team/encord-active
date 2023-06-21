import json
import uuid
from typing import Optional, Set, Tuple
from sqlmodel import Session
from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.project import ProjectFileStructure
from encord_active.db.models import Project, ProjectDataMetadata, ProjectDataUnitMetadata, get_engine
from encord_active.db.update import upsert_project_and_data
from encord_active.lib.project.metadata import fetch_project_meta


def up(pfs: ProjectFileStructure) -> None:
    project_meta = fetch_project_meta(pfs.project_dir)
    project_hash: uuid.UUID = uuid.UUID(project_meta["project_hash"])
    object_hashes: Set[Tuple[uuid.UUID, int, str]] = set()
    with PrismaConnection(pfs) as conn:
        project = Project(
            project_hash=project_hash,
            project_name=project_meta["project_title"],
            project_description=project_meta["project_description"],
            project_remote_ssh_key_path=project_meta["ssh_key_path"],
            project_ontology=json.loads(
                pfs.ontology.read_text(encoding="utf-8")
            ),
        )
        label_rows = conn.labelrow.find_many(
            include={
                "data_units": True,
            }
        )
        data_metas = []
        data_units_metas = []
        for label_row in label_rows:
            data_hash = uuid.UUID(label_row.data_hash)
            label_row_json = json.loads(label_row.label_row_json)
            data_units_json: dict = label_row_json["data_units"]
            data_type: str = str(label_row_json["data_type"])
            project_data_meta = ProjectDataMetadata(
                project_hash=project_hash,
                data_hash=data_hash,
                label_hash=uuid.UUID(label_row.label_hash),
                dataset_hash=uuid.uuid4(),
                num_frames=0,
                frames_per_second=0,
                dataset_title=str(label_row_json["dataset_title"]),
                data_title=str(label_row_json["data_title"]),
                data_type=data_type,
                label_row_json=label_row_json,
            )
            fps: Optional[float] = None
            for data_unit in label_row.data_units:
                du_json = data_units_json[data_unit.data_hash]
                if data_type == "image":
                    labels_json = du_json["labels"]
                else:
                    labels_json = du_json["labels"].get(str(data_unit.frame), {})
                if data_unit.fps > 0.0:
                    fps = data_unit.fps
                objects = labels_json.get("objects", [])
                project_data_unit_meta = ProjectDataUnitMetadata(
                    project_hash=project_hash,
                    data_hash=data_hash,
                    frame=data_unit.frame,
                    data_unit_hash=uuid.UUID(data_unit.data_hash),
                    data_uri=data_unit.data_uri,
                    data_uri_is_video=data_type == "video",
                    objects=labels_json.get("objects", []),
                    classifications=labels_json.get("classifications", []),
                )
                data_units_metas.append(project_data_unit_meta)
            # Apply to parent
            project_data_meta.frames_per_second = fps
            data_metas.append(project_data_meta)

    path = pfs.project_dir.parent.expanduser().resolve() / "encord-active.sqlite"
    engine = get_engine(path)
    with Session(engine) as sess:
        upsert_project_and_data(
            sess,
            project,
            data_metas,
            data_units_metas
        )
        sess.commit()
    raise NotImplemented("Only run migration for exactly 1!!! project")