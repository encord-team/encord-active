import json
import uuid
from typing import Optional, Set, Tuple, Dict, Union
from sqlmodel import Session
from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.metrics.utils import load_metric_dataframe, load_available_metrics
from encord_active.lib.project import ProjectFileStructure
from encord_active.db.models import Project, ProjectDataMetadata, ProjectDataUnitMetadata, get_engine, \
    ProjectLabelAnalytics, ProjectDataAnalytics
from encord_active.lib.project.metadata import fetch_project_meta

WELL_KNOWN_METRICS: Dict[str, str] = {
    "Area": "metric_area",
    "Aspect Ratio": "metric_aspect_ratio",
    "Blue Values": "metric_blue",
    "Blur": "metric_blur",
    "Brightness": "metric_brightness",
    "Contrast": "metric_contrast",
    "Frame object density": "$SKIP",
    "Green Values": "metric_green",
    "Image Difficulty": "$SKIP",
    "Image Singularity": "$SKIP",
    "Object Count": "metric_object_count",
    "Random Values on Images": "metric_random",
    "Red Values": "metric_red",
    "Sharpness": "$SKIP",  # SKIPPED : Derived Metric: 1.0 - Blue (FIXME: check)
    "Annotation Duplicates": "metric_label_duplicates",
    "Annotation closeness to image borders": "metric_label_border_closeness",
    "Inconsistent Object Classification and Track IDs": "$SKIP",
    "Missing Objects and Broken Tracks": "metric_label_missing_or_broken_tracks",
    "Object Annotation Quality": "$SKIP",
    "Object Area - Absolute": "metric_area",
    "Object Area - Relative": "metric_area_relative",
    "Object Aspect Ratio": "metric_aspect_ratio",
    "Polygon Shape Similarity": "metric_label_poly_similarity",
    "Random Values on Objects": "metric_random"
}


def up(pfs: ProjectFileStructure) -> None:
    project_meta = fetch_project_meta(pfs.project_dir)
    project_hash: uuid.UUID = uuid.UUID(project_meta["project_hash"])
    object_metrics: Dict[Tuple[uuid.UUID, int, str], Dict[str, Union[int, float]]] = {}
    data_metrics: Dict[Tuple[uuid.UUID, int], Dict[str, Union[int, float]]] = {}

    # Load metadata
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
                num_frames=len(label_row.data_units),
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
                for obj in objects:
                    object_hash = str(obj["objectHash"])
                    object_metrics[(data_hash, data_unit.frame, object_hash)] = {

                    }
                data_metrics[(data_hash, data_unit.frame)] = {
                    "metric_width": data_unit.width,
                    "metric_height": data_unit.height,
                    "metric_area": data_unit.width * data_unit.height,
                    "metric_area_relative": 1.0,
                    "metric_object_count": len(objects),
                }
                project_data_unit_meta = ProjectDataUnitMetadata(
                    project_hash=project_hash,
                    data_hash=data_hash,
                    frame=data_unit.frame,
                    data_unit_hash=uuid.UUID(data_unit.data_hash),
                    data_uri=data_unit.data_uri,
                    data_uri_is_video=data_type == "video",
                    objects=objects,
                    classifications=labels_json.get("classifications", []),
                )
                data_units_metas.append(project_data_unit_meta)
            # Apply to parent
            project_data_meta.frames_per_second = fps
            data_metas.append(project_data_meta)

    # Load metrics
    metrics = load_available_metrics(pfs.metrics)
    for metric in metrics:
        metric_column_name = WELL_KNOWN_METRICS[metric.name]
        if metric_column_name == "$SKIP":
            continue
        metric_df = load_metric_dataframe(metric, normalize=False).to_dict(orient="records")
        for metric_entry in metric_df:
            label_hash, du_hash_str, frame_str, *rest = metric_entry["identifier"].split("_")
            object_hash: Optional[str] = None
            du_hash = uuid.UUID(du_hash_str)
            frame = int(frame_str)
            metrics_dict: Dict[str, Union[int, float]]
            if len(rest) > 0:
                object_hash = rest[0]
                if (du_hash, frame, object_hash) not in object_metrics:
                    raise ValueError(
                        f"Metric references invalid object!: data_hash={du_hash}, frame={frame}, object={object_hash}"
                    )
                metrics_dict = object_metrics[(du_hash, frame, object_hash)]
            else:
                metrics_dict = data_metrics[(du_hash, frame)]
            if metric_column_name not in metrics_dict:
                metrics_dict[metric_column_name] = metric_entry["score"]

    label_db_metrics = [
        ProjectLabelAnalytics(
            project_hash=project_hash,
            data_hash=data_hash,
            frame=frame,
            object_hash=object_hash,
            **metrics
        )
        for (data_hash, frame, object_hash), metrics in object_metrics.items()
    ]

    data_db_metrics = [
        ProjectDataAnalytics(
            project_hash=project_hash,
            data_hash=data_hash,
            frame=frame,
            **metrics
        )
        for (data_hash, frame), metrics in data_metrics.items()
    ]

    path = pfs.project_dir.parent.expanduser().resolve() / "encord-active.sqlite"
    engine = get_engine(path)
    with Session(engine) as sess:
        sess.add(project)
        sess.add_all(data_metas)
        sess.add_all(data_units_metas)
        sess.add_all(label_db_metrics)
        sess.add_all(data_db_metrics)
        sess.commit()
    raise ValueError("Only run migration for exactly 1!!! project")
