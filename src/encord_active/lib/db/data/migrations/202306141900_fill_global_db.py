import json
import math
import uuid
from typing import Optional, Set, Tuple, Dict, Union
from sqlmodel import Session
from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.embeddings.utils import load_label_embeddings
from encord_active.lib.metrics.types import EmbeddingType
from encord_active.lib.metrics.utils import load_metric_dataframe, load_available_metrics
from encord_active.lib.model_predictions import reader
from encord_active.lib.model_predictions.writer import MainPredictionType
from encord_active.lib.project import ProjectFileStructure
from encord_active.db.models import Project, ProjectDataMetadata, ProjectDataUnitMetadata, get_engine, \
    ProjectLabelAnalytics, ProjectDataAnalytics, ProjectTag, ProjectTaggedDataUnit, ProjectTaggedLabel, \
    ProjectClassificationAnalytics, ProjectPrediction, ProjectPredictionLabelResults
from encord_active.lib.project.metadata import fetch_project_meta

WELL_KNOWN_METRICS: Dict[str, str] = {
    "Area": "metric_area",
    "Aspect Ratio": "metric_aspect_ratio",
    "Blue Values": "metric_blue",
    "Blur": "metric_blur",
    "Brightness": "metric_brightness",
    "Contrast": "metric_contrast",
    "Frame object density": "metric_object_density",
    "Green Values": "metric_green",
    "Image Difficulty": "metric_image_difficulty",
    "Image Singularity": "metric_image_singularity",
    "Object Count": "metric_object_count",
    "Random Values on Images": "metric_random",
    "Red Values": "metric_red",
    "Sharpness": "$SKIP",  # SKIPPED : Derived Metric: 1.0 - Blue (FIXME: check)
    "Annotation Duplicates": "metric_label_duplicates",
    "Annotation closeness to image borders": "metric_label_border_closeness",
    "Inconsistent Object Classification and Track IDs": "metric_label_inconsistent_classification_and_track",
    "Missing Objects and Broken Tracks": "metric_label_missing_or_broken_tracks",
    "Object Annotation Quality": "metric_label_annotation_quality",
    "Object Area - Absolute": "metric_area",
    "Object Area - Relative": "metric_area_relative",
    "Object Aspect Ratio": "metric_aspect_ratio",
    "Polygon Shape Similarity": "metric_label_poly_similarity",
    "Random Values on Objects": "metric_random",
    'Image-level Annotation Quality': "$SKIP",  # FIXME:
    "Shape outlier detection": "$SKIP",  # FIXME:
}

DERIVED_DATA_METRICS: Set[str] = {
    "metric_width",
    "metric_height",
    "metric_area",
    "metric_area_relative",
    "metric_aspect_ratio",
    "metric_object_count",
}

DERIVED_LABEL_METRICS: Set[str] = set()


def up(pfs: ProjectFileStructure) -> None:
    project_meta = fetch_project_meta(pfs.project_dir)
    project_hash: uuid.UUID = uuid.UUID(project_meta["project_hash"])
    object_metrics: Dict[Tuple[uuid.UUID, int, str], Dict[str, Union[int, float, bytes, str]]] = {}
    classification_metrics: Dict[Tuple[uuid.UUID, int, str], Dict[str, str]] = {}
    data_metrics: Dict[Tuple[uuid.UUID, int], Dict[str, Union[int, float, bytes]]] = {}

    # Load metadata
    with PrismaConnection(pfs) as conn:
        project = Project(
            project_hash=project_hash,
            project_name=project_meta["project_title"],
            project_description=project_meta["project_description"],
            project_remote_ssh_key_path=project_meta["ssh_key_path"] if project_meta.get("has_remote", False) else None,
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
            if "label_hash" in label_row_json and label_row.label_hash != label_row_json["label_hash"]:
                raise ValueError(
                    f"Inconsistent label hash in label row json: "
                    f"{label_row.label_hash} != {label_row_json['label_hash']}"
                )
            if "data_hash" in label_row_json and label_row.data_hash != label_row_json["data_hash"]:
                raise ValueError(
                    f"Inconsistent data hash in label row json: "
                    f"{label_row.data_hash} != {label_row_json['data_hash']}"
                )
            data_units_json: dict = label_row_json["data_units"]
            data_type: str = str(label_row_json["data_type"])
            project_data_meta = ProjectDataMetadata(
                project_hash=project_hash,
                data_hash=data_hash,
                label_hash=uuid.UUID(label_row.label_hash),
                dataset_hash=uuid.UUID(label_row_json["dataset_hash"]),
                num_frames=len(label_row.data_units),
                frames_per_second=None,
                dataset_title=str(label_row_json["dataset_title"]),
                data_title=str(label_row_json["data_title"]),
                data_type=data_type,
                label_row_json=label_row_json,
            )
            fps: Optional[float] = None
            for data_unit in label_row.data_units:
                du_hash = uuid.UUID(data_unit.data_hash)
                du_json = data_units_json[data_unit.data_hash]
                if data_type == "image" or data_type == "img_group":
                    labels_json = du_json["labels"]
                elif data_type == "video":
                    labels_json = du_json["labels"].get(str(data_unit.frame), {})
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")
                if data_unit.fps > 0.0:
                    fps = data_unit.fps
                objects = labels_json.get("objects", [])
                object_hashes = set()
                for obj in objects:
                    object_hash = str(obj["objectHash"])
                    if object_hash in object_hashes:
                        raise ValueError(
                            f"Duplicate object_hash={object_hash} in du_hash={du_hash}, frame={data_unit.frame}"
                        )
                    object_hashes.add(object_hash)
                    object_metrics[(du_hash, data_unit.frame, object_hash)] = {
                        "feature_hash": str(obj["featureHash"]),
                    }
                classifications = labels_json.get("classifications", [])
                for classify in classifications:
                    classification_hash = str(classify["classificationHash"])
                    if classification_hash in object_hashes:
                        raise ValueError(
                            f"Duplicate object_hash/classification_hash={classification_hash} "
                            f"in du_hash={du_hash}, frame={data_unit.frame}"
                        )
                    object_hashes.add(classification_hash)
                    classification_metrics[(du_hash, data_unit.frame, classification_hash)] = {
                        "feature_hash": str(obj["featureHash"]),
                    }

                # FIXME: if len(classifications) == 0 && len(objects) == 0:
                #  what should the correct behaviour be?

                data_metrics[(du_hash, data_unit.frame)] = {
                    "metric_width": data_unit.width,
                    "metric_height": data_unit.height,
                    "metric_area": data_unit.width * data_unit.height,
                    "metric_area_relative": 1.0,
                    "metric_aspect_ratio": float(data_unit.width) / float(data_unit.height),
                    "metric_object_count": len(objects),
                }
                project_data_unit_meta = ProjectDataUnitMetadata(
                    project_hash=project_hash,
                    du_hash=du_hash,
                    # FIXME: check calculation of this value is consistent!
                    frame=data_unit.frame if data_type == "video" else 0,
                    data_hash=data_hash,
                    data_uri=data_unit.data_uri,
                    data_uri_is_video=data_type == "video",
                    objects=objects,
                    classifications=classifications,
                )
                data_units_metas.append(project_data_unit_meta)
            # Apply to parent
            project_data_meta.frames_per_second = fps
            data_metas.append(project_data_meta)

        tag_id_map: Dict[int, uuid.UUID] = {}
        project_tag_definitions = []
        for tag in conn.tag.find_many():
            tag_uuid = uuid.uuid4()
            tag_id_map[tag.id] = tag_uuid
            project_tag_definitions.append(ProjectTag(
                tag_uuid=tag_uuid,
                project_hash=project_hash,
                name=tag.name,
            ))

        project_data_tags = []
        project_label_tags = []
        for item_tag in conn.itemtag.find_many():
            du_hash = uuid.UUID(item_tag.data_hash)
            frame = item_tag.frame
            object_hash = item_tag.object_hash if len(item_tag.object_hash) > 0 else None
            tag_uuid = tag_id_map[item_tag.tag_id]
            if object_hash is None:
                _exists = data_metrics[(du_hash, frame)]
                project_data_tags.append(ProjectTaggedDataUnit(
                    project_hash=project_hash,
                    du_hash=du_hash,
                    frame=frame,
                    tag_hash=tag_uuid,
                ))
            else:
                _exists = object_metrics[(du_hash, frame, object_hash)]
                project_label_tags.append(ProjectTaggedLabel(
                    project_hash=project_hash,
                    du_hash=du_hash,
                    frame=frame,
                    object_hash=object_hash,
                    tag_hash=tag_uuid,
                ))

    # Load metrics
    metrics = load_available_metrics(pfs.metrics)
    for metric in metrics:
        metric_column_name = WELL_KNOWN_METRICS[metric.name]
        if metric_column_name == "$SKIP":
            continue
        metric_df = load_metric_dataframe(metric, normalize=False).to_dict(orient="records")
        for metric_entry in metric_df:
            label_hash, du_hash_str, frame_str, *rest = metric_entry["identifier"].split("_")
            du_hash = uuid.UUID(du_hash_str)
            frame = int(frame_str)
            metrics_dict: Dict[str, Union[int, float]]
            if len(rest) == 1 or len(rest) == 2:
                # FIXME: len(rest) == 2 may need special case handling, not sure why it is generated!
                if len(rest) == 2:
                    print(f"WARNING: dropping second object hash in extended identifier: {metric_entry['identifier']}")
                object_hash = rest[0]
                if (du_hash, frame, object_hash) not in object_metrics:
                    raise ValueError(
                        f"Metric references invalid object!: du_hash={du_hash}, frame={frame}, object={object_hash}"
                    )
                metrics_dict = object_metrics[(du_hash, frame, object_hash)]
                metrics_derived = DERIVED_LABEL_METRICS
            elif len(rest) == 0:
                metrics_dict = data_metrics[(du_hash, frame)]
                metrics_derived = DERIVED_DATA_METRICS
            else:
                raise ValueError(f"Unknown packing of identifier: {metric_entry['identifier']}")
            if metric_column_name not in metrics_dict:
                metrics_dict[metric_column_name] = metric_entry["score"]
            elif metric_column_name not in metrics_derived:
                raise ValueError(
                    f"Duplicate metric assignment for, column={metric_column_name},"
                    f"identifier{metric_entry['identifier']}"
                )

    # Load embeddings
    for embedding_type, embedding_name in [
        (EmbeddingType.IMAGE, "embedding_clip"), (EmbeddingType.OBJECT, "embedding_clip"),
        # FIXME: EmbeddingType.CLASSIFICATION??
        # FIXME: (EmbeddingType.HU_MOMENTS, "embedding_hu")
    ]:
        label_embeddings = load_label_embeddings(embedding_type, pfs)
        for embedding in label_embeddings:
            du_hash = uuid.UUID(embedding["data_unit"])
            frame = int(embedding["frame"])
            object_hash: Optional[str] = str(embedding["labelHash"]) \
                if embedding.get("labelHash", None) is not None \
                else None
            embedding_bytes: bytes = embedding["embedding"].tobytes()
            if object_hash is not None:
                metrics_dict = object_metrics[(du_hash, frame, object_hash)]
            else:
                metrics_dict = data_metrics[(du_hash, frame)]
            metrics_dict[embedding_name] = embedding_bytes

    # Load 2d embeddings
    # FIXME: implement (reduction should be marked as separate)

    # Load predictions
    prediction_hash = uuid.uuid4()
    predictions_run_db = []
    predictions_objects_db = []
    for prediction_type in [
        MainPredictionType.OBJECT,
        # FIXME: MainPredictionType.CLASSIFICATION
    ]:
        predictions_dir = pfs.predictions
        metrics_dir = pfs.metrics
        predictions_metric_datas = reader.get_prediction_metric_data(predictions_dir, metrics_dir)
        model_predictions = reader.get_model_predictions(predictions_dir, predictions_metric_datas, prediction_type)
        if model_predictions is None:
            # No predictions to migrate, hence can be safely ignored
            continue
        for model_prediction in model_predictions.to_dict(orient="records"):
            identifier: str = model_prediction.pop('identifier')
            model_prediction.pop('url')
            model_prediction.pop('Unnamed: 0')
            img_id = int(model_prediction.pop('img_id'))
            class_id = int(model_prediction.pop('class_id'))
            confidence = float(model_prediction.pop('confidence'))
            rle = float(model_prediction.pop('rle'))
            iou = float(model_prediction.pop('iou'))
            pbb = [
                float(model_prediction.pop('x1')),
                float(model_prediction.pop('y1')),
                float(model_prediction.pop('x2')),
                float(model_prediction.pop('y2'))
            ]
            p_metrics = {}
            f_metrics = {}
            for metric_name, metric_value in model_prediction.items():
                metric_target = metric_name[-4:]
                metric_key = WELL_KNOWN_METRICS[metric_name[:-4]]
                if metric_key == "$SKIP":
                    continue
                if metric_target == ' (P)':
                    p_metrics[metric_key] = metric_value
                elif metric_target == ' (F)':
                    f_metrics[metric_key] = metric_value
                else:
                    raise ValueError(f"Unknown metric target: '{metric_target}'")

            # Add the new prediction to the database!!!
            label_hash, du_hash_str, frame_str, obj_or_class_hash, *rest = identifier.split("_")
            du_hash = uuid.UUID(du_hash_str)
            frame = int(frame_str)
            if len(rest) != 0:
                print(f"WARNING: throwing away rest of identifier in prediction: {rest} (ident={identifier})")

            predictions_objects_db.append(
                ProjectPredictionLabelResults(
                    prediction_hash=prediction_hash,
                    du_hash=du_hash,
                    frame=frame,
                    object_hash=obj_or_class_hash, # FIXME: need to condition on if this is classification or not!
                    confidence=confidence,
                    # FIXME: nan -> null, is this correct behaviour?
                    rle=rle,
                    iou=iou,
                    # FIXME: pbb (aim to support all prediction descriptors (bytes??)
                )
            )

        # label_metric_datas = reader.get_label_metric_data(metrics_dir)
        # labels = reader.get_labels(predictions_dir, label_metric_datas, prediction_type)
        # FIXME: store prediction metadata for the prediction run.

        # Ensure prediction currently stored actually exists.
        if len(predictions_run_db) == 0:
            predictions_run_db.append(
                ProjectPrediction(
                    project_hash=project_hash,
                    prediction_hash=prediction_hash,
                    name="Migrated Prediction",
                )
            )

    label_db_metrics = [
        ProjectLabelAnalytics(
            project_hash=project_hash,
            du_hash=du_hash,
            frame=frame,
            object_hash=object_hash,
            **metrics
        )
        for (du_hash, frame, object_hash), metrics in object_metrics.items()
    ]

    classification_db_metrics = [
        ProjectClassificationAnalytics(
            project_hash=project_hash,
            du_hash=du_hash,
            frame=frame,
            classification_hash=classification_hash,
            **metrics,
        )
        for (du_hash, frame, classification_hash), metrics in classification_metrics.items()
    ]

    data_db_metrics = [
        ProjectDataAnalytics(
            project_hash=project_hash,
            du_hash=du_hash,
            frame=frame,
            **metrics
        )
        for (du_hash, frame), metrics in data_metrics.items()
    ]

    path = pfs.project_dir.parent.expanduser().resolve() / "encord-active.sqlite"
    engine = get_engine(path)
    with Session(engine) as sess:
        sess.add(project)
        sess.add_all(data_metas)
        sess.add_all(data_units_metas)
        sess.add_all(label_db_metrics)
        sess.add_all(classification_db_metrics)
        sess.add_all(data_db_metrics)
        sess.add_all(project_tag_definitions)
        sess.add_all(project_data_tags)
        sess.add_all(project_label_tags)
        sess.add_all(predictions_run_db)
        sess.add_all(predictions_objects_db)
        sess.commit()
