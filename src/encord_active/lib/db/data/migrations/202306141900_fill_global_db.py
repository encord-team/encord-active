import json
import math
import uuid
from typing import Dict, List, Optional, Set, Tuple, Union

from sqlmodel import Session

from encord_active.db.metrics import (
    AnnotationMetrics,
    DataAnnotationSharedMetrics,
    DataMetrics,
    MetricDefinition,
    MetricType,
)
from encord_active.db.models import (
    AnnotationType,
    Project,
    ProjectAnnotationAnalytics,
    ProjectDataAnalytics,
    ProjectDataMetadata,
    ProjectDataUnitMetadata,
    ProjectPrediction,
    ProjectPredictionObjectResults,
    ProjectPredictionUnmatchedResults,
    ProjectTag,
    ProjectTaggedAnnotation,
    ProjectTaggedDataUnit,
    get_engine,
)
from encord_active.lib.common.data_utils import file_path_to_url, url_to_file_path
from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.embeddings.utils import load_label_embeddings
from encord_active.lib.metrics.types import EmbeddingType
from encord_active.lib.metrics.utils import (
    load_available_metrics,
    load_metric_dataframe,
)
from encord_active.lib.model_predictions import reader
from encord_active.lib.model_predictions.writer import MainPredictionType
from encord_active.lib.project import ProjectFileStructure
from encord_active.lib.project.metadata import fetch_project_meta

WELL_KNOWN_METRICS: Dict[str, str] = {
    "Area": "metric_area",
    "Aspect Ratio": "metric_aspect_ratio",
    "Blue Values": "metric_blue",
    "Blur": "$SKIP",  # Virtual attribute = 1.0 - metric_sharpness
    "Brightness": "metric_brightness",
    "Contrast": "metric_contrast",
    "Frame object density": "metric_object_density",
    "Green Values": "metric_green",
    "Image Difficulty": "metric_image_difficulty",
    "Image Singularity": "metric_image_singularity",
    "Object Count": "metric_object_count",
    "Random Values on Images": "metric_random",
    "Red Values": "metric_red",
    "Sharpness": "metric_sharpness",
    "Annotation Duplicates": "metric_label_duplicates",
    "Annotation closeness to image borders": "metric_label_border_closeness",
    "Inconsistent Object Classification and Track IDs": "metric_label_inconsistent_classification_and_track",
    "Missing Objects and Broken Tracks": "metric_label_missing_or_broken_tracks",
    "Object Annotation Quality": "metric_annotation_quality",
    "Object Area - Absolute": "metric_area",
    "Object Area - Relative": "metric_area_relative",
    "Object Aspect Ratio": "metric_aspect_ratio",
    "Polygon Shape Similarity": "metric_label_poly_similarity",
    "Random Values on Objects": "metric_random",
    "Image-level Annotation Quality": "metric_annotation_quality",
    "Shape outlier detection": "metric_label_shape_outlier",
}

# Metrics that need to be migrated to the normalised format from percentage for consistency with other metrics.
WELL_KNOWN_PERCENTAGE_METRICS: Set[str] = {
    "metric_object_density",
    "metric_area_relative",
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


def assert_valid_args(cls, dct: dict) -> dict:
    for k in dct.keys():
        if k not in cls.__fields__:
            raise ValueError(f"Invalid field type: {k} for class: {cls.__name__}")
    return dct


def _assign_metrics(
    metric_column_name: str,
    metrics_dict: dict,
    score: float,
    metric_types: Dict[str, MetricDefinition],
    metrics_derived: Set[str],
    error_identifier: str,
):
    if metric_column_name not in metrics_dict:
        if metric_column_name in WELL_KNOWN_PERCENTAGE_METRICS:
            score = score / 100.0
        metric_def = metric_types[metric_column_name]
        if metric_column_name == "metric_sharpness" or metric_column_name == "metric_label_shape_outlier":
            score = min(max(score, 0.0), 1.0)
            # FIXME: properly scale sharpness to a normalised result!!!
        elif metric_def.type == MetricType.NORMAL:
            # Clamp 0->1 (some metrics do not generate correctly clamped scores.
            original_score = score
            score = min(max(score, 0.0), 1.0)
            if original_score != score and not (math.isnan(score) and math.isnan(original_score)):
                score_delta = float(abs(original_score - score))
                score_p1 = abs((score_delta / float(score)) * 100.0)
                score_p2 = abs((score_delta / float(original_score)) * 100.0)
                if score_p1 >= 0.5 or score_p2 >= 0.5:
                    print(
                        f"WARNING: clamped normal metric score - {metric_column_name}: {original_score} => {score}, "
                        f"Percentage = {score_p1}, {score_p2}"
                        f"identifier={error_identifier}"
                    )
        if metric_def.virtual is None:
            # Virtual attributes should not be stored!!
            metrics_dict[metric_column_name] = score
    elif metric_column_name not in metrics_derived:
        raise ValueError(
            f"Duplicate metric assignment for, column={metric_column_name}," f"identifier={error_identifier}"
        )
    else:
        existing_score = metrics_dict[metric_column_name]
        if existing_score != score:
            score_delta = float(abs(existing_score - score))
            score_p1 = abs((score_delta / float(score)) * 100.0)
            score_p2 = abs((score_delta / float(existing_score)) * 100.0)
            if score_p1 >= 0.5 or score_p2 >= 0.5:
                print(
                    f"WARNING: different derived and calculated scores: "
                    f"Percentage = {score_p1}, {score_p2}"
                    f"{metric_column_name}, {existing_score}, {score}"
                )


def up(pfs: ProjectFileStructure) -> None:
    project_meta = fetch_project_meta(pfs.project_dir)
    project_hash: uuid.UUID = uuid.UUID(project_meta["project_hash"])
    annotation_metrics: Dict[
        Tuple[uuid.UUID, int, str], Dict[str, Union[int, float, bytes, str, AnnotationType, None]]
    ] = {}
    data_metrics: Dict[Tuple[uuid.UUID, int], Dict[str, Union[int, float, bytes]]] = {}
    data_classification_metrics: Dict[
        Tuple[uuid.UUID, int], List[Dict[str, Union[int, float, bytes, str, AnnotationType, None]]]
    ] = {}

    database_dir = pfs.project_dir.parent.expanduser().resolve()

    # Load metadata
    with PrismaConnection(pfs) as conn:
        project = Project(
            project_hash=project_hash,
            project_name=project_meta["project_title"],
            project_description=project_meta["project_description"],
            project_remote_ssh_key_path=project_meta["ssh_key_path"] if project_meta.get("has_remote", False) else None,
            project_ontology=json.loads(pfs.ontology.read_text(encoding="utf-8")),
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
            label_row_json = json.loads(label_row.label_row_json or "")
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
                num_frames=len(label_row.data_units or []),
                frames_per_second=None,
                dataset_title=str(label_row_json["dataset_title"]),
                data_title=str(label_row_json["data_title"]),
                data_type=data_type,
                label_row_json=label_row_json,
            )
            fps: Optional[float] = None
            for data_unit in label_row.data_units or []:
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
                object_hashes_seen = set()
                for obj in objects:
                    object_hash = str(obj["objectHash"])
                    if object_hash in object_hashes_seen:
                        raise ValueError(
                            f"Duplicate object_hash={object_hash} in du_hash={du_hash}, frame={data_unit.frame}"
                        )
                    object_hashes_seen.add(object_hash)
                    annotation_type = AnnotationType(str(obj["shape"]))
                    annotation_metrics[(du_hash, data_unit.frame, object_hash)] = {
                        "feature_hash": str(obj["featureHash"]),
                        "annotation_type": annotation_type,
                        "annotation_value": obj.get("value", None),
                        "annotation_creator": str(obj["createdBy"]) if obj["manualAnnotation"] else None,
                        "annotation_confidence": float(obj["confidence"]),
                    }
                classifications = labels_json.get("classifications", [])
                for classify in classifications:
                    classification_hash = str(classify["classificationHash"])
                    if classification_hash in object_hashes_seen:
                        raise ValueError(
                            f"Duplicate object_hash/classification_hash={classification_hash} "
                            f"in du_hash={du_hash}, frame={data_unit.frame}"
                        )
                    object_hashes_seen.add(classification_hash)
                    annotation_metrics[(du_hash, data_unit.frame, classification_hash)] = {
                        "feature_hash": str(classify["featureHash"]),
                        "annotation_type": AnnotationType.CLASSIFICATION,
                        "annotation_value": classify.get("value", None),
                        "annotation_creator": str(classify["createdBy"]) if classify["manualAnnotation"] else None,
                        "annotation_confidence": float(classify["confidence"]),
                    }
                    data_classification_metrics.setdefault((du_hash, data_unit.frame), []).append(
                        annotation_metrics[(du_hash, data_unit.frame, classification_hash)]
                    )

                data_metrics[(du_hash, data_unit.frame)] = {
                    "metric_width": data_unit.width,
                    "metric_height": data_unit.height,
                    "metric_area": data_unit.width * data_unit.height,
                    "metric_aspect_ratio": float(data_unit.width) / float(data_unit.height),
                    "metric_object_count": len(objects),
                }
                # Ensure relative uri paths remain correct!!
                data_uri = data_unit.data_uri
                if data_uri is not None and data_uri.startswith("relative://"):
                    data_uri_path = url_to_file_path(data_uri, pfs.project_dir)
                    if data_uri_path is not None:
                        data_uri = file_path_to_url(data_uri_path.expanduser().resolve(), database_dir)
                # Add data unit to the database
                project_data_unit_meta = ProjectDataUnitMetadata(
                    project_hash=project_hash,
                    du_hash=du_hash,
                    # FIXME: check calculation of this value is consistent!
                    frame=data_unit.frame if data_type == "video" else 0,
                    data_hash=data_hash,
                    data_uri=data_uri,
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
            project_tag_definitions.append(
                ProjectTag(
                    tag_uuid=tag_uuid,
                    project_hash=project_hash,
                    name=tag.name,
                )
            )

        project_data_tags: List[ProjectTaggedDataUnit] = []
        project_object_tags: List[ProjectTaggedAnnotation] = []
        for item_tag in conn.itemtag.find_many():
            du_hash = uuid.UUID(item_tag.data_hash)
            frame = item_tag.frame
            object_hash_opt = item_tag.object_hash if len(item_tag.object_hash) > 0 else None
            tag_uuid = tag_id_map[item_tag.tag_id]
            if object_hash_opt is None:
                _exists = data_metrics[(du_hash, frame)]  # type: ignore
                project_data_tags.append(
                    ProjectTaggedDataUnit(
                        project_hash=project_hash,
                        du_hash=du_hash,
                        frame=frame,
                        tag_hash=tag_uuid,
                    )
                )
            else:
                _exists = annotation_metrics[(du_hash, frame, object_hash)]  # type: ignore
                project_object_tags.append(
                    ProjectTaggedAnnotation(
                        project_hash=project_hash,
                        du_hash=du_hash,
                        frame=frame,
                        object_hash=object_hash_opt,
                        tag_hash=tag_uuid,
                    )
                )

    # Load metrics
    metrics = load_available_metrics(pfs.metrics)
    for metric in metrics:
        metric_column_name = WELL_KNOWN_METRICS[metric.name]
        if metric_column_name == "$SKIP":
            continue
        metric_df = load_metric_dataframe(metric, normalize=False).to_dict(orient="records")
        for metric_entry in metric_df:
            label_hash, du_hash_str, frame_str, *object_hash_list = metric_entry["identifier"].split("_")
            du_hash = uuid.UUID(du_hash_str)
            frame = int(frame_str)
            metrics_dict: Dict[str, Union[int, float]]
            if len(object_hash_list) >= 1:
                for object_hash in object_hash_list:
                    if (du_hash, frame, object_hash) not in annotation_metrics:
                        print(
                            f"WARNING: Metric references invalid object!: "
                            f"du_hash={du_hash}, frame={frame}, object={object_hash}"
                        )
                        continue
                    _assign_metrics(
                        metric_column_name=metric_column_name,
                        metrics_dict=annotation_metrics[(du_hash, frame, object_hash)],
                        score=metric_entry["score"],
                        metric_types=AnnotationMetrics,
                        metrics_derived=DERIVED_LABEL_METRICS,
                        error_identifier=metric_entry["identifier"],
                    )
            else:
                _assign_metrics(
                    metric_column_name=metric_column_name,
                    metrics_dict=data_metrics[(du_hash, frame)],
                    score=metric_entry["score"],
                    metric_types=DataMetrics,
                    metrics_derived=DERIVED_DATA_METRICS,
                    error_identifier=metric_entry["identifier"],
                )
                if metric_column_name in DataAnnotationSharedMetrics:
                    for classification_dict in data_classification_metrics.get((du_hash, frame), []):
                        _assign_metrics(
                            metric_column_name=metric_column_name,
                            metrics_dict=classification_dict,
                            score=metric_entry["score"],
                            metric_types=AnnotationMetrics,
                            metrics_derived=DERIVED_LABEL_METRICS,
                            error_identifier=f"classify for: {metric_entry['identifier']}",
                        )

    # Load embeddings
    for embedding_type, embedding_name in [
        (EmbeddingType.IMAGE, "embedding_clip"),
        (EmbeddingType.OBJECT, "embedding_clip"),
        (EmbeddingType.CLASSIFICATION, "embedding_clip"),
        # FIXME: (EmbeddingType.HU_MOMENTS, "embedding_hu")
    ]:
        label_embeddings = load_label_embeddings(embedding_type, pfs)
        for embedding in label_embeddings:
            du_hash = uuid.UUID(embedding["data_unit"])
            frame = int(embedding["frame"])
            object_hash_emb_opt: Optional[str] = (
                str(embedding["labelHash"]) if embedding.get("labelHash", None) is not None else None
            )
            embedding_bytes: bytes = embedding["embedding"].tobytes()
            if object_hash_emb_opt is not None:
                metrics_dict_emb: Dict[str, Union[int, float, bytes, str, AnnotationType, None]] = annotation_metrics[
                    (du_hash, frame, object_hash_emb_opt)
                ]
            else:
                metrics_dict_emb = data_metrics[(du_hash, frame)]  # type: ignore
            metrics_dict_emb[embedding_name] = embedding_bytes

    # Load 2d embeddings
    # FIXME: implement (reduction should be marked as separate)

    # Load predictions
    predictions_run_db: List[ProjectPrediction] = []
    predictions_objects_db: List[ProjectPredictionObjectResults] = []
    predictions_missed_db: List[ProjectPredictionUnmatchedResults] = []
    for prediction_type in [MainPredictionType.OBJECT, MainPredictionType.CLASSIFICATION]:
        prediction_hash = uuid.uuid4()
        predictions_dir = pfs.predictions

        # Work out what prediction type we are loading and the correct folders to search for both variants of legacy
        # predictions storage. If they exist.
        if not predictions_dir.exists():
            continue
        predictions_child_dirs = list(predictions_dir.iterdir())
        if len(predictions_child_dirs) == 0:
            continue
        names = {child_dir.name for child_dir in predictions_child_dirs}
        if prediction_type.value in names and "predictions.csv" not in names:
            predictions_dir = predictions_dir / prediction_type.value
        elif prediction_type == MainPredictionType.CLASSIFICATION:
            # SKIP, will already be loaded by object classification type
            continue

        # Load all metadata for this prediction store.
        metrics_dir = pfs.metrics
        predictions_metric_datas = reader.get_prediction_metric_data(predictions_dir, metrics_dir)
        model_predictions = reader.get_model_predictions(predictions_dir, predictions_metric_datas, prediction_type)
        class_idx_dict = reader.get_class_idx(predictions_dir)
        gt_matched = reader.get_gt_matched(predictions_dir)

        label_metric_datas = reader.get_label_metric_data(metrics_dir)
        labels = reader.get_labels(predictions_dir, label_metric_datas, prediction_type)
        if gt_matched is None or labels is None or model_predictions is None:
            raise ValueError("Missing prediction files for migration!")

        # Setup inverse matching dictionary for calculating the label being matched against the
        gt_matched_inverted_list: List[Tuple[Tuple[int, int, int], int]] = [
            ((int(class_id), int(img_id), int(pidx)), int(mapping["lidx"]))
            for class_id, rest in gt_matched.items()
            for img_id, mapping_list in rest.items()
            for mapping in mapping_list
            for pidx in mapping["pidxs"]
        ]
        gt_matched_inverted_map = dict(gt_matched_inverted_list)
        if len(gt_matched_inverted_list) != len(gt_matched_inverted_map):
            raise ValueError(
                f"Inconsistency in prediction mapping lookup: "
                f"{len(gt_matched_inverted_list)} / {len(gt_matched_inverted_map)}"
            )

        # Tracking for applying associated metadata to the query.
        model_prediction_best_match_candidates: Dict[
            Tuple[uuid.UUID, int, str], List[ProjectPredictionObjectResults]
        ] = {}
        model_prediction_unmatched_indices = set(range(len(labels)))

        for model_prediction in model_predictions.to_dict(orient="records"):
            identifier: str = model_prediction.pop("identifier")
            model_prediction.pop("url")
            pidx = int(model_prediction.pop("Unnamed: 0"))
            img_id = int(model_prediction.pop("img_id"))
            class_id = int(model_prediction.pop("class_id"))
            feature_hash = class_idx_dict[str(class_id)]["featureHash"]
            confidence = float(model_prediction.pop("confidence"))
            theta = model_prediction.pop("theta", None)
            rle = str(model_prediction.pop("rle"))
            iou = float(model_prediction.pop("iou"))
            pbb: List[float] = [
                float(model_prediction.pop("x1")),
                float(model_prediction.pop("y1")),
                float(model_prediction.pop("x2")),
                float(model_prediction.pop("y2")),
            ]
            p_metrics: dict = {}
            f_metrics: dict = {}
            for metric_name, metric_value in model_prediction.items():
                metric_target = metric_name[-4:]
                metric_key = WELL_KNOWN_METRICS[metric_name[:-4]]
                if metric_key == "$SKIP":
                    continue
                if metric_target == " (P)":
                    if metric_key in AnnotationMetrics:
                        _assign_metrics(
                            metric_column_name=metric_key,
                            metrics_dict=p_metrics,
                            score=metric_value,
                            metric_types=AnnotationMetrics,
                            metrics_derived=set(),
                            error_identifier=identifier,
                        )
                    elif metric_key == "metric_object_density" or metric_key == "metric_object_count":
                        pass  # FIXME: this p-metric should be stored somewhere.
                    else:
                        print(f"DEBUG, Extra p_metric: {metric_key}")
                elif metric_target == " (F)":
                    f_metrics[metric_key] = metric_value
                else:
                    raise ValueError(f"Unknown metric target: '{metric_target}'")

            # Decompose identifier
            label_hash, du_hash_str, frame_str, *object_hashes = identifier.split("_")
            if len(object_hashes) == 0:
                raise ValueError(f"Missing label hash: {identifier}")
            du_hash = uuid.UUID(du_hash_str)
            frame = int(frame_str)

            # label_match_id => match_properties
            label_match_id = gt_matched_inverted_map.get((class_id, img_id, pidx), None)
            match_object_hash = None
            match_feature_hash = None
            if label_match_id is not None:
                matched_label = labels.iloc[label_match_id].to_dict()
                if int(matched_label.pop("Unnamed: 0")) != label_match_id:
                    raise ValueError(f"Inconsistent lookup: {label_match_id}")
                if label_match_id in model_prediction_unmatched_indices:
                    model_prediction_unmatched_indices.remove(label_match_id)
                matched_label_class_id = matched_label.pop("class_id")
                matched_label_img_id = matched_label.pop("img_id")
                matched_label_identifier = matched_label.pop("identifier")
                (
                    matched_label_label_hash,
                    matched_label_du_hash_str,
                    matched_label_frame_str,
                    *matched_label_hashes,
                ) = matched_label_identifier.split("_")
                if len(matched_label_hashes) != 1:
                    raise ValueError(f"Matched against multiple labels, this is invalid: {matched_label_identifier}")
                if uuid.UUID(matched_label_du_hash_str) != du_hash:
                    raise ValueError(f"Matched against different du_hash: {du_hash}")
                if int(matched_label_frame_str) != frame:
                    raise ValueError(f"Matched against different frame: {frame}")
                # FIXME: this has metrics - should they be used anywhere??

                # Set new values
                match_object_hash = matched_label_hashes[0]
                match_feature_hash = class_idx_dict[str(matched_label_class_id)]["featureHash"]

            # Add the new prediction to the database!!!
            for object_hash in object_hashes:
                new_predictions_object_db = ProjectPredictionObjectResults(
                    # Identifier
                    prediction_hash=prediction_hash,
                    du_hash=du_hash,
                    frame=frame,
                    object_hash=object_hash,
                    # Prediction metadata
                    feature_hash=feature_hash,
                    confidence=confidence,
                    iou=iou,
                    # FIXME: pbb (aim to support all prediction descriptors (bytes??)
                    # Ground truth match metadata
                    match_object_hash=match_object_hash,
                    match_feature_hash=match_feature_hash,
                    match_duplicate_iou=-1.0
                    if match_feature_hash is not None and match_feature_hash == feature_hash
                    else 1.0,  # 1.0 = always duplicate, -1.0 = never duplicate.
                    # Metrics
                    **assert_valid_args(ProjectPredictionObjectResults, p_metrics),
                )
                predictions_objects_db.append(new_predictions_object_db)
                if match_object_hash is not None:
                    # Assign all duplicate matches to match with non-highest iou
                    match_key = (du_hash, frame, match_object_hash)
                    all_match_candidates = model_prediction_best_match_candidates.setdefault(match_key, [])
                    all_match_candidates.append(new_predictions_object_db)

        # Post-process, determine metadata for dynamic duplicate detection.
        # A duplicate is a valid (feature hash match & iou >= THRESHOLD)
        # match with the highest confidence.
        # So we sort and detect the threshold where the result changes.
        for model_prediction_group in model_prediction_best_match_candidates.values():
            if len(model_prediction_group) <= 1:
                # No post-processing needed (feature hash mismatches are by default excluded anyway).
                continue
            model_prediction_group.sort(
                key=lambda m_obj: (m_obj.iou, m_obj.confidence, m_obj.object_hash), reverse=True
            )

            def confidence_compare_key(m_obj):
                return m_obj.confidence, m_obj.iou, m_obj.object_hash

            # Currently ordered by maximum iou, (first entry is trivially correct as it is the only one with
            #  the desired iou).
            current_best_confidence = None
            current_best_compare_key = None
            for model_prediction_entry in model_prediction_group:
                if model_prediction_entry.feature_hash == model_prediction_entry.match_feature_hash:
                    model_compare_key = confidence_compare_key(model_prediction_entry)
                    if current_best_confidence is None:
                        current_best_confidence = model_prediction_entry
                        current_best_compare_key = model_compare_key
                    elif current_best_compare_key is None:
                        raise ValueError(f"Correctness violation for sorted order: {model_prediction_group}")
                    elif model_compare_key > current_best_compare_key:
                        # IOU decrease has changed the maximum, assign thresholds.
                        current_best_confidence.match_duplicate_iou = model_prediction_entry.iou
                        current_best_confidence = model_prediction_entry
                        current_best_compare_key = model_compare_key
                    elif model_compare_key < current_best_compare_key:
                        # This model is always a duplicate
                        model_prediction_entry.match_duplicate_iou = model_prediction_entry.iou
                    else:
                        raise ValueError(f"Failed to generate deterministic ordering for iou: {model_prediction_group}")
                else:
                    # Never match as feature hash comparison is incorrect.
                    model_prediction_entry.match_duplicate_iou = 1.0

        # Add match failures to side table.
        for missing_label_idx in model_prediction_unmatched_indices:
            missing_label = labels.iloc[missing_label_idx].to_dict()
            if int(missing_label.pop("Unnamed: 0")) != missing_label_idx:
                raise ValueError(f"Inconsistent lookup: {missing_label}")
            missing_label_class_id = int(missing_label.pop("class_id"))
            missing_feature_hash = class_idx_dict[str(missing_label_class_id)]["featureHash"]
            missing_label_identifier = missing_label.pop("identifier")
            _missing_lh, missing_du_hash_str, missing_frame_str, *matched_label_hashes = missing_label_identifier.split(
                "_"
            )
            if len(matched_label_hashes) != 1:
                raise ValueError(f"Matched against multiple labels, this is invalid: {missing_label_identifier}")

            predictions_missed_db.append(
                ProjectPredictionUnmatchedResults(
                    prediction_hash=prediction_hash,
                    du_hash=uuid.UUID(missing_du_hash_str),
                    frame=int(missing_frame_str),
                    object_hash=str(matched_label_hashes[0]),
                    feature_hash=str(missing_feature_hash),
                )
            )

        # Ensure prediction currently stored actually exists.
        # FIXME: store prediction metadata for the prediction run.
        if len(predictions_run_db) == 0:
            predictions_run_db.append(
                ProjectPrediction(
                    project_hash=project_hash,
                    prediction_hash=prediction_hash,
                    name="Migrated Prediction",
                )
            )

    metrics_db_objects = [
        ProjectAnnotationAnalytics(
            project_hash=project_hash,
            du_hash=du_hash,
            frame=frame,
            object_hash=object_hash,
            **assert_valid_args(ProjectAnnotationAnalytics, metrics),
        )
        for (du_hash, frame, object_hash), metrics in annotation_metrics.items()
    ]

    metrics_db_data = [
        ProjectDataAnalytics(
            project_hash=project_hash, du_hash=du_hash, frame=frame, **assert_valid_args(ProjectDataAnalytics, metrics)
        )
        for (du_hash, frame), metrics in data_metrics.items()
    ]

    path = database_dir / "encord-active.sqlite"
    engine = get_engine(path)
    with Session(engine) as sess:
        sess.add(project)
        sess.add_all(data_metas)
        sess.add_all(data_units_metas)
        sess.add_all(metrics_db_objects)
        sess.add_all(metrics_db_data)
        sess.add_all(project_tag_definitions)
        sess.add_all(project_data_tags)
        sess.add_all(project_object_tags)
        sess.add_all(predictions_run_db)
        sess.add_all(predictions_objects_db)
        sess.add_all(predictions_missed_db)
        sess.commit()

        # Now correctly assign duplicates
        for prediction in predictions_run_db:
            prediction_hash = prediction.prediction_hash
