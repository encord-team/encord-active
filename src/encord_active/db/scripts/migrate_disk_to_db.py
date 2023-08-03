import json
import logging
import math
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from encord.objects import OntologyStructure
from encord.objects.common import PropertyType
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
    EmbeddingReductionType,
    Project,
    ProjectAnnotationAnalytics,
    ProjectAnnotationAnalyticsExtra,
    ProjectAnnotationAnalyticsReduced,
    ProjectDataAnalytics,
    ProjectDataAnalyticsExtra,
    ProjectDataAnalyticsReduced,
    ProjectDataMetadata,
    ProjectDataUnitMetadata,
    ProjectEmbeddingReduction,
    ProjectPrediction,
    ProjectPredictionAnalytics,
    ProjectPredictionAnalyticsExtra,
    ProjectPredictionAnalyticsFalseNegatives,
    ProjectTag,
    ProjectTaggedAnnotation,
    ProjectTaggedDataUnit,
    get_engine,
)
from encord_active.db.scripts.delete_project import delete_project_from_db
from encord_active.lib.common.data_utils import file_path_to_url, url_to_file_path
from encord_active.lib.common.utils import mask_to_polygon, rle_to_binary_mask
from encord_active.lib.db.connection import DBConnection, PrismaConnection
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.db.tags import Tag, TagScope
from encord_active.lib.embeddings.dimensionality_reduction import get_2d_embedding_data
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
    "Image Diversity": "metric_image_difficulty",  # NOTE: Renamed
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

DATA_LEVEL_ACTUALLY_CLASSIFICATION_METRICS: Set[str] = {
    "metric_annotation_quality",
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
    description_dict: Optional[Tuple[Dict[str, Union[bytes, float]], Dict[str, str]]],
    description: Optional[str],
):
    if metric_column_name not in metrics_dict:
        raw_score = score
        if metric_column_name in WELL_KNOWN_PERCENTAGE_METRICS:
            score = score / 100.0
        metric_def = metric_types[metric_column_name]
        # Run set of patches to generate correctly scaled results.
        if metric_column_name == "metric_sharpness":
            # Normalise metric sharpness
            # NOTE: max sharpness comes from 50% 255 & 50%
            # 127.5 assuming max = 255
            # However for 4x laplacian kernel, max = 4 * 255
            # and min is not 0, its -max
            # Not sure what laplacian kernel cv2 uses or if its
            # more complete. As had to increase the value from 1020
            # due to larger values being returned from the metric.
            score = math.sqrt(score) / (8.0 * 255.0)
        elif metric_column_name == "metric_label_shape_outlier":
            # NOTE: guesswork, not based on any analysis of hu moments
            score = score / 10000.0
            pass
        # Update the score
        if metric_def.type == MetricType.NORMAL:
            # Clamp 0->1 (some metrics do not generate correctly clamped scores.
            original_score = score
            score = min(max(score, 0.0), 1.0)
            if original_score != score and not (math.isnan(score) and math.isnan(original_score)):
                score_delta = float(abs(original_score - score))
                score_p1 = abs((score_delta / float(score)) * 100.0)
                score_p2 = abs((score_delta / float(original_score)) * 100.0)
                if score_p1 >= 0.5 or score_p2 >= 0.5:
                    logging.getLogger("MigrateDB").debug(
                        f"WARNING: clamped normal metric score[Original={raw_score}] - {metric_column_name}:"
                        f" {original_score} => {score}, "
                        f"Percentage = {score_p1}, {score_p2}"
                        f"identifier={error_identifier}"
                    )

        metrics_dict[metric_column_name] = score
    elif metric_column_name not in metrics_derived:
        raise ValueError(
            f"Duplicate metric assignment for, column={metric_column_name}," f"identifier={error_identifier}"
        )
    else:
        existing_score = metrics_dict[metric_column_name]
        if existing_score != score:
            score_delta = float(abs(existing_score - score))
            score_p1 = score_delta if score == 0.0 else abs((score_delta / float(score)) * 100.0)
            score_p2 = score_delta if existing_score == 0.0 else abs((score_delta / float(existing_score)) * 100.0)
            if score_p1 >= 0.5 or score_p2 >= 0.5:
                logging.getLogger("MigrateDB").debug(
                    f"WARNING: different derived and calculated scores: \n"
                    f"Percentage = {score_p1}, {score_p2}\n"
                    f"{metric_column_name}: existing={existing_score}, new={score}"
                )
    # Assign description
    if (
        description_dict is not None
        and description is not None
        and isinstance(description, str)
        and len(description) > 0
    ):
        _, metric_description_map = description_dict
        if metric_column_name not in metric_description_map:
            metric_description_map[metric_column_name] = description
        elif description not in metric_description_map[metric_column_name]:
            metric_description_map[metric_column_name] = str(
                metric_description_map[metric_column_name] + ", " + description
            )


def _load_embeddings(
    pfs: ProjectFileStructure,
    data_metric_extra: Dict[Tuple[uuid.UUID, int], Tuple[Dict[str, Union[bytes, float]], Dict[str, str]]],
    annotation_metric_extra: Dict[Tuple[uuid.UUID, int, str], Tuple[Dict[str, Union[bytes, float]], Dict[str, str]]],
) -> None:
    for embedding_type, embedding_name in [
        (EmbeddingType.IMAGE, "embedding_clip"),
        (EmbeddingType.OBJECT, "embedding_clip"),
        (EmbeddingType.CLASSIFICATION, "embedding_clip"),
        # FIXME: (EmbeddingType.HU_MOMENTS, "embedding_hu"),
    ]:
        label_embeddings = load_label_embeddings(embedding_type, pfs)
        for embedding in label_embeddings:
            du_hash = uuid.UUID(embedding["data_unit"])
            frame = int(embedding["frame"])
            annotation_hash: Optional[str] = (
                str(embedding["labelHash"]) if embedding.get("labelHash", None) is not None else None
            )
            embedding_bytes: bytes = embedding["embedding"].tobytes()
            if annotation_hash is not None:
                embedding_dict, _ = annotation_metric_extra[(du_hash, frame, annotation_hash)]
            else:
                embedding_dict, _ = data_metric_extra[(du_hash, frame)]
            embedding_dict[embedding_name] = embedding_bytes


def _load_embeddings_2d(
    pfs: ProjectFileStructure,
    project_hash: uuid.UUID,
    reduced_embeddings: List[ProjectEmbeddingReduction],
    data_reduced_embeddings: List[ProjectDataAnalyticsReduced],
    annotation_reduced_embeddings: List[ProjectAnnotationAnalyticsReduced],
) -> None:
    reduction_hash_map: Dict[str, uuid.UUID] = {}
    for embedding_type, embedding_name in [
        (EmbeddingType.IMAGE, "embedding_clip"),
        (EmbeddingType.OBJECT, "embedding_clip"),
        (EmbeddingType.CLASSIFICATION, "embedding_clip"),
    ]:
        reduced_embedding = get_2d_embedding_data(pfs, embedding_type)
        if reduced_embedding is not None:
            reduction_hash = reduction_hash_map.setdefault(embedding_name, uuid.uuid4())
            for entry in reduced_embedding.to_dict("records"):
                identifier = entry.pop("identifier")
                x = float(entry.pop("x"))
                y = float(entry.pop("y"))
                label = str(entry.pop("label"))  # FIXME: ignore?
                label_hash, du_hash_str, frame_str, *annotation_hash_list = identifier.split("_")
                du_hash = uuid.UUID(du_hash_str)
                frame = int(frame_str)
                if len(annotation_hash_list) == 0:
                    data_reduced_embeddings.append(
                        ProjectDataAnalyticsReduced(
                            project_hash=project_hash,
                            du_hash=du_hash,
                            frame=frame,
                            reduction_hash=reduction_hash,
                            x=x,
                            y=y,
                        )
                    )
                for annotation_hash in annotation_hash_list:
                    annotation_reduced_embeddings.append(
                        ProjectAnnotationAnalyticsReduced(
                            project_hash=project_hash,
                            du_hash=du_hash,
                            frame=frame,
                            object_hash=annotation_hash,
                            reduction_hash=reduction_hash,
                            x=x,
                            y=y,
                        )
                    )
    for k, v in reduction_hash_map.items():
        reduced_embeddings.append(
            ProjectEmbeddingReduction(
                reduction_hash=v,
                project_hash=project_hash,
                reduction_name=f"Migrated ({k})",
                reduction_description="",
                reduction_type=EmbeddingReductionType.UMAP,
                reduction_bytes="".encode("utf-8"),
            )
        )


# TP condition: E.iou >= :iou && E.match_iou < :iou (calculated via post-processing in function below)


_PREDICTION_MATCH_IOU_ALWAYS_FP: float = 1.0
_PREDICTION_MATCH_IOU_ALWAYS_TP: float = -1.0


def __prediction_iou_range_post_process(model_prediction_group: List[ProjectPredictionAnalytics]) -> None:
    if len(model_prediction_group) <= 1:
        single_model = model_prediction_group[0]
        if single_model.feature_hash != single_model.match_feature_hash:
            raise ValueError("Bug: should be filtered out earlier")

        # Unconditionally match as no clashes to have higher confidence.
        single_model.match_duplicate_iou = _PREDICTION_MATCH_IOU_ALWAYS_TP
    else:

        def confidence_compare_key(m_obj: ProjectPredictionAnalytics):
            return m_obj.metric_label_confidence, m_obj.iou, m_obj.object_hash

        def iou_compare_key(m_obj: ProjectPredictionAnalytics) -> Tuple[float, float, str]:
            return m_obj.iou, m_obj.metric_label_confidence, m_obj.object_hash

        # Consider each candidate in confidence order:
        # max-confidence: always the TP if it matches (if iou is valid and hence not null)
        # next-max-confidence:
        #  if n.iou > iou => NEVER match as:
        #    either BOTH >= :iou OR (iou >= :iou AND n.iou < :iou)
        #    for that case max-confidence is TP always hence next-max-confidence is always FP or null
        #  else:
        #    either BOTH >= :iou OR (n.iou >= :iou AND iou < :iou)
        #    for that case if iou < :iou:
        #       max-confidence = null
        #       next-max-confidence = TP
        #    else:
        #       max-confidence = TP
        #       next-max-confidence = FP
        # Therefore it follows:
        #  next-max-confidence is TP iff:
        #    n.iou >= :iou && iou < :iou
        #
        #
        # which generalizes (C -> N) where C.iou > N.iou and C.confidence > N.confidence:
        #   N is TP iff:
        #     N.iou >= :iou && C.iou < :iou
        #
        #
        # TP = { E where E.iou >= :iou AND E.confidence = max }
        #
        # case 1:
        #   A = {iou = 0.5, conf=0.5}
        #   B = {iou = 0.7, conf=0.5}
        #  => 2 resolutions: A = TP always, B = FP always OR (B = TP always, A = TP if iou < 0.5)
        #
        # case 2:
        #  A = {iou = 0.5, conf=0.3}
        #  B = {iou = 0.5, conf=0.7}
        #  => resolution B = TP always, B = FP always (required)

        model_prediction_group.sort(key=iou_compare_key, reverse=True)

        # Iterate over each model prediction group in order of decreasing confidence.
        # Each candidate will remain the TP for some iou range until a newer
        # the maximum iou value will be tracked to determine the iou range that the given entry is a true-positive.
        # The max confidence candidate is the TP until an entry with smaller iou is found.
        current_true_positive: ProjectPredictionAnalytics = model_prediction_group[0]
        current_true_positive.match_duplicate_iou = _PREDICTION_MATCH_IOU_ALWAYS_TP
        current_true_positive_confidence: float = current_true_positive.metric_label_confidence

        for model_prediction_entry in model_prediction_group[1:]:
            if model_prediction_entry.feature_hash != model_prediction_entry.match_feature_hash:
                raise ValueError("Bug: should be filtered out earlier")
            model_confidence: float = model_prediction_entry.metric_label_confidence
            if model_confidence > current_true_positive_confidence:
                # This model is considered the true positive whenever it matches, as it has a higher confidence
                # and smaller iou. The boundary on the parent remaining a true-positive is defined.
                model_prediction_entry.match_duplicate_iou = _PREDICTION_MATCH_IOU_ALWAYS_TP
                current_true_positive.match_duplicate_iou = model_prediction_entry.iou

                # Set the new true-positive for the new smaller iou value.
                current_true_positive = model_prediction_entry
                current_true_positive_confidence = model_confidence
            else:
                # This model is always a false postive, as there exists a prediction with higher confidence
                # that for the whole iou range this is valid that is the new true positive.
                model_prediction_entry.match_duplicate_iou = _PREDICTION_MATCH_IOU_ALWAYS_FP

        """
        # Verify query behaviour is as-expected. (FIXME: slow - remove this before release, this just ensures no bugs)
        for test_iou_S in range(101):
            test_iou = float(test_iou_S) / 100.0
            db_query = [
                m
                for m in model_prediction_group
                if m.iou >= test_iou > m.match_duplicate_iou
            ]
            correct_filter_verification = [
                m
                for m in model_prediction_group
                if m.iou >= test_iou
            ]
            correct_filter_verification.sort(key=confidence_compare_key, reverse=True)
            correct_filter_verification = correct_filter_verification[:1]
            if correct_filter_verification != db_query:
                raise ValueError(
                    f"Prediction calculation bug, iou={test_iou}:\n"
                    f"Correct result: {correct_filter_verification}\n"
                    f"DB Query result: {db_query}\n"
                    f"Debugging: {model_prediction_group}\n"
                )
        """


def __prediction_iou_bound_false_negative(
    key: Tuple[uuid.UUID, int, str],
    model_prediction_group: List[ProjectPredictionAnalytics],
    annotation_metrics: Dict[
        Tuple[uuid.UUID, int, str], Dict[str, Union[int, float, bytes, str, AnnotationType, None]]
    ],
    prediction_hash,
) -> ProjectPredictionAnalyticsFalseNegatives:
    du_hash, frame, annotation_hash = key
    iou_threshold = max((model.iou for model in model_prediction_group), default=2.0)  # Always a false negative

    if len(model_prediction_group) == 0:
        raise ValueError("Bugged Model Prediction Group")

    # iou: 0.1, 0.4, 0.6, 0.7 (filtered to valid iou)
    #  -> 0.1
    # hence iou < 0.1 : this is a FN
    # 2.0 for unconditional as 1.0 would otherwise return bad value at iou = 1.0
    missing_annotation_metrics = annotation_metrics[(du_hash, frame, annotation_hash)]
    return ProjectPredictionAnalyticsFalseNegatives(
        prediction_hash=prediction_hash,
        du_hash=du_hash,
        frame=frame,
        object_hash=annotation_hash,
        iou_threshold=iou_threshold,
        **assert_valid_args(ProjectPredictionAnalyticsFalseNegatives, missing_annotation_metrics),
    )


def __list_valid_child_feature_hashes(ontology_dict) -> Set[str]:
    ontology = OntologyStructure.from_dict(ontology_dict)
    valid_child_ontology_feature_hashes = set()
    for class_label in ontology.classifications:
        # Only allow nested to layer 1 IFF attributes[0] is RADIO && only 1 RADIO attribute.
        num_radio = len(
            [
                class_question
                for class_question in class_label.attributes
                if class_question.get_property_type() == PropertyType.RADIO
            ]
        )
        if num_radio != 1:
            continue
        class_question = class_label.attributes[0]
        if class_question.get_property_type() == PropertyType.RADIO:
            for counter, option in enumerate(class_question.options):
                valid_child_ontology_feature_hashes.add(option.feature_node_hash)

    return valid_child_ontology_feature_hashes


def __prediction_feature_hash(class_idx_dict: dict, class_id: int, is_classify: bool) -> str:
    class_id_meta = class_idx_dict[str(class_id)]
    return class_id_meta["optionHash"] if is_classify else class_id_meta["featureHash"]


def __label_classify_feature_hash(label, classification_answers, valid_classify_child_feature_hashes) -> str:
    classification_hash = str(label["classificationHash"])
    root_feature_hash = str(label["featureHash"])
    answers = classification_answers.get(classification_hash, None)
    if answers is None:
        return root_feature_hash
    for classify in answers["classifications"]:
        classify_answers = classify["answers"]
        if not isinstance(classify_answers, list):
            continue
        for answer in classify["answers"]:
            answer_feature_hash = str(answer["featureHash"])
            if answer_feature_hash in valid_classify_child_feature_hashes:
                return answer_feature_hash
    return root_feature_hash


def __migrate_predictions(
    predictions_dir: Path,
    metrics_dir: Path,
    prediction_type: MainPredictionType,
    annotation_metrics: Dict[
        Tuple[uuid.UUID, int, str], Dict[str, Union[int, float, bytes, str, AnnotationType, None]]
    ],
    data_metrics: Dict[Tuple[uuid.UUID, int], Dict[str, Union[int, float, bytes]]],
    prediction_hash: uuid.UUID,
    project_hash: uuid.UUID,
    predictions_annotations_db: List[ProjectPredictionAnalytics],
    predictions_annotations_db_extra: List[ProjectPredictionAnalyticsExtra],
) -> List[ProjectPredictionAnalyticsFalseNegatives]:
    # Used for validation
    predictions_missed_db = []
    predictions_existing_hashes: Set[Tuple[uuid.UUID, int, str]] = set()

    # Load all metadata for this prediction store.
    predictions_metric_datas = reader.read_prediction_metric_data(predictions_dir, metrics_dir)
    model_predictions = reader.read_model_predictions(predictions_dir, predictions_metric_datas, prediction_type)
    class_idx_dict = reader.read_class_idx(predictions_dir)
    gt_matched = reader.read_gt_matched(predictions_dir)

    label_metric_datas = reader.read_label_metric_data(metrics_dir)
    labels = reader.read_labels(predictions_dir, label_metric_datas, prediction_type)
    if (
        gt_matched is None
        and model_predictions is not None
        and labels is not None
        and prediction_type == MainPredictionType.CLASSIFICATION
    ):
        # Classification prediction projects do not store gt_matched
        #  generate equivalent function
        gt_matched = {}
        for model_prediction in model_predictions.to_dict(orient="records"):
            pidx = int(model_prediction["Unnamed: 0"])
            img_id = int(model_prediction["img_id"])
            class_id = int(model_prediction["class_id"])
            label_find_f = labels[labels["img_id"] == img_id]
            label_find = label_find_f[label_find_f["class_id"] == class_id].to_dict(orient="records")
            if len(label_find) > 0:
                mapping_list = gt_matched.setdefault(class_id, {}).setdefault(img_id, [])
                for label_find_entry in label_find:
                    lidx = int(label_find_entry["Unnamed: 0"])  # type: ignore
                    mapping_list.append({"lidx": lidx, "pidxs": [pidx]})

    if gt_matched is None or labels is None or model_predictions is None:
        raise ValueError(
            f"Missing prediction files for migration:\n"
            f" GT exists = {gt_matched is not None}\n"
            f" Labels exists = {labels is not None}\n"
            f" Model Predictions exists = {model_predictions is not None}\n"
            f" ==========\n"
            f" Prediction dir = {predictions_dir}\n"
            f" Metric dir = {metrics_dir}\n"
            f" Prediction type = {prediction_type}\n"
        )

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
    model_prediction_best_match_candidates: Dict[Tuple[uuid.UUID, int, str], List[ProjectPredictionAnalytics]] = {}
    model_prediction_unmatched_indices = set(range(len(labels)))

    is_classify = prediction_type == MainPredictionType.CLASSIFICATION
    generated_classify_hashes = set()
    for model_prediction in model_predictions.to_dict(orient="records"):
        identifier: str = model_prediction.pop("identifier")
        model_prediction.pop("url")
        pidx = int(model_prediction.pop("Unnamed: 0"))
        img_id = int(model_prediction.pop("img_id"))
        class_id = int(model_prediction.pop("class_id"))
        feature_hash = __prediction_feature_hash(class_idx_dict, class_id, is_classify)
        confidence = float(model_prediction.pop("confidence"))
        theta = model_prediction.pop("theta", None)  # FIXME ??
        rle = None if is_classify else str(model_prediction.pop("rle"))
        iou = 1.0 if is_classify else float(model_prediction.pop("iou"))
        pbb: List[float] = (
            []
            if is_classify
            else [
                float(model_prediction.pop("x1")),
                float(model_prediction.pop("y1")),
                float(model_prediction.pop("x2")),
                float(model_prediction.pop("y2")),
            ]
        )
        p_metrics: dict = {}
        f_metrics: dict = {}
        metric_name: str
        for metric_name, metric_value in model_prediction.items():  # type: ignore
            if metric_name.endswith(")"):
                metric_target = metric_name[-4:]
                metric_key = WELL_KNOWN_METRICS[metric_name[:-4]]
            else:
                # Non-annotated metrics (new version - harder to process)
                # attempt to guess what the metric should be associated with.
                metric_target = " (P)"
                if metric_name in [
                    "Random Values on Images",
                    "Aspect Ratio",
                    "Area",
                    "Image Difficulty",
                    "Image Diversity",
                    "Image Singularity",
                ]:
                    metric_target = " (F)"
                metric_key = WELL_KNOWN_METRICS[metric_name]
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
                        description_dict=None,
                        description=None,
                    )
                elif metric_key == "metric_object_density" or metric_key == "metric_object_count":
                    pass  # FIXME: this p-metric should be stored somewhere.
                else:
                    raise ValueError(f"Unknown prediction metric: {metric_key}")
            elif metric_target == " (F)":
                # Frame metrics (we have them saved already)
                if metric_key in f_metrics:
                    raise ValueError("Duplicate frame metric on prediction")
                f_metrics[metric_key] = metric_value
            else:
                raise ValueError(f"Unknown metric target: '{metric_target}'")

        # Decompose identifier
        label_hash, du_hash_str, frame_str, *object_hashes = identifier.split("_")
        if len(object_hashes) == 0:
            if is_classify:
                # create a fake classification hash (add hardening against duplicate hashes)
                new_hash = str(uuid.uuid4()).replace("-", "")[:8]
                while new_hash in generated_classify_hashes:
                    new_hash = str(uuid.uuid4()).replace("-", "")[:8]
                generated_classify_hashes.add(new_hash)
                object_hashes = [new_hash]
            else:
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
            match_feature_hash = __prediction_feature_hash(class_idx_dict, matched_label_class_id, is_classify)

        # Select annotation side tables for prediction
        frame_metrics = data_metrics[(du_hash, frame)]
        img_w: int = int(frame_metrics["metric_width"])
        img_h: int = int(frame_metrics["metric_height"])
        if is_classify:
            annotation_type = AnnotationType.CLASSIFICATION
            annotation_array = np.array([], dtype=np.single)
        elif rle is None or rle == "nan":
            annotation_type = AnnotationType.BOUNDING_BOX
            annotation_array = np.array(
                [pb / (img_w if i % 2 == 0 else img_h) for i, pb in enumerate(pbb)], dtype=np.single
            )
        else:
            try:
                rle_json = json.loads(rle.replace("'", '"'))
            except Exception:
                print(f"Invalid rle clause = {rle}")
                raise
            if rle_json["size"] != [img_h, img_w]:
                raise ValueError(f"Bad rle size: {rle_json['size']} != [{img_h},{img_w}]")
            mask = rle_to_binary_mask(rle_json)
            polygon, bbox = mask_to_polygon(mask)
            if polygon is None:
                annotation_type = AnnotationType.BITMASK
                annotation_array = np.array(rle_json["counts"], dtype=np.int_)
            else:
                annotation_type = AnnotationType.POLYGON
                annotation_array = np.array([[float(x) / img_w, float(y) / img_h] for x, y in polygon], dtype=np.single)

        # Add the new prediction to the database!!!
        if len(object_hashes) == 0:
            raise ValueError(f"Missing object hash: {identifier}")

        for object_hash in object_hashes:
            if (du_hash, frame, object_hash) in predictions_existing_hashes:
                raise ValueError(f"Duplicate (du_hash, frame, object_hash): {du_hash}, {frame}, {object_hash}")
            predictions_existing_hashes.add((du_hash, frame, object_hash))
            new_predictions_object_db = ProjectPredictionAnalytics(
                # Identifier
                prediction_hash=prediction_hash,
                du_hash=du_hash,
                frame=frame,
                object_hash=object_hash,
                # Prediction metadata
                project_hash=project_hash,
                feature_hash=feature_hash,
                metric_label_confidence=confidence,
                iou=iou,
                annotation_type=annotation_type,
                # Ground truth match metadata
                match_object_hash=match_object_hash,
                match_feature_hash=match_feature_hash,
                match_duplicate_iou=_PREDICTION_MATCH_IOU_ALWAYS_FP,  # Never match, overriden if match_object_hash is set.
                # Metrics
                **assert_valid_args(ProjectPredictionAnalytics, p_metrics),
            )
            predictions_annotations_db.append(new_predictions_object_db)
            predictions_annotations_db_extra.append(
                ProjectPredictionAnalyticsExtra(
                    # Identifier
                    prediction_hash=prediction_hash,
                    du_hash=du_hash,
                    frame=frame,
                    object_hash=object_hash,
                    # Extra annotation
                    annotation_bytes=annotation_array.tobytes(),
                )
            )
            if match_object_hash is not None and match_feature_hash is not None and match_feature_hash == feature_hash:
                # Assign all duplicate matches to match with non-highest iou
                match_key = (du_hash, frame, match_object_hash)
                all_match_candidates = model_prediction_best_match_candidates.setdefault(match_key, [])
                all_match_candidates.append(new_predictions_object_db)
            else:
                # Never return TP, always consider this false-positive
                new_predictions_object_db.match_duplicate_iou = _PREDICTION_MATCH_IOU_ALWAYS_FP

    # Post-process, determine metadata for dynamic duplicate detection.
    # A duplicate is a valid (feature hash match & iou >= THRESHOLD)
    # match with the highest confidence.
    # So we sort and detect the threshold where the result changes.
    for model_prediction_group in model_prediction_best_match_candidates.values():
        __prediction_iou_range_post_process(model_prediction_group)

    for key, model_prediction_group in model_prediction_best_match_candidates.items():
        predictions_missed_db.append(
            __prediction_iou_bound_false_negative(key, model_prediction_group, annotation_metrics, prediction_hash)
        )

    # Add match failures to side table.
    for missing_label_idx in model_prediction_unmatched_indices:
        missing_label = labels.iloc[missing_label_idx].to_dict()
        if int(missing_label.pop("Unnamed: 0")) != missing_label_idx:
            raise ValueError(f"Inconsistent lookup: {missing_label}")
        missing_label_class_id = int(missing_label.pop("class_id"))
        missing_feature_hash = __prediction_feature_hash(class_idx_dict, missing_label_class_id, is_classify)
        missing_label_identifier = missing_label.pop("identifier")
        _missing_lh, missing_du_hash_str, missing_frame_str, *matched_label_hashes = missing_label_identifier.split("_")
        if len(matched_label_hashes) != 1:
            raise ValueError(f"Matched against multiple labels, this is invalid: {missing_label_identifier}")

        missing_du_hash = uuid.UUID(missing_du_hash_str)
        missing_frame = int(missing_frame_str)
        missing_annotation_hash = str(matched_label_hashes[0])
        missing_annotation_metrics = annotation_metrics[(missing_du_hash, missing_frame, missing_annotation_hash)]
        if str(missing_feature_hash) != missing_annotation_metrics["feature_hash"]:
            raise ValueError("Inconsistent feature_hash for missed annotation!")
        predictions_missed_db.append(
            ProjectPredictionAnalyticsFalseNegatives(
                prediction_hash=prediction_hash,
                du_hash=missing_du_hash,
                frame=missing_frame,
                object_hash=missing_annotation_hash,
                iou_threshold=-1.0,  # Always a false negative
                **assert_valid_args(ProjectPredictionAnalyticsFalseNegatives, missing_annotation_metrics),
            )
        )

    # Ensure the same number of predictions are defined
    if len(predictions_missed_db) != len(labels):
        raise ValueError(
            "Bug while generating missed prediction array: "
            f"{len(predictions_missed_db)} != {len(gt_matched_inverted_list)}"
        )

    return predictions_missed_db


def migrate_disk_to_db(pfs: ProjectFileStructure, delete_existing_project: bool = False) -> None:
    project_meta = fetch_project_meta(pfs.project_dir)
    project_hash: uuid.UUID = uuid.UUID(project_meta["project_hash"])
    annotation_metrics: Dict[
        Tuple[uuid.UUID, int, str], Dict[str, Union[int, float, bytes, str, AnnotationType, None]]
    ] = {}
    data_metrics: Dict[Tuple[uuid.UUID, int], Dict[str, Union[int, float, bytes]]] = {}
    data_classification_metrics: Dict[
        Tuple[uuid.UUID, int], List[Dict[str, Union[int, float, bytes, str, AnnotationType, None]]]
    ] = {}
    data_metric_extra: Dict[Tuple[uuid.UUID, int], Tuple[Dict[str, Union[bytes, float]], Dict[str, str]]] = {}
    annotation_metric_extra: Dict[
        Tuple[uuid.UUID, int, str], Tuple[Dict[str, Union[bytes, float]], Dict[str, str]]
    ] = {}

    data_to_classifications: Dict[Tuple[uuid.UUID, int], List[str]] = {}

    database_dir = pfs.project_dir.parent.expanduser().resolve()

    # Load metadata
    with PrismaConnection(pfs) as conn:
        ontology_dict = json.loads(pfs.ontology.read_text(encoding="utf-8"))
        project = Project(
            project_hash=project_hash,
            project_name=project_meta["project_title"],
            project_description=project_meta.get("project_description", ""),
            project_remote_ssh_key_path=project_meta["ssh_key_path"] if project_meta.get("has_remote", False) else None,
            project_ontology=ontology_dict,
        )
        valid_classify_child_feature_hashes = __list_valid_child_feature_hashes(ontology_dict)
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
            classification_answers: dict = label_row_json.get("classification_answers", {})
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
            expected_data_units: Set[str] = set(data_units_json.keys())
            for data_unit in label_row.data_units or []:
                du_hash = uuid.UUID(data_unit.data_hash)
                du_json = data_units_json[data_unit.data_hash]
                if data_type != "video":
                    expected_data_units.remove(data_unit.data_hash)
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
                        "annotation_email": str(obj["createdBy"]),
                        "annotation_manual": bool(obj["manualAnnotation"]),
                        "metric_label_confidence": float(obj["confidence"]),
                    }
                    annotation_metric_extra[(du_hash, data_unit.frame, object_hash)] = ({}, {})
                classifications = labels_json.get("classifications", [])
                for classify in classifications:
                    classification_hash = str(classify["classificationHash"])
                    if classification_hash in object_hashes_seen:
                        raise ValueError(
                            f"Duplicate object_hash/classification_hash={classification_hash} "
                            f"in du_hash={du_hash}, frame={data_unit.frame}"
                        )
                    object_hashes_seen.add(classification_hash)
                    data_to_classifications.setdefault((du_hash, data_unit.frame), []).append(classification_hash)
                    annotation_metrics[(du_hash, data_unit.frame, classification_hash)] = {
                        "feature_hash": __label_classify_feature_hash(
                            classify, classification_answers, valid_classify_child_feature_hashes
                        ),
                        "annotation_type": AnnotationType.CLASSIFICATION,
                        "annotation_email": str(classify["createdBy"]),
                        "annotation_manual": bool(classify["manualAnnotation"]),
                        "metric_label_confidence": float(classify["confidence"]),
                    }
                    annotation_metric_extra[(du_hash, data_unit.frame, classification_hash)] = ({}, {})
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
                data_metric_extra[(du_hash, data_unit.frame)] = ({}, {})
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
                    width=data_unit.width,
                    height=data_unit.height,
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
            if len(expected_data_units) > 0 and data_type != "video":
                raise ValueError(
                    f"Validation failure: prisma db missed data units for label row: \n"
                    f"For: {project_data_meta.label_hash}/{expected_data_units}\n"
                    f"Label row JSON: {project_data_meta.label_row_json}\n"
                    f"Prisma DB State: {label_row}\n"
                )
            project_data_meta.frames_per_second = fps
            data_metas.append(project_data_meta)

        # Migrate prisma db tag definitions.
        tag_id_map: Dict[int, uuid.UUID] = {}
        tag_name_map: Dict[str, uuid.UUID] = {}
        project_tag_definitions = []
        for tag in conn.tag.find_many():
            tag_uuid = uuid.uuid4()
            tag_id_map[tag.id] = tag_uuid
            tag_name_map[tag.name] = tag_uuid
            project_tag_definitions.append(
                ProjectTag(
                    tag_hash=tag_uuid,
                    project_hash=project_hash,
                    name=tag.name,
                    description="",
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

    # Migrate merged metric tag definitions.
    with DBConnection(pfs) as metric_conn:
        all_metrics = MergedMetrics(metric_conn).all()
        all_metrics = all_metrics.reset_index()
        all_metrics_records = all_metrics.to_dict("records")
        for merged_metric in all_metrics_records:
            _, du_hash_str, frame_str, *annotation_hashes = merged_metric["identifier"].split("_")
            du_hash = uuid.UUID(du_hash_str)
            frame = int(frame_str)
            metric_tags: List[Tag] = merged_metric["tags"]
            for metric_tag in metric_tags:
                if metric_tag.name in tag_name_map:
                    tag_uuid = tag_name_map[metric_tag.name]
                else:
                    tag_uuid = uuid.uuid4()
                    tag_name_map[metric_tag.name] = tag_uuid
                    project_tag_definitions.append(
                        ProjectTag(
                            tag_hash=tag_uuid,
                            project_hash=project_hash,
                            name=metric_tag.name,
                            description="",
                        )
                    )
                if len(annotation_hashes) == 0 or metric_tag.scope == TagScope.DATA:
                    project_data_tags.append(
                        ProjectTaggedDataUnit(
                            project_hash=project_hash,
                            du_hash=du_hash,
                            frame=frame,
                            tag_hash=tag_uuid,
                        )
                    )
                else:
                    for annotation_hash in annotation_hashes:
                        project_object_tags.append(
                            ProjectTaggedAnnotation(
                                project_hash=project_hash,
                                du_hash=du_hash,
                                frame=frame,
                                object_hash=annotation_hash,
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
            if (du_hash, frame) not in data_metrics:
                raise ValueError(
                    f"Metric references invalid frame!:\n"
                    f"du_hash={du_hash}, frame={frame}, objects?={object_hash_list}\n"
                    f"identifier={metric_entry['identifier']} metric={metric}\n"
                )

            # 1 classification metric - does not actually assign classification metric hash
            if len(object_hash_list) == 0 and metric_column_name in DATA_LEVEL_ACTUALLY_CLASSIFICATION_METRICS:
                object_hash_list = data_to_classifications[(du_hash, frame)]

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
                        description_dict=annotation_metric_extra[(du_hash, frame, object_hash)],
                        description=metric_entry.get("description", None),
                    )
            else:
                _assign_metrics(
                    metric_column_name=metric_column_name,
                    metrics_dict=data_metrics[(du_hash, frame)],
                    score=metric_entry["score"],
                    metric_types=DataMetrics,
                    metrics_derived=DERIVED_DATA_METRICS,
                    error_identifier=metric_entry["identifier"],
                    description_dict=data_metric_extra[(du_hash, frame)],
                    description=metric_entry.get("description", None),
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
                            description_dict=None,
                            description=None,
                        )

    # Load embeddings
    _load_embeddings(pfs=pfs, data_metric_extra=data_metric_extra, annotation_metric_extra=annotation_metric_extra)

    # Load 2d embeddings
    data_reduced_embeddings: List[ProjectDataAnalyticsReduced] = []
    annotation_reduced_embeddings: List[ProjectAnnotationAnalyticsReduced] = []
    reduced_embeddings: List[ProjectEmbeddingReduction] = []
    _load_embeddings_2d(
        pfs=pfs,
        project_hash=project_hash,
        reduced_embeddings=reduced_embeddings,
        data_reduced_embeddings=data_reduced_embeddings,
        annotation_reduced_embeddings=annotation_reduced_embeddings,
    )

    # Load predictions
    predictions_run_db: List[ProjectPrediction] = []
    predictions_annotations_db: List[ProjectPredictionAnalytics] = []
    predictions_annotations_db_extra: List[ProjectPredictionAnalyticsExtra] = []
    predictions_missed_db: List[ProjectPredictionAnalyticsFalseNegatives] = []
    for prediction_type in [MainPredictionType.OBJECT, MainPredictionType.CLASSIFICATION]:
        prediction_hash = uuid.uuid4()
        predictions_dir = pfs.predictions

        # Work out what prediction type we are loading and the correct folders to search for both variants of legacy
        # predictions storage. If they exist.
        if not predictions_dir.exists():
            continue  # Case 1: no prediction folder

        predictions_child_dirs = list(predictions_dir.iterdir())
        if len(predictions_child_dirs) == 0:
            continue  # Case 2: empty prediction folder

        # Select between different prediction types:
        #  if no object / classification folder but prediction folder exists => use folder & always object
        names = {child_dir.name for child_dir in predictions_child_dirs}
        if "predictions.csv" not in names:
            # 1 or 2 predictions for objects / classifications in the child structure
            predictions_dir = predictions_dir / prediction_type.value
        elif prediction_type == MainPredictionType.CLASSIFICATION:
            # 1 prediction for objects - hence classification predictions should be skipped.
            continue
        else:
            # 1 prediction for objects at the root folder
            pass

        # Not check if child folder is valid
        if not predictions_dir.exists():
            continue  # Case 3: one of CLASSIFICATION or OBJECT exists

        extend_missed_db = __migrate_predictions(
            predictions_dir=predictions_dir,
            metrics_dir=pfs.metrics,
            prediction_type=prediction_type,
            data_metrics=data_metrics,
            annotation_metrics=annotation_metrics,
            prediction_hash=prediction_hash,
            project_hash=project_hash,
            predictions_annotations_db=predictions_annotations_db,
            predictions_annotations_db_extra=predictions_annotations_db_extra,
        )
        predictions_missed_db.extend(extend_missed_db)
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

    metrics_db_objects_extra = [
        ProjectAnnotationAnalyticsExtra(
            project_hash=project_hash,
            du_hash=du_hash,
            frame=frame,
            object_hash=object_hash,
            metric_metadata=metric_metadata,
            **assert_valid_args(ProjectAnnotationAnalyticsExtra, embeddings),
        )
        for (du_hash, frame, object_hash), (embeddings, metric_metadata) in annotation_metric_extra.items()
    ]

    metrics_db_data = [
        ProjectDataAnalytics(
            project_hash=project_hash, du_hash=du_hash, frame=frame, **assert_valid_args(ProjectDataAnalytics, metrics)
        )
        for (du_hash, frame), metrics in data_metrics.items()
    ]

    metrics_db_data_extra = [
        ProjectDataAnalyticsExtra(
            project_hash=project_hash,
            du_hash=du_hash,
            frame=frame,
            metric_metadata=metric_metadata,
            **assert_valid_args(ProjectDataAnalyticsExtra, embeddings),
        )
        for (du_hash, frame), (embeddings, metric_metadata) in data_metric_extra.items()
    ]

    path = database_dir / "encord-active.sqlite"
    engine = get_engine(path)
    if delete_existing_project:
        delete_project_from_db(engine, project_hash)
    with Session(engine) as sess:
        sess.add(project)
        sess.add_all(data_metas)
        sess.add_all(data_units_metas)
        sess.add_all(metrics_db_objects)
        sess.add_all(metrics_db_objects_extra)
        sess.add_all(metrics_db_data)
        sess.add_all(metrics_db_data_extra)
        sess.add_all(reduced_embeddings)
        sess.add_all(data_reduced_embeddings)
        sess.add_all(annotation_reduced_embeddings)
        sess.add_all(project_tag_definitions)
        sess.add_all(project_data_tags)
        sess.add_all(project_object_tags)
        sess.add_all(predictions_run_db)
        sess.add_all(predictions_annotations_db)
        sess.add_all(predictions_annotations_db_extra)
        sess.add_all(predictions_missed_db)
        sess.commit()

        # Now correctly assign duplicates
        for prediction in predictions_run_db:
            prediction_hash = prediction.prediction_hash
