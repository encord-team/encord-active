import logging
import math
import random
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Type, TypeVar, Union, cast

import av
import encord
import numpy as np
import torch
import torchvision
from PIL import Image
from pynndescent import NNDescent
from torch import Tensor
from torchvision.io import VideoReader
from torchvision.ops import masks_to_boxes
from tqdm import tqdm

from encord_active.analysis.config import AnalysisConfig
from encord_active.analysis.types import (
    AnnotationMetadata,
    MetricDependencies,
    NearestNeighbors,
    RandomSampling,
)
from encord_active.analysis.util.torch import (
    obj_to_mask,
    obj_to_points,
    pillow_to_tensor,
)
from encord_active.db.models import (
    AnnotationType,
    ProjectAnnotationAnalytics,
    ProjectAnnotationAnalyticsExtra,
    ProjectDataAnalytics,
    ProjectDataAnalyticsExtra,
    ProjectDataMetadata,
    ProjectDataUnitMetadata,
    ProjectPredictionAnalytics,
    ProjectPredictionAnalyticsExtra,
    ProjectPredictionAnalyticsFalseNegatives,
)
from encord_active.lib.common.data_utils import download_image, url_to_file_path

from ..db.enums import annotation_type_from_str
from ..db.metrics import MetricType
from ..lib.encord.utils import get_encord_project
from .base import BaseAnalysis, BaseFrameBatchInput, BaseFrameInput, BaseMetric
from .embedding import BaseEmbedding, NearestImageEmbeddingQuery
from .metric import RandomSamplingQuery

TTypeAnnotationAnalytics = TypeVar("TTypeAnnotationAnalytics", ProjectAnnotationAnalytics, ProjectPredictionAnalytics)
TTypeAnnotationAnalyticsExtra = TypeVar(
    "TTypeAnnotationAnalyticsExtra", ProjectAnnotationAnalyticsExtra, ProjectPredictionAnalyticsExtra
)


class Executor:
    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config


class SimpleExecutor(Executor):
    def __init__(self, config: AnalysisConfig) -> None:
        super().__init__(config=config)
        self.data_metrics: Dict[Tuple[uuid.UUID, int], MetricDependencies] = {}
        self.annotation_metrics: Dict[Tuple[uuid.UUID, int, str], MetricDependencies] = {}
        self.prediction_false_negatives: Dict[Tuple[uuid.UUID, int, str], Tuple[str, float]] = {}
        self.encord_projects: Dict[uuid.UUID, encord.Project] = {}

    def execute_from_db(
        self,
        data_meta: List[ProjectDataMetadata],
        du_meta: List[ProjectDataUnitMetadata],
        database_dir: Path,
        project_hash: uuid.UUID,
        project_ssh_path: Optional[str],
    ) -> Tuple[
        List[ProjectDataAnalytics],
        List[ProjectDataAnalyticsExtra],
        List[ProjectAnnotationAnalytics],
        List[ProjectAnnotationAnalyticsExtra],
    ]:
        du_dict: Dict[uuid.UUID, List[ProjectDataUnitMetadata]] = {}
        for du in du_meta:
            du_dict.setdefault(du.data_hash, []).append(du)
        values = [(data, du_dict[data.data_hash]) for data in data_meta]
        self.execute(
            values,
            database_dir=database_dir,
            project_hash=project_hash,
            project_ssh_path=project_ssh_path,
        )
        return self._pack_output(project_hash=project_hash)

    def execute_prediction_from_db(
        self,
        data_meta: List[ProjectDataMetadata],
        ground_truth_annotation_meta: List[ProjectDataUnitMetadata],
        predicted_annotation_meta: List[ProjectDataUnitMetadata],
        database_dir: Path,
        project_hash: uuid.UUID,
        prediction_hash: uuid.UUID,
        project_ssh_path: Optional[str],
    ) -> Tuple[
        List[ProjectPredictionAnalytics],
        List[ProjectPredictionAnalyticsExtra],
        List[ProjectPredictionAnalyticsFalseNegatives],
    ]:
        du_dict: Dict[uuid.UUID, List[ProjectDataUnitMetadata]] = {}
        for du in predicted_annotation_meta:
            du_dict.setdefault(du.data_hash, []).append(du)
        values = [(data, du_dict[data.data_hash]) for data in data_meta]
        gt_lookup = {(gt.du_hash, gt.frame): gt for gt in ground_truth_annotation_meta}
        self.execute(
            values,
            database_dir=database_dir,
            project_hash=project_hash,
            project_ssh_path=project_ssh_path,
            gt_lookup=gt_lookup,
        )
        return self._pack_prediction(project_hash=project_hash, prediction_hash=prediction_hash)

    def execute(
        self,
        values: List[Tuple[ProjectDataMetadata, List[ProjectDataUnitMetadata]]],
        database_dir: Path,
        project_hash: uuid.UUID,
        project_ssh_path: Optional[str],
        gt_lookup: Optional[Dict[Tuple[uuid.UUID, int], ProjectDataUnitMetadata]] = None,
        batch: Optional[int] = None,  # Disable the batch execution path, use legacy code-path instead.
    ) -> None:
        # Split into videos & images
        video_values = [v for v in values if v[0].data_type == "video"]
        image_values = [(d, dle) for d, dl in values if d.data_type != "video" for dle in dl]
        image_values.sort(key=lambda d: (d[1].width, d[1].height))
        for data_metadata, du_metadata_list in video_values:
            # Well-formed
            du_metadata_list.sort(key=lambda v: (v.du_hash, v.frame))
            if data_metadata.data_type != "video":
                raise ValueError(f"Bug separating videos: {data_metadata.data_hash}")

            # Run assertions
            if du_metadata_list[0].du_hash != du_metadata_list[-1].du_hash:
                raise ValueError(f"Video has different du_hash: {data_metadata.data_hash}")
            if du_metadata_list[-1].frame + 1 != len(du_metadata_list) or du_metadata_list[0].frame != 0:
                raise ValueError(f"Video has skipped frames: {data_metadata.data_hash}")
            if len({x.frame for x in du_metadata_list}) != len(du_metadata_list):
                raise ValueError(f"Video has frame duplicates: {data_metadata.data_hash}")

        # Video best thing
        with torch.no_grad():
            # Stage 1
            if batch is None:
                self._stage_1_video_no_batching(
                    video_values,
                    database_dir=database_dir,
                    project_hash=project_hash,
                    project_ssh_path=project_ssh_path,
                    gt_lookup=gt_lookup,
                )
                self._stage_1_image_no_batching(
                    image_values,
                    database_dir=database_dir,
                    project_hash=project_hash,
                    project_ssh_path=project_ssh_path,
                    gt_lookup=gt_lookup,
                )
            else:
                self._stage_1_video_batching(
                    video_values,
                    database_dir=database_dir,
                    project_hash=project_hash,
                    project_ssh_path=project_ssh_path,
                    gt_lookup=gt_lookup,
                )
                self._stage_1_image_batching(
                    image_values,
                    database_dir=database_dir,
                    project_hash=project_hash,
                    project_ssh_path=project_ssh_path,
                    batching=batch,
                    gt_lookup=gt_lookup,
                )

            # Post-processing stages
            self._stage_2()
            self._stage_3()

    def _stage_1_video_batching(
        self,
        videos: List[Tuple[ProjectDataMetadata, List[ProjectDataUnitMetadata]]],
        database_dir: Path,
        project_hash: uuid.UUID,
        project_ssh_path: Optional[str],
        gt_lookup: Optional[Dict[Tuple[uuid.UUID, int], ProjectDataUnitMetadata]],
    ):
        if len(videos) == 0:
            return
        # Select video backend
        if self.config.device == "cuda" or self.config.device.type == "cuda":
            torchvision.set_video_backend("cuda")
        else:
            torchvision.set_video_backend("pyav")
        # Start execution
        for data_metadata, du_meta_list in tqdm(
            videos, desc=f"Metric Computation Stage 1: Videos ({torchvision.get_video_backend()})"
        ):
            video_start = du_meta_list[0]
            video_url_or_path = self._get_url_or_path(
                uri=video_start.data_uri,
                database_dir=database_dir,
                project_hash=project_hash,
                project_ssh_path=project_ssh_path,
                label_hash=data_metadata.label_hash,
                du_hash=video_start.du_hash,
            )
            reader = VideoReader(
                src=str(video_url_or_path),
                stream="video:0",
            )
            frame_metadata = reader.get_metadata()
            frame_iterator = enumerate(reader)
            first_frame = next(frame_iterator, None)
            if first_frame is not None:
                # Execute first frame
                pass
            for frame_idx, frame in frame_iterator:
                data: torch.Tensor = frame["data"]
                pts: float = frame["pts"]  # FIXME: use for frame_idx validation?!

    def _stage_1_video_no_batching(
        self,
        videos: List[Tuple[ProjectDataMetadata, List[ProjectDataUnitMetadata]]],
        database_dir: Path,
        project_hash: uuid.UUID,
        project_ssh_path: Optional[str],
        gt_lookup: Optional[Dict[Tuple[uuid.UUID, int], ProjectDataUnitMetadata]],
    ):
        if len(videos) == 0:
            return
        for data_metadata, du_meta_list in tqdm(videos, desc="Metric Computation Stage 1: Videos"):
            video_start = du_meta_list[0]
            video_url_or_path = self._get_url_or_path(
                uri=video_start.data_uri,
                database_dir=database_dir,
                project_hash=project_hash,
                project_ssh_path=project_ssh_path,
                label_hash=data_metadata.label_hash,
                du_hash=video_start.du_hash,
            )
            frames: List[Tuple[BaseFrameInput, Optional[Dict[str, AnnotationMetadata]]]] = []
            iter_frame_num = -1
            # FIXME: use torchvision VideoReader!!
            with av.open(str(video_url_or_path), mode="r") as container:
                for frame in tqdm(
                    container.decode(video=0),
                    desc=f" - Processing Video: {data_metadata.data_title}",
                    total=container.streams.video[0].frames,
                    leave=False,
                ):
                    frame_num: int = frame.index
                    frame_du_meta = du_meta_list[frame_num]
                    if frame_du_meta.frame != frame_num:
                        raise ValueError(f"Frame metadata mismatch: {frame_du_meta.frame} != {frame_num}")
                    if frame.is_corrupt:
                        raise ValueError(f"Corrupt video frame: {frame_num}")
                    if iter_frame_num + 1 != frame_num:
                        raise ValueError(f"AV video frame out of sequence: {iter_frame_num} -> {frame_num}")
                    iter_frame_num = frame_num
                    frame_image = frame.to_image().convert("RGB")
                    frame_input = self._get_frame_input(
                        image=frame_image,
                        du_meta=frame_du_meta,
                    )
                    gt = None if gt_lookup is None else gt_lookup[frame_du_meta.du_hash, frame_du_meta.frame]
                    if gt is None:
                        gt_input = None
                    else:
                        gt_input, _ignore = self._get_frame_annotations(
                            gt, image_width=frame_image.width, image_height=frame_image.height
                        )
                    frames.append((frame_input, gt_input))

                    # Run stage 1 metrics
                    if len(frames) == 3:
                        # Process non-first & non-last frame
                        # FIXME: seq metrics may work better with (curr, next, next_next)
                        #  need to sanity check on changes to design before fully enabling.
                        #  video metrics.
                        self._execute_stage_1_metrics_no_batching(
                            prev_frame=frames[0][0],
                            current_frame=frames[1][0],
                            next_frame=frames[1][0],
                            current_frame_gt=frames[1][1],
                            du_hash=du_meta_list[0].du_hash,
                            frame=iter_frame_num - 1,
                        )
                        frames = frames[1:]
                    elif len(frames) == 2:
                        # Process first frame
                        self._execute_stage_1_metrics_no_batching(
                            prev_frame=None,
                            current_frame=frames[0][0],
                            next_frame=frames[1][0],
                            current_frame_gt=frames[0][1],
                            du_hash=du_meta_list[0].du_hash,
                            frame=iter_frame_num - 1,
                        )

                # Process final frame (special case handling for 1-frame video as well)
                if len(frames) == 1 or len(frames) == 2:
                    self._execute_stage_1_metrics_no_batching(
                        prev_frame=frames[-2][0] if len(frames) == 2 else None,
                        current_frame=frames[-1][0],
                        next_frame=None,
                        current_frame_gt=frames[-1][1],
                        du_hash=du_meta_list[0].du_hash,
                        frame=iter_frame_num,
                    )
                else:
                    raise ValueError("Video frame processing bug")

    def _stage_1_image_no_batching(
        self,
        images: List[Tuple[ProjectDataMetadata, ProjectDataUnitMetadata]],
        database_dir: Path,
        project_hash: uuid.UUID,
        project_ssh_path: Optional[str],
        gt_lookup: Optional[Dict[Tuple[uuid.UUID, int], ProjectDataUnitMetadata]],
    ):
        for data_metadata, du_meta in tqdm(images, desc="Metric Computation Stage 1: Images"):
            gt = None if gt_lookup is None else gt_lookup[du_meta.du_hash, du_meta.frame]
            image = self._open_image(
                uri=du_meta.data_uri,
                database_dir=database_dir,
                project_hash=project_hash,
                project_ssh_path=project_ssh_path,
                label_hash=data_metadata.label_hash,
                du_hash=du_meta.du_hash,
            )
            frame_input = self._get_frame_input(
                image=image,
                du_meta=du_meta,
            )
            if gt is None:
                gt_input = None
            else:
                gt_input, _ignore = self._get_frame_annotations(gt, image_width=image.width, image_height=image.height)
            self._execute_stage_1_metrics_no_batching(
                prev_frame=None,
                current_frame=frame_input,
                next_frame=None,
                current_frame_gt=gt_input,
                du_hash=du_meta.du_hash,
                frame=du_meta.frame,
            )

    def _stage_1_image_batching(
        self,
        values: List[Tuple[ProjectDataMetadata, ProjectDataUnitMetadata]],
        database_dir: Path,
        project_hash: uuid.UUID,
        project_ssh_path: Optional[str],
        batching: int,
        gt_lookup: Optional[Dict[Tuple[uuid.UUID, int], ProjectDataUnitMetadata]],
    ) -> None:
        grouped_values: Dict[Tuple[int, int], List[Tuple[ProjectDataMetadata, ProjectDataUnitMetadata]]] = {}
        for data_metadata, du_metadata in values:
            grouped_values.setdefault((du_metadata.width, du_metadata.height), []).append((data_metadata, du_metadata))
        for group in grouped_values.values():
            for i in range(0, len(group), batching):
                grouped_batch = group[i : i + batching]
                frame_inputs = [
                    self._get_frame_input(
                        image=self._open_image(
                            uri=du_meta.data_uri,
                            database_dir=database_dir,
                            project_hash=project_hash,
                            project_ssh_path=project_ssh_path,
                            label_hash=meta.label_hash,
                            du_hash=du_meta.du_hash,
                        ),
                        du_meta=du_meta,
                    )
                    for meta, du_meta in grouped_batch
                ]

                batch_inputs = BaseFrameBatchInput(
                    images=torch.stack([frame.image for frame in frame_inputs]),
                    images_deps={},
                    annotations=None,  # FIXME: actually assign these values to something correct!!
                )

                # Execute
                image_results = {}
                object_results = {}
                classification_results = {}
                for evaluation in self.config.analysis:
                    metric_result = evaluation.raw_calculate_batch(
                        prev_frame=None,
                        frame=batch_inputs,
                        next_frame=None,
                    )
                    non_ephemeral = isinstance(evaluation, BaseAnalysis) or isinstance(evaluation, BaseEmbedding)
                    if metric_result.images is not None:
                        batch_inputs.images_deps[evaluation.ident] = metric_result.images
                        if non_ephemeral:
                            image_results[evaluation.ident] = metric_result.images
                    if metric_result.objects is not None:
                        if batch_inputs.annotations is not None:
                            batch_inputs.annotations.objects_deps[evaluation.ident] = metric_result.objects
                            if non_ephemeral:
                                object_results[evaluation.ident] = metric_result.objects
                        else:
                            raise ValueError(f"Object result but no annotations: {evaluation.ident}")
                    if metric_result.classifications is not None:
                        if batch_inputs.annotations is not None:
                            batch_inputs.annotations.classifications_deps[
                                evaluation.ident
                            ] = metric_result.classifications
                            if non_ephemeral:
                                classification_results[evaluation.ident] = metric_result.classifications
                        else:
                            raise ValueError(f"Object result but no annotations: {evaluation.ident}")

    def _open_image(
        self,
        uri: Optional[str],
        database_dir: Path,
        project_hash: uuid.UUID,
        project_ssh_path: Optional[str],
        label_hash: uuid.UUID,
        du_hash: uuid.UUID,
    ):
        img_url_or_path = self._get_url_or_path(
            uri=uri,
            database_dir=database_dir,
            project_hash=project_hash,
            project_ssh_path=project_ssh_path,
            label_hash=label_hash,
            du_hash=du_hash,
        )
        if isinstance(img_url_or_path, Path):
            return Image.open(img_url_or_path)
        return download_image(img_url_or_path, database_dir, cache=False)

    def _get_url_or_path(
        self,
        uri: Optional[str],
        database_dir: Path,
        project_hash: uuid.UUID,
        project_ssh_path: Optional[str],
        label_hash: uuid.UUID,
        du_hash: uuid.UUID,
    ) -> Union[str, Path]:
        if uri is not None:
            url_path = url_to_file_path(uri, database_dir)
            if url_path is not None:
                return url_path
            else:
                return uri
        elif project_ssh_path is not None:
            if project_hash in self.encord_projects:
                encord_project = self.encord_projects[project_hash]
            else:
                encord_project = get_encord_project(ssh_key_path=project_ssh_path, project_hash=str(project_hash))
                self.encord_projects[project_hash] = encord_project
            return encord_project.get_label_row(str(label_hash), get_signed_url=True,)["data_units"][
                str(du_hash)
            ]["data_link"]
        else:
            raise ValueError(f"Cannot resolve project url: {project_hash} / {label_hash} / {du_hash}")

    def _execute_stage_1_metrics_no_batching(
        self,
        prev_frame: Optional[BaseFrameInput],
        current_frame: BaseFrameInput,
        next_frame: Optional[BaseFrameInput],
        current_frame_gt: Optional[Dict[str, AnnotationMetadata]],
        du_hash: uuid.UUID,
        frame: int,
    ) -> None:
        # FIXME: add sanity checking that values are not mutated by metric runs
        image_results_deps: MetricDependencies = {"metric_metadata": {}}  # String comments (currently not used)
        annotation_results_deps: Dict[str, MetricDependencies] = {
            k: {
                "feature_hash": meta.feature_hash,
                "annotation_type": meta.annotation_type,
                "annotation_email": meta.annotation_email,
                "annotation_manual": meta.annotation_manual,
                "metric_confidence": meta.annotation_confidence,
                "metric_metadata": {},  # String comments (currently not used)
            }
            for k, meta in current_frame.annotations.items()
        }
        for evaluation in self.config.analysis:
            metric_result = evaluation.raw_calculate(
                prev_frame=prev_frame,
                frame=current_frame,
                next_frame=next_frame,
            )
            non_ephemeral = isinstance(evaluation, BaseAnalysis) or isinstance(evaluation, BaseEmbedding)
            if metric_result.image is not None:
                current_frame.image_deps[evaluation.ident] = metric_result.image
                if non_ephemeral:
                    image_results_deps[evaluation.ident] = metric_result.image
            for annotation_hash, annotation_value in metric_result.annotations.items():
                annotation_dep = current_frame.annotations_deps[annotation_hash]
                annotation_result_deps = annotation_results_deps[annotation_hash]
                if annotation_value is not None:
                    annotation_dep[evaluation.ident] = annotation_value
                    if non_ephemeral:
                        annotation_result_deps[evaluation.ident] = annotation_value

        if current_frame_gt is not None:
            # Calculate iou matches & iou thresholds
            min_iou_threshold_map: Dict[str, float] = {}
            gt_matches: Dict[str, Dict[str, MetricDependencies]] = {}
            for annotation_hash, annotation_meta in current_frame.annotations.items():
                # Calculate iou & best_match & best_match_feature_hash
                best_iou: float = 0.0
                best_iou_match: Optional[Tuple[str, str]] = None
                if annotation_meta.mask is None:
                    # classification
                    for gt_hash, gt_meta in current_frame_gt.items():
                        if gt_meta.feature_hash == annotation_meta.feature_hash:
                            best_iou = 1.0
                            best_iou_match = (gt_hash, gt_meta.feature_hash)
                else:
                    pass
                annotation_result = annotation_results_deps[annotation_hash]
                annotation_result["iou"] = best_iou
                if best_iou_match is None:
                    annotation_result["match_object_hash"] = None
                    annotation_result["match_feature_hash"] = None
                else:
                    annotation_result["match_object_hash"] = best_iou_match[0]
                    annotation_result["match_feature_hash"] = best_iou_match[1]
                    if best_iou_match[1] == annotation_meta.feature_hash:
                        gt_iou_threshold = min_iou_threshold_map.get(best_iou_match[0], -1.0)
                        min_iou_threshold_map[best_iou_match[0]] = max(gt_iou_threshold, best_iou)
                        gt_matches.setdefault(best_iou_match[0], {})[annotation_hash] = annotation_result
                    else:
                        annotation_result["match_duplicate_iou"] = 1.0  # Always false positive

            # Populate calculation of ranges
            for gt_match_set in gt_matches.values():
                gt_match_list = list(gt_match_set.items())
                if len(gt_match_list) == 1:
                    gt_match_list[0][1]["match_duplicate_iou"] = -1.0  # Always true positive
                else:
                    # Order by iou descending, calc the highest confidence value (TP) in the range of iou values
                    # considered for the TP associated with an IOU range.
                    def iou_compare_key(m_obj: Tuple[str, MetricDependencies]):
                        oh, m_deps = m_obj
                        return m_deps["iou"], m_deps["metric_confidence"], oh

                    gt_match_list.sort(key=iou_compare_key, reverse=True)

                    current_true_positive: MetricDependencies = gt_match_list[0][1]
                    current_true_positive["match_duplicate_iou"] = -1.0
                    current_true_positive_confidence: float = float(
                        cast(float, current_true_positive["metric_confidence"])
                    )
                    for mh_oh, mh_dict in gt_match_list[1:]:
                        model_confidence: float = float(cast(float, mh_dict["metric_confidence"]))
                        if model_confidence > current_true_positive_confidence:
                            mh_dict["match_duplicate_iou"] = -1.0
                            current_true_positive["match_duplicate_iou"] = mh_dict["iou"]
                            # Change of TP at iou threshold
                            current_true_positive = mh_dict
                            current_true_positive_confidence = model_confidence
                        else:
                            # Always FP
                            mh_dict["match_duplicate_iou"] = 1.0

            # Populate prediction false negatives table
            for gt_hash, gt_meta in current_frame_gt.items():
                self.prediction_false_negatives[(du_hash, frame, gt_hash)] = (
                    gt_meta.feature_hash,
                    min_iou_threshold_map.get(gt_hash, -1.0),
                )

        # Splat the results
        self.data_metrics[(du_hash, frame)] = image_results_deps
        for annotation_hash, annotation_deps in annotation_results_deps.items():
            self.annotation_metrics[(du_hash, frame, annotation_hash)] = annotation_deps

    def _get_frame_annotations(
        self,
        du_meta: ProjectDataUnitMetadata,
        image_width: int,
        image_height: int,
    ) -> Tuple[Dict[str, AnnotationMetadata], bool]:
        annotations = {}
        hash_collision = False
        for obj in du_meta.objects:
            obj_hash = str(obj["objectHash"])
            feature_hash = str(obj["featureHash"])
            annotation_type = annotation_type_from_str(str(obj["shape"]))
            if obj_hash in annotations:
                hash_collision = True
            else:
                try:
                    points = obj_to_points(annotation_type, obj, img_w=image_width, img_h=image_height)
                    mask = obj_to_mask(
                        self.config.device, annotation_type, img_w=image_width, img_h=image_height, points=points
                    )
                except Exception:
                    print(f"DEBUG Obj to Points & Mask for {annotation_type}: {obj}")
                    raise
                if mask is not None and mask.shape != (image_height, image_width):
                    raise ValueError(f"Wrong mask tensor shape: {mask.shape} != {(image_height, image_width)}")
                bounding_box = None
                if mask is not None:
                    mask = mask.to(self.config.device)
                    try:
                        bounding_box = masks_to_boxes(mask.unsqueeze(0))
                    except Exception:
                        print(
                            f"DEBUG BB-Gen Failure: {mask.shape}, {torch.min(mask)}, "
                            f"{torch.max(mask)}, {annotation_type} => {obj}"
                        )
                        raise
                annotations[obj_hash] = AnnotationMetadata(
                    feature_hash=feature_hash,
                    annotation_type=annotation_type,
                    points=None if points is None else points.to(self.config.device),
                    mask=mask,
                    annotation_email=str(obj["createdBy"]),
                    annotation_manual=bool(obj["manualAnnotation"]),
                    annotation_confidence=float(obj["confidence"]),
                    bounding_box=bounding_box,
                )
        for cls in du_meta.classifications:
            cls_hash = str(cls["classificationHash"])
            feature_hash = str(cls["featureHash"])
            if cls_hash in annotations:
                hash_collision = True
            else:
                annotations[cls_hash] = AnnotationMetadata(
                    feature_hash=feature_hash,
                    annotation_type=AnnotationType.CLASSIFICATION,
                    points=None,
                    mask=None,
                    annotation_email=str(cls["createdBy"]),
                    annotation_manual=bool(cls["manualAnnotation"]),
                    annotation_confidence=float(cls["confidence"]),
                    bounding_box=None,
                )
        return annotations, hash_collision

    def _get_frame_input(self, image: Image.Image, du_meta: ProjectDataUnitMetadata) -> BaseFrameInput:
        annotations, hash_collision = self._get_frame_annotations(
            du_meta=du_meta, image_width=image.width, image_height=image.height
        )
        annotations_deps: Dict[str, MetricDependencies] = {
            annotation_hash: {} for annotation_hash in annotations.keys()
        }

        image_tensor = pillow_to_tensor(image).to(self.config.device)
        if image_tensor.shape != (3, image.height, image.width):
            raise ValueError(f"Wrong image tensor shape: {image_tensor.shape} != {(3, image.height, image.width)}")
        return BaseFrameInput(
            image=image_tensor,
            image_deps={},
            annotations=annotations,
            annotations_deps=annotations_deps,
            hash_collision=hash_collision,
        )

    def _stage_2(self):
        # Stage 2: Support for complex value based dependencies.
        # Note: in-efficient, but needed for first version of this.
        # Stage 2 evaluation will always be very closely bound to the executor
        # and the associated storage and indexing strategy.
        for metric in tqdm(self.config.derived_embeddings, desc="Metric Computation Stage 2: Building indices"):
            metrics_tqdm_desc = f"                          : Generating indices for {metric.ident}"
            metrics_tqdm = tqdm([self.data_metrics, self.annotation_metrics], desc=metrics_tqdm_desc)
            for metrics in metrics_tqdm:
                metrics_tqdm.set_description(metrics_tqdm_desc)
                if len(metrics) == 0:
                    continue
                if isinstance(metric, NearestImageEmbeddingQuery):
                    # Embedding query lookup
                    max_neighbours = int(metric.max_neighbours)
                    if max_neighbours > 50:
                        raise ValueError(f"Too many embedding nn-query results: {metric.ident} => {max_neighbours}")
                    data_keys = list(metrics.keys())
                    metrics_tqdm.set_description(f"{metrics_tqdm_desc}  [Loading data]")
                    data_list = [metrics[k][metric.embedding_source].cpu().numpy() for k in data_keys]
                    metrics_tqdm.set_description(f"{metrics_tqdm_desc}  [Preparing data]")
                    data = np.stack(data_list)
                    # FIXME: tune threshold where using pynndescent is actually faster.
                    if len(data_list) <= 4096:
                        # Exhaustive search is faster & more correct for very small datasets.
                        metrics_tqdm.set_description(f"{metrics_tqdm_desc}  [Exhaustive search]")
                        data_indices_list = []
                        data_distances_list = []
                        for data_key, data_embedding in zip(data_keys, data_list):
                            data_raw_offsets = data - data_embedding
                            data_raw_similarities = np.linalg.norm(data_raw_offsets, axis=1)
                            data_raw_index = np.argsort(data_raw_similarities)[:max_neighbours]
                            data_indices_list.append(data_raw_index)
                            data_distances_list.append(data_raw_similarities[data_raw_index])
                        data_indices = np.stack(data_indices_list)
                        data_distances = np.stack(data_distances_list)
                    else:
                        metrics_tqdm.set_description(f"{metrics_tqdm_desc}  [Creating index]")
                        nn_descent = NNDescent(
                            data=data,
                            parallel_batch_queries=True,
                            n_jobs=-1,
                        )
                        metrics_tqdm.set_description(f"{metrics_tqdm_desc}  [Building index]")
                        nn_descent.prepare()
                        metrics_tqdm.set_description(f"{metrics_tqdm_desc}  [Querying index]")
                        data_indices, data_distances = nn_descent.query(
                            query_data=data,
                            k=max_neighbours,
                        )
                    metrics_tqdm.set_description(f"{metrics_tqdm_desc}  [Finishing]")
                    for i, k in enumerate(data_keys):
                        indices = data_indices[i]
                        distances = data_distances[i]

                        # Post-process query result:
                        if len(indices) > 0 and indices[0] == i:
                            indices = indices[1:]
                            distances = distances[1:]
                        while len(distances) > 0 and math.isinf(distances[-1]):
                            indices = distances[:-1]
                            distances = distances[:-1]

                        # Transform value to result.
                        nearest_neighbor_result = NearestNeighbors(
                            similarities=distances.tolist(),
                            metric_deps=[dict(metrics[data_keys[int(j)]]) for j in indices.tolist()],
                            metric_keys=[data_keys[int(j)] for j in indices.tolist()],
                        )
                        metrics[k][metric.ident] = nearest_neighbor_result
                elif isinstance(metric, RandomSamplingQuery):
                    # Random sampling
                    limit = metric.limit
                    if limit > 100:
                        raise ValueError("Too large random sampling amount")
                    data_keys = list(metrics.keys())
                    for i, dk in enumerate(data_keys):
                        other_data_keys = data_keys[:i] + data_keys[i + 1 :]
                        chosen_keys = random.choices(other_data_keys, k=max(limit, len(other_data_keys)))
                        metrics[dk][metric.ident] = RandomSampling(
                            metric_deps=[dict(metrics[k]) for k in chosen_keys.tolist()], metric_keys=chosen_keys
                        )
                else:
                    raise ValueError(
                        f"Unsupported index stage metric[{getattr(metric, 'ident', None)}]: {metric.__name__}"
                    )

    def _stage_3(self) -> None:
        # Stage 3: Derived metrics from stage 1 & stage 2 metrics
        # dependencies must be pure.
        for data_metric in tqdm(self.data_metrics.values(), desc="Metric Computation Stage 3: Derived - Data"):
            for metric in self.config.derived_metrics:
                derived_metric = metric.calculate(dict(data_metric), None)
                if derived_metric is not None:
                    data_metric[metric.ident] = derived_metric
        for annotation_metric in tqdm(
            self.annotation_metrics.values(), desc="Metric Computation Stage 3: Derived - Annotation"
        ):
            for metric in self.config.derived_metrics:
                derived_metric = metric.calculate(dict(annotation_metric), annotation_metric["annotation_type"])
                if derived_metric is not None:
                    annotation_metric[metric.ident] = derived_metric

    def _validate_pack_deps(
        self,
        deps: MetricDependencies,
        ty_analytics: Type[Union[ProjectPredictionAnalytics, ProjectAnnotationAnalytics, ProjectDataAnalytics]],
        ty_extra: Type[
            Union[ProjectPredictionAnalyticsExtra, ProjectAnnotationAnalyticsExtra, ProjectDataAnalyticsExtra]
        ],
    ) -> Tuple[Set[str], Set[str]]:
        metric_lookup = {
            m.ident: m.metric_type
            for m in (self.config.analysis + self.config.derived_metrics)
            if isinstance(m, BaseMetric)
        }
        # Hardcoded special metric
        metric_lookup["metric_confidence"] = MetricType.NORMAL
        analysis_values = {k for k in deps if k in ty_analytics.__fields__ and k.startswith("metric_")}
        for k in analysis_values:
            m = metric_lookup[k]
            v_raw = deps[k]
            v = v_raw.item() if isinstance(v_raw, Tensor) else v_raw
            if not isinstance(v, float) and not isinstance(v, int):
                raise ValueError(f"Unsupported metric type: {type(v)}")
            if m == MetricType.NORMAL:
                if v < 0.0 or v > 1.0:
                    raise ValueError(f"Normal metric out of bounds: {k}[{m}] = {v}")
            elif m == MetricType.UINT or m == MetricType.UFLOAT:
                if v < 0:
                    raise ValueError(f"Unsigned metric out of bounds: {k}[{m}] = {v}")
            else:
                raise ValueError(f"Unsupported metric type: {m}")

        # Extra values & missing values
        extra_values = set(deps.keys()) - analysis_values
        unassigned_metric_values = {
            k
            for k in ty_analytics.__fields__
            if (k.startswith("metric_") or k.startswith("embedding_"))
            and k not in analysis_values
            and not k.startswith("metric_custom")
        }
        log = logging.getLogger("Metric Executor")
        if len(unassigned_metric_values) > 0:
            log.debug(f"WARNING: unassigned analytics metrics({ty_analytics.__name__}) = {unassigned_metric_values}")
        for extra_value in extra_values:
            if (
                not (
                    extra_value.startswith("metric_")
                    or extra_value.startswith("embedding_")
                    or extra_value.startswith("derived_")
                )
                or extra_value not in ty_extra.__fields__
            ):
                raise ValueError(f"Unknown field name: '{extra_value}' for {ty_extra.__name__}")
        unassigned_extra_values = {
            k
            for k in ty_extra.__fields__
            if (k.startswith("metric_") or k.startswith("embedding_") or k.startswith("derived_"))
            and k not in extra_values
        }
        if len(unassigned_extra_values) > 0:
            log.debug(f"WARNING: unassigned analytics extra value({ty_extra.__name__}) = {unassigned_extra_values}")
        return analysis_values, extra_values

    @classmethod
    def _pack_metric(
        cls, v: Union[Tensor, float, int, str, None, NearestNeighbors, RandomSampling, Dict[str, str], AnnotationType]
    ) -> Union[bytes, float, int, str, None]:
        if isinstance(v, Tensor):
            return v.item()
        if (
            isinstance(v, NearestNeighbors)
            or isinstance(v, RandomSampling)
            or isinstance(v, dict)
            or isinstance(v, AnnotationType)
        ):
            raise ValueError(f"Unsupported metric type: {type(v)}")
        return v

    @classmethod
    def _pack_extra(
        cls,
        k: str,
        v: Union[Tensor, float, int, str, None, NearestNeighbors, RandomSampling, Dict[str, str], AnnotationType],
    ) -> Union[bytes, float, int, str, None, dict]:
        if isinstance(v, AnnotationType):
            raise ValueError(f"Annotation type is not a valid extra property: {k}")
        elif isinstance(v, Tensor):
            if k.startswith("embedding_"):
                if len(v.shape) != 1:
                    raise ValueError(f"Invalid embedding shape: {v.shape}")
                embedding_len = None
                if k == "embedding_hu":
                    embedding_len = 7
                elif k == "embedding_clip":
                    embedding_len = 512
                if v.shape[0] != embedding_len:
                    raise ValueError(f"Embedding has wrong length: {v.shape} != {embedding_len or 'Unknown'}")
                return v.cpu().numpy().tobytes()
            return v.item()
        elif isinstance(v, NearestNeighbors):
            return {
                "similarities": v.similarities,
                "keys": [[str(kv) if isinstance(kv, uuid.UUID) else kv for kv in k] for k in v.metric_keys],
            }
        elif isinstance(v, RandomSampling):
            return {"keys": [[str(kv) if isinstance(kv, uuid.UUID) else kv for kv in k] for k in v.metric_keys]}
        return v

    def _pack_output(
        self,
        project_hash: uuid.UUID,
    ) -> Tuple[
        List[ProjectDataAnalytics],
        List[ProjectDataAnalyticsExtra],
        List[ProjectAnnotationAnalytics],
        List[ProjectAnnotationAnalyticsExtra],
    ]:
        data_analysis = []
        data_analysis_extra = []
        annotation_analysis = []
        annotation_analysis_extra = []

        for (du_hash, frame), data_deps in self.data_metrics.items():
            analysis_values, extra_values = self._validate_pack_deps(
                data_deps,
                ProjectDataAnalytics,
                ProjectDataAnalyticsExtra,
            )
            data_analysis.append(
                ProjectDataAnalytics(
                    project_hash=project_hash,
                    du_hash=du_hash,
                    frame=frame,
                    **{k: self._pack_metric(v) for k, v in data_deps.items() if k in analysis_values},
                )
            )
            data_analysis_extra.append(
                ProjectDataAnalyticsExtra(
                    project_hash=project_hash,
                    du_hash=du_hash,
                    frame=frame,
                    **{
                        k: self._pack_extra(k, v)
                        for k, v in data_deps.items()
                        if k in extra_values and not k.startswith("derived_")
                    },
                )
            )
        for (du_hash, frame, annotation_hash), annotation_deps in self.annotation_metrics.items():
            feature_hash = annotation_deps.pop("feature_hash")
            annotation_type = annotation_deps.pop("annotation_type")
            annotation_email = annotation_deps.pop("annotation_email")
            annotation_manual = annotation_deps.pop("annotation_manual")
            analysis_values, extra_values = self._validate_pack_deps(
                annotation_deps,
                ProjectAnnotationAnalytics,
                ProjectAnnotationAnalyticsExtra,
            )
            annotation_analysis.append(
                ProjectAnnotationAnalytics(
                    project_hash=project_hash,
                    du_hash=du_hash,
                    frame=frame,
                    object_hash=annotation_hash,
                    feature_hash=feature_hash,
                    annotation_type=annotation_type,
                    annotation_email=annotation_email,
                    annotation_manual=annotation_manual,
                    **{k: self._pack_metric(v) for k, v in annotation_deps.items() if k in analysis_values},
                )
            )
            annotation_analysis_extra.append(
                ProjectAnnotationAnalyticsExtra(
                    project_hash=project_hash,
                    du_hash=du_hash,
                    frame=frame,
                    object_hash=annotation_hash,
                    **{k: self._pack_extra(k, v) for k, v in annotation_deps.items() if k in extra_values},
                )
            )

        return data_analysis, data_analysis_extra, annotation_analysis, annotation_analysis_extra

    def _pack_prediction(
        self, project_hash: uuid.UUID, prediction_hash: uuid.UUID
    ) -> Tuple[
        List[ProjectPredictionAnalytics],
        List[ProjectPredictionAnalyticsExtra],
        List[ProjectPredictionAnalyticsFalseNegatives],
    ]:
        prediction_analysis = []
        prediction_analysis_extra = []
        prediction_false_negatives = []

        for (du_hash, frame, annotation_hash), annotation_deps in self.annotation_metrics.items():
            feature_hash = annotation_deps.pop("feature_hash")
            annotation_type = annotation_deps.pop("annotation_type")
            annotation_email = annotation_deps.pop("annotation_email")
            annotation_manual = annotation_deps.pop("annotation_manual")
            analysis_values, extra_values = self._validate_pack_deps(
                annotation_deps,
                ProjectPredictionAnalytics,
                ProjectPredictionAnalyticsExtra,
            )
            prediction_analysis.append(
                ProjectPredictionAnalytics(
                    prediction_hash=prediction_hash,
                    project_hash=project_hash,
                    du_hash=du_hash,
                    frame=frame,
                    object_hash=annotation_hash,
                    feature_hash=feature_hash,
                    annotation_type=annotation_type,
                    annotation_email=annotation_email,
                    annotation_manual=annotation_manual,
                    **{k: self._pack_metric(v) for k, v in annotation_deps.items() if k in analysis_values},
                )
            )
            prediction_analysis_extra.append(
                ProjectPredictionAnalyticsExtra(
                    prediction_hash=prediction_hash,
                    du_hash=du_hash,
                    frame=frame,
                    object_hash=annotation_hash,
                    **{k: self._pack_extra(k, v) for k, v in annotation_deps.items() if k in extra_values},
                )
            )

        for (du_hash, frame, annotation_hash), (feature_hash, iou_threshold) in self.prediction_false_negatives.items():
            prediction_false_negatives.append(
                ProjectPredictionAnalyticsFalseNegatives(
                    prediction_hash=prediction_hash,
                    du_hash=du_hash,
                    frame=frame,
                    annotation_hash=annotation_hash,
                    iou_threshold=iou_threshold,
                    feature_hash=feature_hash,
                )
            )

        return prediction_analysis, prediction_analysis_extra, prediction_false_negatives
