import logging
import math
import random
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union, cast

import av
import encord
import numpy as np
import torch
import torchvision
from encord.http.constants import RequestsSettings
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
    ProjectAnnotationAnalytics,
    ProjectAnnotationAnalyticsDerived,
    ProjectAnnotationAnalyticsExtra,
    ProjectCollaborator,
    ProjectDataAnalytics,
    ProjectDataAnalyticsDerived,
    ProjectDataAnalyticsExtra,
    ProjectDataMetadata,
    ProjectDataUnitMetadata,
    ProjectPredictionAnalytics,
    ProjectPredictionAnalyticsDerived,
    ProjectPredictionAnalyticsExtra,
    ProjectPredictionAnalyticsFalseNegatives,
)

from ..db.enums import AnnotationType, DataType
from ..db.local_data import db_uri_to_local_file_path, open_database_uri_image
from ..db.metrics import MetricType
from .base import BaseAnalysis, BaseFrameBatchInput, BaseFrameInput, BaseMetric
from .embedding import (
    BaseEmbedding,
    EmbeddingDistanceMetric,
    NearestImageEmbeddingQuery,
)
from .metric import RandomSamplingQuery

TTypeAnnotationAnalytics = TypeVar("TTypeAnnotationAnalytics", ProjectAnnotationAnalytics, ProjectPredictionAnalytics)
TTypeAnnotationAnalyticsExtra = TypeVar(
    "TTypeAnnotationAnalyticsExtra", ProjectAnnotationAnalyticsExtra, ProjectPredictionAnalyticsExtra
)


ExecutorLogger: logging.Logger = logging.getLogger("executor")


class Executor:
    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config


class SimpleExecutor(Executor):
    def __init__(self, config: AnalysisConfig, pool_size: int = 8) -> None:
        super().__init__(config=config)
        self.data_metrics: Dict[Tuple[uuid.UUID, int], MetricDependencies] = {}
        self.annotation_metrics: Dict[Tuple[uuid.UUID, int, str], MetricDependencies] = {}
        self.prediction_false_negatives: Dict[Tuple[uuid.UUID, int, str], Tuple[str, float]] = {}
        self.encord_projects: Dict[uuid.UUID, encord.Project] = {}
        self.encord_client: Optional[encord.EncordUserClient] = None
        self.encord_project_urls: Dict[Tuple[uuid.UUID, uuid.UUID], str] = {}
        self.pool_size = pool_size
        print(f"Simple executor: using {self.config.device.type} device")

    def execute_from_db(
        self,
        data_meta: List[ProjectDataMetadata],
        du_meta: List[ProjectDataUnitMetadata],
        collaborators: List[ProjectCollaborator],
        database_dir: Path,
        project_hash: uuid.UUID,
        project_ssh_key: Optional[str],
    ) -> Tuple[
        List[ProjectDataAnalytics],
        List[ProjectDataAnalyticsExtra],
        List[ProjectDataAnalyticsDerived],
        List[ProjectAnnotationAnalytics],
        List[ProjectAnnotationAnalyticsExtra],
        List[ProjectAnnotationAnalyticsDerived],
        List[ProjectCollaborator],
    ]:
        du_dict: Dict[uuid.UUID, List[ProjectDataUnitMetadata]] = {}
        for du in du_meta:
            du_dict.setdefault(du.data_hash, []).append(du)
        values = [(data, du_dict[data.data_hash]) for data in data_meta]
        self.execute(
            values,
            database_dir=database_dir,
            project_hash=project_hash,
            project_ssh_key=project_ssh_key,
        )
        return self._pack_output(project_hash=project_hash, collaborators=collaborators)

    def execute_from_db_subset(
        self,
        database_dir: Path,
        project_hash: uuid.UUID,
        precalculated_data: List[ProjectDataAnalytics],
        precalculated_data_extra: List[ProjectDataAnalyticsExtra],
        precalculated_annotation: List[ProjectAnnotationAnalytics],
        precalculated_annotation_extra: List[ProjectAnnotationAnalyticsExtra],
        precalculated_collaborators: List[ProjectCollaborator],
    ) -> Tuple[
        List[ProjectDataAnalytics],
        List[ProjectDataAnalyticsExtra],
        List[ProjectDataAnalyticsDerived],
        List[ProjectAnnotationAnalytics],
        List[ProjectAnnotationAnalyticsExtra],
        List[ProjectAnnotationAnalyticsDerived],
        List[ProjectCollaborator],
    ]:
        self._stage_1_load_from_existing_data(
            precalculated_data=precalculated_data,
            precalculated_data_extra=precalculated_data_extra,
            precalculated_annotation=precalculated_annotation,
            precalculated_annotation_extra=precalculated_annotation_extra,
            precalculated_collaborators=precalculated_collaborators,
        )
        self.execute(
            [],
            database_dir=database_dir,
            project_hash=project_hash,
            project_ssh_key=None,
        )
        return self._pack_output(project_hash=project_hash, collaborators=[])

    def execute_prediction_from_db(
        self,
        data_meta: List[ProjectDataMetadata],
        ground_truth_annotation_meta: List[ProjectDataUnitMetadata],
        predicted_annotation_meta: List[ProjectDataUnitMetadata],
        collaborators: List[ProjectCollaborator],
        database_dir: Path,
        project_hash: uuid.UUID,
        prediction_hash: uuid.UUID,
        project_ssh_key: Optional[str],
    ) -> Tuple[
        List[ProjectPredictionAnalytics],
        List[ProjectPredictionAnalyticsExtra],
        List[ProjectPredictionAnalyticsDerived],
        List[ProjectPredictionAnalyticsFalseNegatives],
        List[ProjectCollaborator],
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
            project_ssh_key=project_ssh_key,
            gt_lookup=gt_lookup,
        )
        return self._pack_prediction(
            project_hash=project_hash, prediction_hash=prediction_hash, collaborators=collaborators
        )

    def execute(
        self,
        values: List[Tuple[ProjectDataMetadata, List[ProjectDataUnitMetadata]]],
        database_dir: Path,
        project_hash: uuid.UUID,
        project_ssh_key: Optional[str],
        gt_lookup: Optional[Dict[Tuple[uuid.UUID, int], ProjectDataUnitMetadata]] = None,
        batch: Optional[int] = None,  # Disable the batch execution path, use legacy code-path instead.
        skip_stage_1: bool = False,
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

        # Pre-fetch encord-urls
        # if project_ssh_path is not None:
        # self._batch_populate_lookup_cache(
        #    project_hash=project_hash,
        #    project_ssh_path=project_ssh_path,
        #    values=values,
        # )

        # Video best thing
        with torch.no_grad():
            # Stage 1
            if batch is None and not skip_stage_1:
                self._stage_1_video_no_batching(
                    video_values,
                    database_dir=database_dir,
                    project_hash=project_hash,
                    project_ssh_key=project_ssh_key,
                    gt_lookup=gt_lookup,
                )
                self._stage_1_image_no_batching(
                    image_values,
                    database_dir=database_dir,
                    project_hash=project_hash,
                    project_ssh_key=project_ssh_key,
                    gt_lookup=gt_lookup,
                )
            elif not skip_stage_1:
                self._stage_1_video_batching(
                    video_values,
                    database_dir=database_dir,
                    project_hash=project_hash,
                    project_ssh_key=project_ssh_key,
                    gt_lookup=gt_lookup,
                )
                self._stage_1_image_batching(
                    image_values,
                    database_dir=database_dir,
                    project_hash=project_hash,
                    project_ssh_key=project_ssh_key,
                    batching=batch or 1,
                    gt_lookup=gt_lookup,
                )

            # Post-processing stages
            self._stage_2()
            self._stage_3()

    def _batch_populate_lookup_cache(
        self,
        project_hash: uuid.UUID,
        project_ssh_key: str,
        values: List[Tuple[ProjectDataMetadata, List[ProjectDataUnitMetadata]]],
    ) -> None:
        if project_hash in self.encord_projects:
            encord_project = self.encord_projects[project_hash]
        else:
            if self.encord_client is None:
                self.encord_client = encord.EncordUserClient.create_with_ssh_private_key(
                    project_ssh_key,
                    requests_settings=RequestsSettings(max_retries=5),
                )
            encord_project = self.encord_client.get_project(str(project_hash))
            self.encord_projects[project_hash] = encord_project
        data_hashes: List[str] = []
        for data_hash_meta, du_meta_list in values:
            if data_hash_meta.data_type == "video":
                continue  # Don't gain much from pre-fetch
            lookup = False
            for du_meta in du_meta_list:
                if du_meta.data_uri is None:
                    lookup = True
                    break
            if lookup:
                data_hashes.append(str(data_hash_meta.data_hash))

        if len(data_hashes) == 0:
            return

        with ThreadPoolExecutor(max_workers=self.pool_size) as pool:
            results = []
            for data_hash in data_hashes:
                results.append(pool.submit(encord_project.get_data, data_hash=data_hash, get_signed_url=True))
            for result in tqdm(results, desc="Prefetching data links"):
                _, images = result.result(timeout=120.0)
                for image in images or []:
                    du_hash = uuid.UUID(image["data_hash"])
                    signed_url = str(image["file_link"])
                    self.encord_project_urls[(project_hash, du_hash)] = signed_url

    def _stage_1_video_batching(
        self,
        videos: List[Tuple[ProjectDataMetadata, List[ProjectDataUnitMetadata]]],
        database_dir: Path,
        project_hash: uuid.UUID,
        project_ssh_key: Optional[str],
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
                project_ssh_key=project_ssh_key,
                label_hash=data_metadata.label_hash,
                data_hash=data_metadata.data_hash,
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
        project_ssh_key: Optional[str],
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
                project_ssh_key=project_ssh_key,
                label_hash=data_metadata.label_hash,
                data_hash=data_metadata.data_hash,
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
                    if iter_frame_num + 1 != frame_num:
                        logging.error(
                            f"AV video frame out of sequence (overriding): {iter_frame_num} (iter) -> {frame_num} (AV)"
                        )
                        frame_num = iter_frame_num + 1
                    iter_frame_num = frame_num

                    # Load video
                    frame_du_meta = du_meta_list[frame_num]
                    if frame_du_meta.frame != frame_num:
                        raise ValueError(f"Frame metadata mismatch: {frame_du_meta.frame} != {frame_num}")
                    if frame.is_corrupt:
                        raise ValueError(f"Corrupt video frame: {frame_num}")
                    frame_image = frame.to_image().convert("RGB")
                    frame_input = self._get_frame_input(
                        image=frame_image, du_meta=frame_du_meta, data_meta=data_metadata
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
        project_ssh_key: Optional[str],
        gt_lookup: Optional[Dict[Tuple[uuid.UUID, int], ProjectDataUnitMetadata]],
    ):
        def _job(data_meta_job: ProjectDataMetadata, du_meta_job: ProjectDataUnitMetadata):
            with torch.no_grad():
                gt = None if gt_lookup is None else gt_lookup[du_meta_job.du_hash, du_meta_job.frame]
                image = self._open_image(
                    uri=du_meta_job.data_uri,
                    database_dir=database_dir,
                    project_hash=project_hash,
                    project_ssh_key=project_ssh_key,
                    label_hash=data_meta_job.label_hash,
                    data_hash=data_meta_job.data_hash,
                    du_hash=du_meta_job.du_hash,
                )
                frame_input = self._get_frame_input(
                    image=image,
                    du_meta=du_meta_job,
                    data_meta=data_meta_job,
                )
                if gt is None:
                    gt_input = None
                else:
                    gt_input, _ignore = self._get_frame_annotations(
                        gt, image_width=image.width, image_height=image.height
                    )
                self._execute_stage_1_metrics_no_batching(
                    prev_frame=None,
                    current_frame=frame_input,
                    next_frame=None,
                    current_frame_gt=gt_input,
                    du_hash=du_meta_job.du_hash,
                    frame=du_meta_job.frame,
                )

        # Thread-pool execute by chunks when running on cpu mode
        if self.config.device.type == "cpu":
            with ThreadPoolExecutor(max_workers=self.pool_size) as pool:
                tqdm_desc = tqdm(total=len(images), desc="Metric Computation Stage 1: Images")
                for i in range(0, len(images), 100):
                    images_group = images[i : i + 100]
                    results = []
                    for data_metadata, du_meta in images_group:
                        results.append(pool.submit(_job, data_meta_job=data_metadata, du_meta_job=du_meta))
                    for result in results:
                        # Wait for completion.
                        result.result(timeout=3000.0)
                        tqdm_desc.update(1)
        else:
            for data_metadata, du_meta in tqdm(images, desc="Metric Computation Stage 1: Images"):
                _job(data_meta_job=data_metadata, du_meta_job=du_meta)

    def _stage_1_image_batching(
        self,
        values: List[Tuple[ProjectDataMetadata, ProjectDataUnitMetadata]],
        database_dir: Path,
        project_hash: uuid.UUID,
        project_ssh_key: Optional[str],
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
                            project_ssh_key=project_ssh_key,
                            label_hash=meta.label_hash,
                            data_hash=du_meta.data_hash,
                            du_hash=du_meta.du_hash,
                        ),
                        du_meta=du_meta,
                        data_meta=meta,
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

    def _stage_1_load_from_existing_data(
        self,
        precalculated_data: List[ProjectDataAnalytics],
        precalculated_data_extra: List[ProjectDataAnalyticsExtra],
        precalculated_annotation: List[ProjectAnnotationAnalytics],
        precalculated_annotation_extra: List[ProjectAnnotationAnalyticsExtra],
        precalculated_collaborators: List[ProjectCollaborator],
    ) -> None:
        user_id_to_email: Dict[int, str] = {
            collaborator.user_id: collaborator.user_email for collaborator in precalculated_collaborators
        }
        data_extra_lookup: Dict[Tuple[uuid.UUID, int], ProjectDataAnalyticsExtra] = {
            (data_extra.du_hash, data_extra.frame): data_extra for data_extra in precalculated_data_extra
        }
        annotation_extra_lookup: Dict[Tuple[uuid.UUID, int, str], ProjectAnnotationAnalyticsExtra] = {
            (annotation_extra.du_hash, annotation_extra.frame, annotation_extra.annotation_hash): annotation_extra
            for annotation_extra in precalculated_annotation_extra
        }

        def _load_embedding(value: Any) -> Any:
            if isinstance(value, bytes):
                return torch.from_numpy(np.frombuffer(value, dtype=np.float64))
            else:
                return value

        for data in precalculated_data:
            data_extra = data_extra_lookup[data.du_hash, data.frame]
            data_metrics: MetricDependencies = {
                "data_type": data.data_type,
                "metric_metadata": data_extra.metric_metadata,
            }
            for analysis in self.config.analysis:
                if analysis.ident in data.__fields__:
                    data_metrics[analysis.ident] = getattr(data, analysis.ident)
                elif analysis.ident in data_extra.__fields__:
                    data_metrics[analysis.ident] = _load_embedding(getattr(data_extra, analysis.ident))
            self.data_metrics[data.du_hash, data.frame] = data_metrics

        for annotation in precalculated_annotation:
            annotation_extra = annotation_extra_lookup[annotation.du_hash, annotation.frame, annotation.annotation_hash]
            annotation_metrics: MetricDependencies = {
                "feature_hash": annotation.feature_hash,
                "annotation_type": annotation.annotation_type,
                "annotation_email": user_id_to_email[annotation.annotation_user_id]
                if annotation.annotation_user_id is not None
                else None,
                "annotation_manual": annotation.annotation_manual,
                "annotation_invalid": annotation.annotation_invalid,
                "metric_confidence": annotation.metric_confidence,
                "metric_metadata": annotation_extra.metric_metadata,
            }
            for analysis in self.config.analysis:
                if analysis.ident in annotation.__fields__:
                    annotation_metrics[analysis.ident] = getattr(annotation, analysis.ident)
                elif analysis.ident in annotation_extra.__fields__:
                    annotation_metrics[analysis.ident] = _load_embedding(getattr(annotation_extra, analysis.ident))
            self.annotation_metrics[
                annotation.du_hash, annotation.frame, annotation.annotation_hash
            ] = annotation_metrics

    def _open_image(
        self,
        uri: Optional[str],
        database_dir: Path,
        project_hash: uuid.UUID,
        project_ssh_key: Optional[str],
        label_hash: uuid.UUID,
        data_hash: uuid.UUID,
        du_hash: uuid.UUID,
    ):
        img_url_or_path = self._get_url_or_path(
            uri=uri,
            database_dir=database_dir,
            project_hash=project_hash,
            project_ssh_key=project_ssh_key,
            label_hash=label_hash,
            data_hash=data_hash,
            du_hash=du_hash,
        )
        if isinstance(img_url_or_path, Path):
            return Image.open(img_url_or_path)
        return open_database_uri_image(img_url_or_path, database_dir)

    def _get_url_or_path(
        self,
        uri: Optional[str],
        database_dir: Path,
        project_hash: uuid.UUID,
        project_ssh_key: Optional[str],
        label_hash: uuid.UUID,
        data_hash: uuid.UUID,
        du_hash: uuid.UUID,
    ) -> Union[str, Path]:
        if uri is not None:
            url_path = db_uri_to_local_file_path(uri, database_dir)
            if url_path is not None:
                return url_path
            else:
                return uri
        elif project_ssh_key is not None:
            if (project_hash, du_hash) in self.encord_project_urls:
                return self.encord_project_urls[(project_hash, du_hash)]
            if project_hash in self.encord_projects:
                encord_project = self.encord_projects[project_hash]
            else:
                if self.encord_client is None:
                    self.encord_client = encord.EncordUserClient.create_with_ssh_private_key(
                        project_ssh_key,
                        requests_settings=RequestsSettings(max_retries=5),
                    )
                encord_project = self.encord_client.get_project(str(project_hash))
                self.encord_projects[project_hash] = encord_project
            video, images = encord_project.get_data(data_hash=str(data_hash), get_signed_url=True)
            for image in images or []:
                du_hash = uuid.UUID(image["data_hash"])
                signed_url = str(image["file_link"])
                self.encord_project_urls[(project_hash, du_hash)] = signed_url
            if video is not None:
                du_hash = uuid.UUID(video["data_hash"]) if "data_hash" in video else data_hash
                signed_url = str(video["file_link"])
                self.encord_project_urls[(project_hash, du_hash)] = signed_url
            return self.encord_project_urls[(project_hash, du_hash)]
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
        image_results_deps: MetricDependencies = {
            "data_type": current_frame.data_type,
            "metric_metadata": {},  # String comments (currently not used)
        }
        annotation_results_deps: Dict[str, MetricDependencies] = {
            k: {
                "feature_hash": meta.feature_hash,
                "annotation_type": meta.annotation_type,
                "annotation_email": meta.annotation_email,
                "annotation_manual": meta.annotation_manual,
                "annotation_invalid": k in current_frame.hash_collision,
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
            if metric_result.image_comment is not None and non_ephemeral:
                image_results_deps["metric_metadata"][evaluation.ident] = metric_result.image_comment  # type: ignore
            for annotation_hash, annotation_value in metric_result.annotations.items():
                annotation_dep = current_frame.annotations_deps[annotation_hash]
                annotation_result_deps = annotation_results_deps[annotation_hash]
                if annotation_value is not None:
                    annotation_dep[evaluation.ident] = annotation_value
                    if non_ephemeral:
                        annotation_result_deps[evaluation.ident] = annotation_value
            for annotation_hash, annotation_comment in metric_result.annotation_comments.items():
                annotation_result_deps = annotation_results_deps[annotation_hash]
                if non_ephemeral:
                    annotation_result_deps["metric_metadata"][evaluation.ident] = annotation_comment  # type: ignore

        if current_frame_gt is not None:
            # Calculate iou matches & iou thresholds
            min_iou_threshold_map: Dict[str, float] = {}
            gt_matches: Dict[str, Dict[str, MetricDependencies]] = {}
            for annotation_hash, annotation_meta in current_frame.annotations.items():
                # Calculate iou & best_match & best_match_feature_hash
                best_iou: float = 0.0
                best_iou_match: Optional[Tuple[str, str]] = None
                if annotation_meta.mask is None or annotation_meta.bounding_box is None:
                    # classification
                    for gt_hash, gt_meta in current_frame_gt.items():
                        if gt_meta.feature_hash == annotation_meta.feature_hash:
                            best_iou = 1.0
                            best_iou_match = (gt_hash, gt_meta.feature_hash)
                else:
                    # object iou match
                    best_iou = 0.0
                    best_iou_match = None
                    b1x1, b1y1, b1x2, b1y2 = annotation_meta.bounding_box
                    for gt_hash, gt_meta in current_frame_gt.items():
                        if gt_meta.mask is None or gt_meta.bounding_box is None:
                            continue
                        b2x1, b2y1, b2x2, b2y2 = gt_meta.bounding_box
                        # Quick bb-bb intersection test to avoid expensive calculation when not needed.
                        if b2x1 > b1x2 or b2x2 < b1x1 or b2y1 > b1y2 or b2y2 < b1y1:
                            intersect = 0.0  # Quick bb-collision exclude
                        else:
                            intersect = float((annotation_meta.mask & gt_meta.mask).sum().item())
                        if intersect == 0.0:
                            iou = 0.0
                        else:
                            iou = intersect / float((annotation_meta.mask | gt_meta.mask).sum().item())
                        if iou > best_iou:
                            best_iou = iou
                            best_iou_match = (gt_hash, gt_meta.feature_hash)
                annotation_result = annotation_results_deps[annotation_hash]
                annotation_result["iou"] = best_iou
                if best_iou_match is None:
                    annotation_result["match_annotation_hash"] = None
                    annotation_result["match_feature_hash"] = None
                    annotation_result["match_duplicate_iou"] = 1.0  # Always false positive
                else:
                    annotation_result["match_annotation_hash"] = best_iou_match[0]
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
    ) -> Tuple[Dict[str, AnnotationMetadata], Set[str]]:
        annotations = {}
        hash_collision = set()
        for obj in du_meta.objects:
            obj_hash = str(obj["objectHash"])
            feature_hash = str(obj["featureHash"])
            annotation_type = AnnotationType(str(obj["shape"]))  # type: ignore
            if obj_hash in annotations:
                hash_collision.add(obj_hash)
            else:
                try:
                    points = obj_to_points(annotation_type, obj, img_w=image_width, img_h=image_height)
                    mask_device = torch.device("cpu") if self.config.device.type == "mps" else self.config.device
                    mask = obj_to_mask(
                        mask_device, annotation_type, obj, img_w=image_width, img_h=image_height, points=points
                    )
                except Exception:
                    print(f"DEBUG Exception Obj to Points & Mask for {annotation_type}: {obj}")
                    raise
                if mask is not None and mask.shape != (image_height, image_width):
                    raise ValueError(f"Wrong mask tensor shape: {mask.shape} != {(image_height, image_width)}")
                bounding_box = None
                if mask is not None:
                    try:
                        if self.config.device.type == "mps":
                            bounding_box = masks_to_boxes(mask.unsqueeze(0)).squeeze(0)
                            mask = mask.to(self.config.device)
                        else:
                            mask = mask.to(self.config.device)
                            bounding_box = masks_to_boxes(mask.unsqueeze(0)).squeeze(0)
                    except Exception:
                        print(
                            f"DEBUG Exception BB-Gen Failure: {mask.shape}, {torch.min(mask)}, "
                            f"{torch.max(mask)}, {annotation_type} => {obj}"
                        )
                        raise
                annotations[obj_hash] = AnnotationMetadata(
                    feature_hash=feature_hash,
                    annotation_type=annotation_type,
                    points=points,
                    mask=mask,
                    annotation_email=str(obj["createdBy"]),
                    annotation_manual=bool(obj["manualAnnotation"]),
                    annotation_confidence=float(obj["confidence"]),
                    bounding_box=bounding_box.cpu() if bounding_box is not None else None,
                )
        for cls in du_meta.classifications:
            cls_hash = str(cls["classificationHash"])
            feature_hash = str(cls["featureHash"])
            if cls_hash in annotations:
                hash_collision.add(cls_hash)
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

    def _get_frame_input(
        self, image: Image.Image, data_meta: ProjectDataMetadata, du_meta: ProjectDataUnitMetadata
    ) -> BaseFrameInput:
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
            data_type=DataType(data_meta.data_type),  # type: ignore
        )

    def _stage_2(self):
        # Stage 2: Support for complex value based dependencies.
        # Note: in-efficient, but needed for first version of this.
        # Stage 2 evaluation will always be very closely bound to the executor
        # and the associated storage and indexing strategy.
        for metric in tqdm(self.config.derived_embeddings, desc="Metric Computation Stage 2: Building indices"):
            metrics_tqdm_desc = f"                          : Generating indices for {metric.ident}"
            metrics_tqdm = tqdm([(False, self.data_metrics), (True, self.annotation_metrics)], desc=metrics_tqdm_desc)
            for is_annotation_metrics, metrics in metrics_tqdm:
                metrics_tqdm.set_description(metrics_tqdm_desc)
                if len(metrics) == 0:
                    continue
                if isinstance(metric, NearestImageEmbeddingQuery):
                    # Embedding query lookup
                    max_neighbours = int(metric.max_neighbours)
                    if max_neighbours > 50:
                        raise ValueError(f"Too many embedding nn-query results: {metric.ident} => {max_neighbours}")
                    if metric.similarity == EmbeddingDistanceMetric.RANDOM:
                        raise ValueError("Please explicitly request random distance metric as limits are different")
                    metrics_subset_list = [metrics]
                    if metric.same_feature_hash and is_annotation_metrics:
                        metrics_subset_list_dict = {}
                        for metric_key, metric_value in metrics.items():
                            metrics_subset_list_dict.setdefault(metric_value["feature_hash"], {})[
                                metric_key
                            ] = metric_value
                        metrics_subset_list = list(metrics_subset_list_dict.values())
                    elif metric.same_feature_hash:
                        continue  # This is an annotation only query.

                    for metrics_subset in metrics_subset_list:
                        data_keys = list(metrics_subset.keys())
                        metrics_tqdm.set_description(f"{metrics_tqdm_desc}  [Loading data]")
                        data_list = [metrics_subset[k][metric.embedding_source].cpu().numpy() for k in data_keys]
                        metrics_tqdm.set_description(f"{metrics_tqdm_desc}  [Preparing data]")
                        data = np.stack(data_list)
                        # FIXME: tune threshold where using pynndescent is actually faster.
                        if len(data_list) <= 4096:
                            # Exhaustive search is faster & more correct for very small datasets.
                            metrics_tqdm.set_description(f"{metrics_tqdm_desc}  [Exhaustive search]")
                            data_indices_list = []
                            data_distances_list = []
                            for data_key, data_embedding in zip(data_keys, data_list):
                                if metric.similarity == EmbeddingDistanceMetric.EUCLIDEAN:
                                    data_raw_offsets = data - data_embedding
                                    data_raw_similarities = np.linalg.norm(data_raw_offsets, axis=1)
                                elif metric.similarity == EmbeddingDistanceMetric.COSINE:
                                    data_raw_dot = np.dot(data, data_embedding)
                                    data_raw_data_dist = np.linalg.norm(data, axis=1)
                                    data_raw_dist = np.linalg.norm(data_embedding)
                                    data_raw_similarities = data_raw_dot / (data_raw_data_dist * data_raw_dist)
                                else:
                                    raise RuntimeError(f"Unknown similarity metric: {metric.similarity}")
                                data_raw_index = np.flip(np.argsort(data_raw_similarities)[:max_neighbours])
                                data_indices_list.append(data_raw_index)
                                data_distances_list.append(data_raw_similarities[data_raw_index])
                            data_indices = np.stack(data_indices_list)
                            data_distances = np.stack(data_distances_list)
                        else:
                            metrics_tqdm.set_description(f"{metrics_tqdm_desc}  [Creating index]")
                            nn_descent = NNDescent(
                                data=data, parallel_batch_queries=True, n_jobs=-1, metric=metric.similarity.name.lower()
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
                                embedding_type=metric.similarity,
                            )
                            metrics[k][metric.ident] = nearest_neighbor_result
                elif isinstance(metric, RandomSamplingQuery):
                    # Random sampling
                    limit = metric.limit
                    if limit > 1000:
                        raise ValueError(f"Too large random sampling amount: {limit}")
                    data_keys = list(metrics.keys())
                    for i, dk in enumerate(data_keys):
                        other_data_keys = data_keys[:i] + data_keys[i + 1 :]
                        if metric.same_feature_hash and is_annotation_metrics:
                            other_data_keys = [
                                k
                                for k in other_data_keys
                                if metrics[k].get("feature_hash", None)
                                == metrics[data_keys[i]].get("feature_hash", None)
                            ]
                        chosen_keys = random.choices(other_data_keys, k=min(limit, len(other_data_keys)))
                        metrics[dk][metric.ident] = RandomSampling(
                            metric_deps=[dict(metrics[k]) for k in chosen_keys], metric_keys=chosen_keys
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
                    if isinstance(derived_metric, tuple):
                        metric_res, metric_comment = derived_metric
                        data_metric[metric.ident] = metric_res
                        data_metric["metric_metadata"][metric.ident] = metric_comment
                    else:
                        data_metric[metric.ident] = derived_metric
        for annotation_metric in tqdm(
            self.annotation_metrics.values(), desc="Metric Computation Stage 3: Derived - Annotation"
        ):
            for metric in self.config.derived_metrics:
                derived_metric = metric.calculate(dict(annotation_metric), annotation_metric["annotation_type"])
                if derived_metric is not None:
                    if isinstance(derived_metric, tuple):
                        metric_res, metric_comment = derived_metric
                        annotation_metric[metric.ident] = metric_res
                        annotation_metric["metric_metadata"][metric.ident] = metric_comment
                    else:
                        annotation_metric[metric.ident] = derived_metric

    def _validate_pack_deps(
        self,
        deps: MetricDependencies,
        ty_analytics: Type[Union[ProjectPredictionAnalytics, ProjectAnnotationAnalytics, ProjectDataAnalytics]],
        ty_extra: Type[
            Union[ProjectPredictionAnalyticsExtra, ProjectAnnotationAnalyticsExtra, ProjectDataAnalyticsExtra]
        ],
        ty_derived: Type[
            Union[ProjectPredictionAnalyticsDerived, ProjectAnnotationAnalyticsDerived, ProjectDataAnalyticsDerived]
        ],
        ty_extra_metric: Set[str],
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        metric_lookup = {
            m.ident: m.metric_type
            for m in (self.config.analysis + self.config.derived_metrics)
            if isinstance(m, BaseMetric)
        }
        # Hardcoded special metric
        metric_lookup["metric_confidence"] = MetricType.NORMAL
        analysis_values = {
            k for k in deps if k in ty_analytics.__fields__ and (k.startswith("metric_") or k in ty_extra_metric)
        }
        for k in analysis_values:
            if k in ty_extra_metric:
                continue
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
        # FIXME: logic here is broken
        extra_values = {v for v in (set(deps.keys()) - analysis_values) if not v.startswith("derived_")}
        derived_values = {v for v in (set(deps.keys()) - analysis_values) if v.startswith("derived_")}
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
                not (extra_value.startswith("metric_") or extra_value.startswith("embedding_"))
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
        return analysis_values, extra_values, derived_values

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
                return v.cpu().numpy().astype(np.float64).tobytes()
            return v.item()
        elif isinstance(v, NearestNeighbors):
            return {
                "similarities": v.similarities,
                "keys": [[str(kv) if isinstance(kv, uuid.UUID) else kv for kv in k] for k in v.metric_keys],
            }
        elif isinstance(v, RandomSampling):
            return {"keys": [[str(kv) if isinstance(kv, uuid.UUID) else kv for kv in k] for k in v.metric_keys]}
        return v

    @classmethod
    def _get_user_id(
        cls,
        email: str,
        lookup: Dict[str, int],
        new_user_uuid: List[ProjectCollaborator],
        project_hash: uuid.UUID,
    ) -> int:
        if email in lookup:
            return lookup[email]
        else:
            user_id = len(lookup)
            lookup[email] = user_id
            new_user_uuid.append(
                ProjectCollaborator(
                    project_hash=project_hash,
                    user_id=user_id,
                    user_email=email,
                )
            )
            return user_id

    def _pack_output(
        self,
        project_hash: uuid.UUID,
        collaborators: List[ProjectCollaborator],
    ) -> Tuple[
        List[ProjectDataAnalytics],
        List[ProjectDataAnalyticsExtra],
        List[ProjectDataAnalyticsDerived],
        List[ProjectAnnotationAnalytics],
        List[ProjectAnnotationAnalyticsExtra],
        List[ProjectAnnotationAnalyticsDerived],
        List[ProjectCollaborator],
    ]:
        data_analysis = []
        data_analysis_extra = []
        data_analysis_derived = []
        annotation_analysis = []
        annotation_analysis_extra = []
        annotation_analysis_derived = []
        new_collaborators: List[ProjectCollaborator] = []
        user_email_lookup = {collab.user_email: collab.user_id for collab in collaborators}
        for (du_hash, frame), data_deps in self.data_metrics.items():
            data_type = data_deps.pop("data_type")
            analysis_values, extra_values, derived_values = self._validate_pack_deps(
                data_deps,
                ProjectDataAnalytics,
                ProjectDataAnalyticsExtra,
                ProjectDataAnalyticsDerived,
                set(),
            )
            data_analysis.append(
                ProjectDataAnalytics(
                    project_hash=project_hash,
                    du_hash=du_hash,
                    frame=frame,
                    data_type=data_type,
                    **{k: self._pack_metric(v) for k, v in data_deps.items() if k in analysis_values},
                )
            )
            data_analysis_extra.append(
                ProjectDataAnalyticsExtra(
                    project_hash=project_hash,
                    du_hash=du_hash,
                    frame=frame,
                    **{k: self._pack_extra(k, v) for k, v in data_deps.items() if k in extra_values},
                )
            )
            for k, v in data_deps.items():
                if k not in derived_values:
                    continue
                if isinstance(v, (RandomSampling, NearestNeighbors)):
                    for idx, (dep_du_hash, dep_frame) in enumerate(v.metric_keys):  # type: ignore
                        data_analysis_derived.append(
                            ProjectDataAnalyticsDerived(
                                project_hash=project_hash,
                                du_hash=du_hash,
                                frame=frame,
                                distance_metric=v.embedding_type
                                if isinstance(v, NearestNeighbors)
                                else EmbeddingDistanceMetric.RANDOM,
                                distance_index=idx,
                                similarity=v.similarities[idx] if isinstance(v, NearestNeighbors) else None,
                                dep_du_hash=dep_du_hash,
                                dep_frame=dep_frame,
                            )
                        )
                else:
                    raise ValueError(f"Unknown derived type: {v}")
        for (du_hash, frame, annotation_hash), annotation_deps in self.annotation_metrics.items():
            feature_hash = annotation_deps.pop("feature_hash")
            annotation_type = annotation_deps.pop("annotation_type")
            annotation_email = annotation_deps.pop("annotation_email")
            annotation_manual = annotation_deps.pop("annotation_manual")
            annotation_invalid = annotation_deps.pop("annotation_invalid")
            analysis_values, extra_values, derived_values = self._validate_pack_deps(
                annotation_deps,
                ProjectAnnotationAnalytics,
                ProjectAnnotationAnalyticsExtra,
                ProjectAnnotationAnalyticsDerived,
                set(),
            )
            annotation_analysis.append(
                ProjectAnnotationAnalytics(
                    project_hash=project_hash,
                    du_hash=du_hash,
                    frame=frame,
                    annotation_hash=annotation_hash,
                    feature_hash=feature_hash,
                    annotation_type=annotation_type,
                    annotation_user_id=self._get_user_id(
                        str(annotation_email), user_email_lookup, new_collaborators, project_hash
                    ),
                    annotation_manual=annotation_manual,
                    annotation_invalid=annotation_invalid,
                    **{k: self._pack_metric(v) for k, v in annotation_deps.items() if k in analysis_values},
                )
            )
            annotation_analysis_extra.append(
                ProjectAnnotationAnalyticsExtra(
                    project_hash=project_hash,
                    du_hash=du_hash,
                    frame=frame,
                    annotation_hash=annotation_hash,
                    **{k: self._pack_extra(k, v) for k, v in annotation_deps.items() if k in extra_values},
                )
            )
            for k, v in annotation_deps.items():
                if k not in derived_values:
                    continue
                if isinstance(v, (RandomSampling, NearestNeighbors)):
                    for idx, (dep_du_hash, dep_frame, dep_annotation_hash) in enumerate(v.metric_keys):  # type: ignore
                        annotation_analysis_derived.append(
                            ProjectAnnotationAnalyticsDerived(
                                project_hash=project_hash,
                                du_hash=du_hash,
                                frame=frame,
                                annotation_hash=annotation_hash,
                                distance_metric=v.embedding_type
                                if isinstance(v, NearestNeighbors)
                                else EmbeddingDistanceMetric.RANDOM,
                                distance_index=idx,
                                similarity=v.similarities[idx] if isinstance(v, NearestNeighbors) else None,
                                dep_du_hash=dep_du_hash,
                                dep_frame=dep_frame,
                                dep_annotation_hash=dep_annotation_hash,
                            )
                        )
                else:
                    raise ValueError(f"Unknown derived type: {v}")

        return (
            data_analysis,
            data_analysis_extra,
            data_analysis_derived,
            annotation_analysis,
            annotation_analysis_extra,
            annotation_analysis_derived,
            new_collaborators,
        )

    def _pack_prediction(
        self,
        project_hash: uuid.UUID,
        prediction_hash: uuid.UUID,
        collaborators: List[ProjectCollaborator],
    ) -> Tuple[
        List[ProjectPredictionAnalytics],
        List[ProjectPredictionAnalyticsExtra],
        List[ProjectPredictionAnalyticsDerived],
        List[ProjectPredictionAnalyticsFalseNegatives],
        List[ProjectCollaborator],
    ]:
        prediction_analysis = []
        prediction_analysis_extra = []
        prediction_analysis_derived = []
        prediction_false_negatives = []
        user_email_lookup = {collab.user_email: collab.user_id for collab in collaborators}
        new_collaborators: List[ProjectCollaborator] = []
        for (du_hash, frame, annotation_hash), annotation_deps in self.annotation_metrics.items():
            feature_hash = annotation_deps.pop("feature_hash")
            annotation_type = annotation_deps.pop("annotation_type")
            annotation_email = annotation_deps.pop("annotation_email")
            annotation_manual = annotation_deps.pop("annotation_manual")
            annotation_invalid = annotation_deps.pop("annotation_invalid")
            analysis_values, extra_values, derived_values = self._validate_pack_deps(
                annotation_deps,
                ProjectPredictionAnalytics,
                ProjectPredictionAnalyticsExtra,
                ProjectPredictionAnalyticsDerived,
                {"iou", "match_annotation_hash", "match_feature_hash", "match_duplicate_iou"},
            )
            prediction_analysis.append(
                ProjectPredictionAnalytics(
                    prediction_hash=prediction_hash,
                    project_hash=project_hash,
                    du_hash=du_hash,
                    frame=frame,
                    annotation_hash=annotation_hash,
                    feature_hash=feature_hash,
                    annotation_type=annotation_type,
                    annotation_user_id=self._get_user_id(
                        str(annotation_email), user_email_lookup, new_collaborators, project_hash
                    ),
                    annotation_manual=annotation_manual,
                    annotation_invalid=annotation_invalid,
                    **{k: self._pack_metric(v) for k, v in annotation_deps.items() if k in analysis_values},
                )
            )
            prediction_analysis_extra.append(
                ProjectPredictionAnalyticsExtra(
                    prediction_hash=prediction_hash,
                    project_hash=project_hash,
                    du_hash=du_hash,
                    frame=frame,
                    annotation_hash=annotation_hash,
                    **{k: self._pack_extra(k, v) for k, v in annotation_deps.items() if k in extra_values},
                )
            )
            for k, v in annotation_deps.items():
                if k not in derived_values:
                    continue
                if isinstance(v, (RandomSampling, NearestNeighbors)):
                    for idx, (dep_du_hash, dep_frame, dep_annotation_hash) in enumerate(v.metric_keys):  # type: ignore
                        prediction_analysis_derived.append(
                            ProjectPredictionAnalyticsDerived(
                                prediction_hash=prediction_hash,
                                project_hash=project_hash,
                                du_hash=du_hash,
                                frame=frame,
                                annotation_hash=annotation_hash,
                                distance_metric=v.embedding_type
                                if isinstance(v, NearestNeighbors)
                                else EmbeddingDistanceMetric.RANDOM,
                                distance_index=idx,
                                similarity=v.similarities[idx] if isinstance(v, NearestNeighbors) else None,
                                dep_du_hash=dep_du_hash,
                                dep_frame=dep_frame,
                                dep_annotation_hash=dep_annotation_hash,
                            )
                        )
                else:
                    raise ValueError(f"Unknown derived type: {v}")

        for (du_hash, frame, annotation_hash), (feature_hash, iou_threshold) in self.prediction_false_negatives.items():
            prediction_false_negatives.append(
                ProjectPredictionAnalyticsFalseNegatives(
                    prediction_hash=prediction_hash,
                    project_hash=project_hash,
                    du_hash=du_hash,
                    frame=frame,
                    annotation_hash=annotation_hash,
                    iou_threshold=iou_threshold,
                    feature_hash=feature_hash,
                )
            )

        return (
            prediction_analysis,
            prediction_analysis_extra,
            prediction_analysis_derived,
            prediction_false_negatives,
            new_collaborators,
        )
