import math
import uuid
import encord
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set, Union, Type

import av
import numpy as np
import torch
import torchvision
from PIL import Image
from pynndescent import NNDescent
from torch import Tensor
from torchvision.io import VideoReader
from torchvision.ops import masks_to_boxes
from tqdm import tqdm

from encord_active.analysis.types import AnnotationMetadata, MetricDependencies, NearestNeighbors
from encord_active.analysis.util.torch import pillow_to_tensor, obj_to_points, obj_to_mask
from .base import BaseFrameInput, BaseAnalysis, BaseFrameBatchInput

from encord_active.analysis.config import AnalysisConfig
from encord_active.db.models import ProjectDataMetadata, ProjectDataUnitMetadata, AnnotationType, \
    ProjectAnnotationAnalytics, ProjectAnnotationAnalyticsExtra, ProjectDataAnalytics, ProjectDataAnalyticsExtra
from encord_active.lib.common.data_utils import download_image, url_to_file_path
from .embedding import BaseEmbedding
from ..lib.encord.utils import get_encord_project


class Executor:
    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config


class SimpleExecutor(Executor):
    def __init__(self, config: AnalysisConfig) -> None:
        super().__init__(config=config)
        self.data_metrics: Dict[Tuple[uuid.UUID, int], MetricDependencies] = {}
        self.annotation_metrics: Dict[Tuple[uuid.UUID, int, str], MetricDependencies] = {}
        self.encord_projects: Dict[uuid.UUID, encord.Project] = {}

    def execute_from_db(
        self,
        data_meta: List[ProjectDataMetadata],
        du_meta: List[ProjectDataUnitMetadata],
        project_dir: Path,
        project_hash: uuid.UUID,
        project_ssh_path: Optional[str]
    ) -> Tuple[
        List[ProjectDataAnalytics],
        List[ProjectDataAnalyticsExtra],
        List[ProjectAnnotationAnalytics],
        List[ProjectAnnotationAnalyticsExtra]
    ]:
        du_dict = {}
        for du in du_meta:
            du_dict.setdefault(du.data_hash, []).append(du)
        values = [
            (data, du_dict[data.data_hash])
            for data in data_meta
        ]
        return self.execute(
            values, project_dir=project_dir, project_hash=project_hash, project_ssh_path=project_ssh_path
        )

    def execute(
        self,
        values: List[Tuple[ProjectDataMetadata, List[ProjectDataUnitMetadata]]],
        project_dir: Path,
        project_hash: uuid.UUID,
        project_ssh_path: Optional[str],
        batch: Optional[int] = None,  # Disable the batch execution path, use legacy code-path instead.
    ) -> Tuple[
        List[ProjectDataAnalytics],
        List[ProjectDataAnalyticsExtra],
        List[ProjectAnnotationAnalytics],
        List[ProjectAnnotationAnalyticsExtra]
    ]:
        # Split into videos & images
        video_values = [
            v for v in values if v[0].data_type == "video"
        ]
        image_values = [
            (d, dle)
            for d, dl in values
            if d.data_type != "video"
            for dle in dl
        ]
        image_values.sort(key=lambda d: (d[1].width, d[1].height))
        for data_metadata, du_metadata_list in video_values:
            # Well-formed
            du_metadata_list.sort(key=lambda v: (v.du_hash, v.frame))
            if data_metadata.data_type != "video":
                raise ValueError("Bug separating videos")

            # Run assertions
            if du_metadata_list[0].du_hash != du_metadata_list[-1].du_hash:
                raise ValueError(f"Video has different du_hash")
            if du_metadata_list[-1].frame + 1 != len(du_metadata_list) or du_metadata_list[0].frame != 0:
                raise ValueError(f"Video has skipped frames")
            if len({x.frame for x in du_metadata_list}) != len(du_metadata_list):
                raise ValueError(f"Video has frame duplicates")

        # Video best thing
        with torch.no_grad():
            # Stage 1
            if batch is None:
                self._stage_1_video_no_batching(
                    video_values,
                    project_dir=project_dir,
                    project_hash=project_hash,
                    project_ssh_path=project_ssh_path,
                )
                self._stage_1_image_no_batching(
                    image_values,
                    project_dir=project_dir,
                    project_hash=project_hash,
                    project_ssh_path=project_ssh_path,
                )
            else:
                self._stage_1_video_batching(
                    video_values,
                    project_dir=project_dir,
                    project_hash=project_hash,
                    project_ssh_path=project_ssh_path,
                )
                self._stage_1_image_batching(
                    image_values,
                    project_dir=project_dir,
                    project_hash=project_hash,
                    project_ssh_path=project_ssh_path,
                    batching=batch,
                )

            # Post-processing stages
            self._stage_2()
            self._stage_3()
            return self._pack_output()

    def _stage_1_video_batching(
        self,
        videos: List[Tuple[ProjectDataMetadata, List[ProjectDataUnitMetadata]]],
        project_dir: Path,
        project_hash: uuid.UUID,
        project_ssh_path: Optional[str],
    ):
        if len(videos) == 0:
            return
        # Select video backend
        if self.config.device.type == "cuda":
            torchvision.set_video_backend("cuda")
        else:
            torchvision.set_video_backend("pyav")
        # Start execution
        for data_metadata, du_meta_list in tqdm(
            videos, desc=f"Metric Computation Stage 1: Videos ({torchvision.get_video_backend()})"
        ):
            video_start = du_meta_list[0]
            video_url_or_path = self._get_url_or_path(
                uri=video_start.data_uri, project_dir=project_dir, project_hash=project_hash,
                project_ssh_path=project_ssh_path, label_hash=data_metadata.label_hash,
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
        project_dir: Path,
        project_hash: uuid.UUID,
        project_ssh_path: Optional[str],
    ):
        if len(videos) == 0:
            return
        for data_metadata, du_meta_list in tqdm(
            videos, desc="Metric Computation Stage 1: Videos"
        ):
            video_start = du_meta_list[0]
            video_url_or_path = self._get_url_or_path(
                uri=video_start.data_uri, project_dir=project_dir, project_hash=project_hash,
                project_ssh_path=project_ssh_path, label_hash=data_metadata.label_hash,
                du_hash=video_start.du_hash,
            )
            prev_frame: Optional[BaseFrameInput] = None
            curr_frame: Optional[BaseFrameInput] = None
            next_frame: Optional[BaseFrameInput] = None
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
                        raise ValueError(f"Corrupt video frame")
                    if iter_frame_num + 1 != frame_num:
                        raise ValueError(f"AV video frame out of sequence: {iter_frame_num} -> {frame_num}")
                    iter_frame_num = frame_num
                    frame_image = frame.to_image().convert("RGB")
                    frame_input = self._get_frame_input(
                        image=frame_image,
                        du_meta=frame_du_meta,
                    )
                    # Step through window
                    if prev_frame is None:
                        prev_frame = frame_input
                    elif curr_frame is None:
                        curr_frame = frame_input
                    elif next_frame is None:
                        next_frame = frame_input
                    else:
                        prev_frame = curr_frame
                        curr_frame = next_frame
                        next_frame = frame_input
                    # Run stage 1 metrics
                    if next_frame is not None:
                        # Process non-first & non-last frame
                        self._execute_stage_1_metrics_no_batching(
                            prev_frame=prev_frame,
                            current_frame=curr_frame,
                            next_frame=next_frame,
                            du_hash=du_meta_list[0].du_hash,
                            frame=iter_frame_num - 1,
                        )
                    elif curr_frame is not None:
                        # Process first frame
                        self._execute_stage_1_metrics_no_batching(
                            prev_frame=None,
                            current_frame=prev_frame,
                            next_frame=curr_frame,
                            du_hash=du_meta_list[0].du_hash,
                            frame=iter_frame_num - 1,
                        )

                # Process final frame
                self._execute_stage_1_metrics_no_batching(
                    prev_frame=curr_frame,
                    current_frame=next_frame,
                    next_frame=None,
                    du_hash=du_meta_list[0].du_hash,
                    frame=iter_frame_num,
                )

    def _stage_1_image_no_batching(
        self,
        images: List[Tuple[ProjectDataMetadata, ProjectDataUnitMetadata]],
        project_dir: Path,
        project_hash: uuid.UUID,
        project_ssh_path: Optional[str],
    ):
        for data_metadata, du_meta in tqdm(
            images, desc="Metric Computation Stage 1: Images"
        ):
            frame_input = self._get_frame_input(
                image=self._open_image(
                    uri=du_meta.data_uri, project_dir=project_dir, project_hash=project_hash,
                    project_ssh_path=project_ssh_path, label_hash=data_metadata.label_hash,
                    du_hash=du_meta.du_hash,
                ),
                du_meta=du_meta
            )
            self._execute_stage_1_metrics_no_batching(
                prev_frame=None,
                current_frame=frame_input,
                next_frame=None,
                du_hash=du_meta.du_hash,
                frame=du_meta.frame,
            )

    def _stage_1_image_batching(
        self,
        values: List[Tuple[ProjectDataMetadata, ProjectDataUnitMetadata]],
        project_dir: Path,
        project_hash: uuid.UUID,
        project_ssh_path: Optional[str],
        batching: int,
    ) -> None:
        grouped_values: Dict[Tuple[int, int], List[Tuple[ProjectDataMetadata, ProjectDataUnitMetadata]]] = {}
        for data_metadata, du_metadata in values:
            grouped_values.setdefault((du_metadata.width, du_metadata.height), []).append((data_metadata, du_metadata))
        for group in grouped_values.values():
            for i in range(0, len(group), batching):
                grouped_batch = group[i:i + batching]
                frame_inputs = [
                    self._get_frame_input(
                        image=self._open_image(
                            uri=du_meta.data_uri, project_dir=project_dir, project_hash=project_hash,
                            project_ssh_path=project_ssh_path, label_hash=meta.label_hash,
                            du_hash=du_meta.du_hash,
                        ),
                        du_meta=du_meta
                    )
                    for meta, du_meta in grouped_batch
                ]

                batch_inputs = BaseFrameBatchInput(
                    images=torch.stack([frame.image for frame in frame_inputs]),
                    images_deps={},
                    annotations=None, # FIXME: actually assign these values to something correct!!
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
                        batch_inputs.images[evaluation.ident] = metric_result.images
                        if non_ephemeral:
                            image_results[evaluation.ident] = metric_result.images
                    if metric_result.objects is not None:
                        batch_inputs.annotations.objects_deps[evaluation.ident] = metric_result.objects
                        if non_ephemeral:
                            object_results[evaluation.ident] = metric_result.objects
                    if metric_result.classifications is not None:
                        batch_inputs.annotations.classifications_deps[evaluation.ident] = metric_result.classifications
                        if non_ephemeral:
                            classification_results[evaluation.ident] = metric_result.classifications

    def _open_image(
        self,
        uri: Optional[str],
        project_dir: Path,
        project_hash: uuid.UUID,
        project_ssh_path: Optional[str],
        label_hash: uuid.UUID,
        du_hash: uuid.UUID,
    ):
        img_url_or_path = self._get_url_or_path(
            uri=uri, project_dir=project_dir, project_hash=project_hash,
            project_ssh_path=project_ssh_path, label_hash=label_hash,
            du_hash=du_hash,
        )
        if isinstance(img_url_or_path, Path):
            return Image.open(img_url_or_path)
        return download_image(img_url_or_path, project_dir=project_dir, cache=False)

    def _get_url_or_path(
        self,
        uri: Optional[str],
        project_dir: Path,
        project_hash: uuid.UUID,
        project_ssh_path: Optional[str],
        label_hash: uuid.UUID,
        du_hash: uuid.UUID,
    ) -> Union[str, Path]:
        if uri is not None:
            url_path = url_to_file_path(uri, project_dir)
            if url_path is not None:
                return url_path
            else:
                return uri
        elif project_ssh_path is not None:
            if project_hash in self.encord_projects:
                encord_project = self.encord_projects[project_hash]
            else:
                encord_project = get_encord_project(
                    ssh_key_path=project_ssh_path, project_hash=str(project_hash)
                )
                self.encord_projects[project_hash] = encord_project
            return encord_project.get_label_row(
                str(label_hash), get_signed_url=True,
            )["data_units"][str(du_hash)]["data_link"]
        else:
            raise ValueError(f"Cannot resolve project url: {project_hash} / {label_hash} / {du_hash}")

    def _execute_stage_1_metrics_no_batching(
        self,
        prev_frame: Optional[BaseFrameInput],
        current_frame: BaseFrameInput,
        next_frame: Optional[BaseFrameInput],
        du_hash: uuid.UUID,
        frame: int,
    ) -> None:
        # FIXME: add sanity checking that values are not mutated by metric runs
        image_results_deps = {
            "metric_metadata": {}  # String comments (currently not used)
        }
        annotation_results_deps = {
            k: {
                "feature_hash": meta.feature_hash,
                "annotation_type": meta.annotation_type,
                "annotation_email": meta.annotation_email,
                "annotation_manual": meta.annotation_manual,
                "metric_label_confidence": meta.annotation_confidence,
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

        # Splat the results
        self.data_metrics[(du_hash, frame)] = image_results_deps
        for annotation_hash, annotation_deps in annotation_results_deps.items():
            self.annotation_metrics[
                (du_hash, frame, annotation_hash)
            ] = annotation_deps

    def _get_frame_input(
        self,
        image: Image.Image,
        du_meta: ProjectDataUnitMetadata
    ) -> BaseFrameInput:
        annotations = {}
        annotations_deps = {}
        hash_collision = False
        for obj in du_meta.objects:
            obj_hash = str(obj["objectHash"])
            feature_hash = str(obj["featureHash"])
            annotation_type = AnnotationType(str(obj["shape"]))
            if obj_hash in annotations:
                hash_collision = True
            else:
                points = obj_to_points(annotation_type, obj, img_w=image.width, img_h=image.height)
                mask = obj_to_mask(
                    self.config.device, annotation_type, img_w=image.width, img_h=image.height, points=points
                )
                if mask is not None and mask.shape != (image.height, image.width):
                    raise ValueError(f"Wrong mask tensor shape: {mask.shape} != {(image.height, image.width)}")
                annotations[obj_hash] = AnnotationMetadata(
                    feature_hash=feature_hash,
                    annotation_type=annotation_type,
                    points=None if points is None else points.to(self.config.device),
                    mask=None if mask is None else mask.to(self.config.device),
                    annotation_email=str(obj["createdBy"]),
                    annotation_manual=bool(obj["manualAnnotation"]),
                    annotation_confidence=float(obj["confidence"]),
                    bounding_box=None if mask is None else masks_to_boxes(mask.unsqueeze(0)).to(self.config.device)
                )
                annotations_deps[obj_hash] = {}
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
                annotations_deps[cls_hash] = {}

        image_tensor = pillow_to_tensor(image).to(self.config.device)
        if image_tensor.shape != (3, image.height, image.width):
            raise ValueError(f"Wrong image tensor shape: {image_tensor.shape} != {(3, image.height, image.width)}")
        return BaseFrameInput(
            image=image_tensor,
            image_deps={},
            annotations=annotations,
            annotations_deps=annotations_deps,
            hash_collision=hash_collision
        )

    def _stage_2(self):
        # Stage 2: Support for complex value based dependencies.
        # Note: in-efficient, but needed for first version of this.
        # Stage 2 evaluation will always be very closely bound to the executor
        # and the associated storage and indexing strategy.
        for metric in tqdm(self.config.derived_embeddings, desc="Metric Computation Stage 2: Building indices"):
            for metrics in tqdm(
                [self.data_metrics, self.annotation_metrics],
                desc=f"                          : Generating indices for {metric.ident}"
            ):
                if len(metrics) == 0:
                    continue
                max_neighbours = int(metric.max_neighbours)
                if max_neighbours > 50:
                    raise ValueError(f"Too many embedding nn-query results")
                data_keys = list(metrics.keys())
                data_list = [
                    metrics[k][metric.embedding_source].cpu().numpy()
                    for k in data_keys
                ]
                data = np.stack(data_list)
                nn_descent = NNDescent(
                    data=data,
                    parallel_batch_queries=True,
                    n_jobs=-1,
                )
                nn_descent.prepare()
                data_indices, data_distances = nn_descent.query(
                    query_data=data,
                    k=max_neighbours,
                )
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
                        metric_deps=[
                            dict(metrics[data_keys[int(j)]])
                            for j in indices.tolist()
                        ],
                        metric_keys=[
                            data_keys[int(j)]
                            for j in indices.tolist()
                        ]
                    )
                    metrics[k][metric.ident] = nearest_neighbor_result

    def _stage_3(self) -> None:
        # Stage 3: Derived metrics from stage 1 & stage 2 metrics
        # dependencies must be pure.
        for data_metric in tqdm(
            self.data_metrics.values(),
            desc="Metric Computation Stage 3: Derived - Data"
        ):
            for metric in self.config.derived_metrics:
                derived_metric = metric.calculate(dict(data_metric), None)
                if derived_metric is not None:
                    data_metric[metric.ident] = derived_metric
        for annotation_metric in tqdm(
            self.annotation_metrics.values(),
            desc="Metric Computation Stage 3: Derived - Annotation"
        ):
            for metric in self.config.derived_metrics:
                derived_metric = metric.calculate(dict(annotation_metric), annotation_metric["annotation_type"])
                if derived_metric is not None:
                    annotation_metric[metric.ident] = derived_metric

    @staticmethod
    def _validate_pack_deps(
        deps: MetricDependencies,
        ty_analytics: Type[Union[ProjectAnnotationAnalytics, ProjectDataAnalytics]],
        ty_extra: Type[Union[ProjectAnnotationAnalyticsExtra, ProjectDataAnalyticsExtra]],
    ) -> Tuple[Set[str], Set[str]]:
        analysis_values = {
            k
            for k in deps
            if k in ty_analytics.__fields__ and k.startswith("metric_")
        }
        extra_values = set(deps.keys()) - analysis_values
        unassigned_metric_values = {
            k for k in ty_analytics.__fields__
            if (
                k.startswith("metric_") or k.startswith("embedding_")
            ) and k not in analysis_values and not k.startswith("metric_custom")
        }
        if len(unassigned_metric_values) > 0:
            print(f"WARNING: unassigned analytics metrics({ty_analytics.__name__}) = {unassigned_metric_values}")
        for extra_value in extra_values:
            if not (
                extra_value.startswith("metric_") or
                extra_value.startswith("embedding_") or
                extra_value.startswith("derived_")
            ) or extra_value not in ty_extra.__fields__:
                raise ValueError(f"Unknown field name: '{extra_value}' for {ty_extra.__name__}")
        unassigned_extra_values = {
            k for k in ty_extra.__fields__
            if (
               k.startswith("metric_") or k.startswith("embedding_") or k.startswith("derived_")
            ) and k not in extra_values
        }
        if len(unassigned_extra_values) > 0:
            print(f"WARNING: unassigned analytics extra value({ty_extra.__name__}) = {unassigned_extra_values}")
        return analysis_values, extra_values

    def _pack_output(self) -> Tuple[
        List[ProjectDataAnalytics],
        List[ProjectDataAnalyticsExtra],
        List[ProjectAnnotationAnalytics],
        List[ProjectAnnotationAnalyticsExtra]
    ]:
        data_analysis = []
        data_analysis_extra = []
        annotation_analysis = []
        annotation_analysis_extra = []

        def _metric(v: Union[Tensor, float, int, str, None]) -> Union[bytes, float, int, str, None]:
            if isinstance(v, Tensor):
                return v.item()
            return v

        def _extra(
            k: str, v: Union[Tensor, float, int, str, None, NearestNeighbors]
        ) -> Union[bytes, float, int, str, None, dict]:
            if isinstance(v, Tensor):
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
                    "keys": [
                        list(k)
                        for k in v.metric_keys
                    ]
                }
            return v

        for (du_hash, frame), data_deps in self.data_metrics.items():
            analysis_values, extra_values = SimpleExecutor._validate_pack_deps(
                data_deps,
                ProjectDataAnalytics,
                ProjectDataAnalyticsExtra,
            )
            data_analysis.append(
                ProjectDataAnalytics(
                    du_hash=du_hash,
                    frame=frame,
                    **{
                        k: _metric(v)
                        for k, v in data_deps.items() if k in analysis_values
                    },
                )
            )
            data_analysis_extra.append(
                ProjectDataAnalyticsExtra(
                    du_hash=du_hash,
                    frame=frame,
                    **{
                        k: _extra(k, v)
                        for k, v in data_deps.items() if k in extra_values and not k.startswith("derived_")
                    },
                )
            )
        for (du_hash, frame, annotation_hash), annotation_deps in self.annotation_metrics.items():
            feature_hash = annotation_deps.pop("feature_hash")
            annotation_type = annotation_deps.pop("annotation_type")
            annotation_email = annotation_deps.pop("annotation_email")
            annotation_manual = annotation_deps.pop("annotation_manual")
            analysis_values, extra_values = SimpleExecutor._validate_pack_deps(
                annotation_deps,
                ProjectAnnotationAnalytics,
                ProjectAnnotationAnalyticsExtra,
            )
            annotation_analysis.append(
                ProjectAnnotationAnalytics(
                    du_hash=du_hash,
                    frame=frame,
                    feature_hash=feature_hash,
                    annotation_type=annotation_type,
                    annotation_email=annotation_email,
                    annotation_manual=annotation_manual,
                    **{
                        k: _metric(v)
                        for k, v in annotation_deps.items() if k in analysis_values
                    },
                )
            )
            annotation_analysis_extra.append(
                ProjectAnnotationAnalyticsExtra(
                    du_hash=du_hash,
                    frame=frame,
                    **{
                        k: _extra(k, v)
                        for k, v in annotation_deps.items() if k in extra_values
                    },
                )
            )

        return (
            data_analysis,
            data_analysis_extra,
            annotation_analysis,
            annotation_analysis_extra
        )
