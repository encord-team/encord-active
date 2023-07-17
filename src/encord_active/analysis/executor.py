import uuid
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from PIL import Image
from pynndescent import NNDescent

from encord_active.analysis.types import MetricResult, AnnotationMetadata
from encord_active.analysis.util.torch import pillow_to_tensor, obj_to_points, obj_to_mask
from .base import BaseFrameInput, BaseAnalysis

from encord_active.analysis.config import AnalysisConfig
from encord_active.db.models import ProjectDataMetadata, ProjectDataUnitMetadata, AnnotationType, \
    ProjectAnnotationAnalytics, ProjectAnnotationAnalyticsExtra, ProjectDataAnalytics, ProjectDataAnalyticsExtra
from encord_active.lib.common.data_utils import download_image


class Executor:
    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config


class SimpleExecutor(Executor):
    def __init__(self, config: AnalysisConfig) -> None:
        super().__init__(config=config)
        self.data_metrics: Dict[Tuple[uuid.UUID, int], Dict[str, MetricResult]] = {}
        self.annotation_metrics: Dict[Tuple[uuid.UUID, int, str], Dict[str, MetricResult]] = {}
        self.data_nn_descent: Dict[str, NNDescent] = {}
        self.annotation_nn_descent: Dict[str, NNDescent] = {}

    def execute(
        self,
        values: List[Tuple[ProjectDataMetadata, List[ProjectDataUnitMetadata]]],
        project_dir: Path,
    ) -> Tuple[
        List[ProjectDataAnalytics],
        List[ProjectDataAnalyticsExtra],
        List[ProjectAnnotationAnalytics],
        List[ProjectAnnotationAnalyticsExtra]
    ]:
        with torch.no_grad():
            for data_metadata, du_metadata_list in values:
                du_metadata_list.sort(key=lambda v: (v.du_hash, v.frame))
                if data_metadata.data_type == "video":
                    raise ValueError(f"Video is not supported")
                else:
                    for du_metadata in du_metadata_list:
                        if du_metadata.data_uri_is_video:
                            raise ValueError(f"Invalid du_metadata input, image marked as video")
                        image = download_image(du_metadata.data_uri, project_dir=project_dir)
                        frame_input = SimpleExecutor._get_frame_input(
                            image=image,
                            du_meta=du_metadata,
                        )
                        self._execute_stage_1_metrics(
                            prev_frame=None,
                            current_frame=frame_input,
                            next_frame=None,
                            du_hash=du_metadata.du_hash,
                            frame=du_metadata.frame,
                        )

        # Post-processing stages
        self._stage_2()
        self._stage_3()
        return self._pack_output()

    def _execute_stage_1_metrics(
        self,
        prev_frame: Optional[BaseFrameInput],
        current_frame: BaseFrameInput,
        next_frame: Optional[BaseFrameInput],
        du_hash: uuid.UUID,
        frame: int,
    ) -> None:
        # FIXME: add sanity checking that values are not mutated by metric runs
        image_results_deps = {
            k: {}
            for k in current_frame.annotations_deps.keys()
        }
        annotation_results_deps = {}
        for evaluation in self.config.analysis:
            metric_result = evaluation.raw_calculate(
                prev_frame=prev_frame,
                frame=current_frame,
                next_frame=next_frame,
            )
            non_ephemeral = isinstance(evaluation, BaseAnalysis)
            if metric_result.image is not None:
                current_frame.image_deps[evaluation.ident] = metric_result
                if non_ephemeral:
                    image_results_deps[evaluation.ident] = metric_result
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

    @staticmethod
    def _get_frame_input(
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
                annotations[obj_hash] = AnnotationMetadata(
                    feature_hash=feature_hash,
                    annotation_type=annotation_type,
                    points=obj_to_points(obj, image.width, image.height),
                    mask=obj_to_mask(obj, image.width, image.height),
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
                    annotation_type=AnnotationType,
                    points=None,
                    mask=None,
                )
                annotations_deps[cls_hash] = {}

        return BaseFrameInput(
            image=pillow_to_tensor(image),
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
        for metric in self.config.derived_embeddings:
            max_neighbours = int(metric.max_neighbours)
            if max_neighbours > 50:
                raise ValueError(f"Too many embedding nn-query results")
            data = [
                values[metric.ident].cpu().detach().numpy()
                for values in self.data_metrics.values()
            ]
            nn_descent = NNDescent(
                data=data,
            )
            nn_descent.prepare()
            for values in self.data_metrics.values():
                values[metric.ident] = nn_descent.query(
                    query_data=values[metric.ident],
                    k=max_neighbours,
                )

    def _stage_3(self) -> None:
        # Stage 3: Derived metrics from stage 1 & stage 2 metrics
        # dependencies must be pure.
        for data_metric in self.data_metrics.values():
            for metric in self.config.derived_metrics:
                metric.calculate(data_metric)
        for annotation_metric in self.annotation_metrics.values():
            for metric in self.config.derived_metrics:
                metric.calculate(annotation_metric)

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

        for (du_hash, frame), data_deps in self.data_metrics.items():
            analysis_values = {
                k
                for k in data_deps
                if k in ProjectDataAnalytics.__fields__ and k.startswith("metric_")
            }
            extra_values = set(data_deps.keys()) - analysis_values
            for extra_value in extra_values:
                if not (
                    extra_value.startswith("metric_") or
                    extra_value.startswith("embedding_") or
                    extra_value.startswith("derived_")
                ) or extra_value not in ProjectDataAnalyticsExtra.__fields__:
                    raise ValueError(f"Unknown field name: {extra_value}")
            data_analysis.append(
                # FIXME: assign extra values correctly.
                ProjectDataAnalytics(
                    du_hash=du_hash,
                    frame=frame,
                    feature_hash="",
                    annotation_type=AnnotationType.CLASSIFICATION,
                    annotation_creator=None,
                    annotation_confidence=1.0,
                    **analysis_values,
                )
            )
        for annotation_deps in self.annotation_metrics:
            pass

        return (
            data_analysis,
            data_analysis_extra,
            annotation_analysis,
            annotation_analysis_extra
        )