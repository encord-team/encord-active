import uuid
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set, Union, Type

import torch
from PIL import Image
from pynndescent import NNDescent
from torch import Tensor
from tqdm import tqdm

from encord_active.analysis.types import MetricResult, AnnotationMetadata, MetricDependencies
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

    def execute_from_db(
        self,
        data_meta: List[ProjectDataMetadata],
        du_meta: List[ProjectDataUnitMetadata],
        project_dir: Path,
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
        return self.execute(values, project_dir=project_dir)

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
            for data_metadata, du_metadata_list in tqdm(values, "Metric Stage 1: Image & Object computation"):
                du_metadata_list.sort(key=lambda v: (v.du_hash, v.frame))
                if data_metadata.data_type == "video":
                    # Run assertions
                    if du_metadata_list[0].du_hash != du_metadata_list[-1].du_hash:
                        raise ValueError(f"Video has different du_hash")
                    if du_metadata_list[-1].frame + 1 != len(du_metadata_list) or du_metadata_list[0].frame != 0:
                        raise ValueError(f"Video has skipped frames")
                    if len({x.frame for x in du_metadata_list}) != len(du_metadata_list):
                        raise ValueError(f"Video has frame duplicates")
                    video_start = du_metadata_list[0]
                    raise ValueError(f"Video is not supported")
                else:
                    for du_metadata in du_metadata_list:
                        if du_metadata.data_uri_is_video:
                            raise ValueError(f"Invalid du_metadata input, image marked as video")
                        frame_input = SimpleExecutor._get_frame_input(
                            image=download_image(du_metadata.data_uri, project_dir=project_dir, cache=False),
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
        image_results_deps = {}
        annotation_results_deps = {
            k: {
                "feature_hash": meta.feature_hash,
                "annotation_type": meta.annotation_type,
                "annotation_email": meta.annotation_email,
                "annotation_manual": meta.annotation_manual,
                "metric_label_confidence": meta.annotation_confidence,
            }
            for k, meta in current_frame.annotations.items()
        }
        for evaluation in self.config.analysis:
            metric_result = evaluation.raw_calculate(
                prev_frame=prev_frame,
                frame=current_frame,
                next_frame=next_frame,
            )
            non_ephemeral = isinstance(evaluation, BaseAnalysis)
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
                points = obj_to_points(annotation_type, obj, image.width, image.height)
                mask = obj_to_mask(annotation_type, image.width, image.height, points)
                annotations[obj_hash] = AnnotationMetadata(
                    feature_hash=feature_hash,
                    annotation_type=annotation_type,
                    points=points.cuda(),
                    mask=mask.cuda(),
                    annotation_email=str(obj["createdBy"]),
                    annotation_manual=bool(obj["manualAnnotation"]),
                    annotation_confidence=float(obj["confidence"]),
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
                )
                annotations_deps[cls_hash] = {}

        return BaseFrameInput(
            image=pillow_to_tensor(image).cuda(),
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
        for metric in tqdm(self.config.derived_embeddings, desc="Metric Stage 2: Building indices"):
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
        for extra_value in extra_values:
            if not (
                extra_value.startswith("metric_") or
                extra_value.startswith("embedding_") or
                extra_value.startswith("derived_")
            ) or extra_value not in ty_extra.__fields__:
                raise ValueError(f"Unknown field name: {extra_value}")
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

        def _extra(k: str, v: Union[Tensor, float, int, str, None]) -> Union[bytes, float, int, str, None]:
            if isinstance(v, Tensor):
                if k.startswith("embedding_"):
                    return v.cpu().numpy().tobytes()
                return v.item()
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
                        for k, v in data_deps if k in analysis_values
                    },
                )
            )
            data_analysis_extra.append(
                ProjectDataAnalyticsExtra(
                    du_hash=du_hash,
                    frame=frame,
                    **{
                        k: _extra(k, v)
                        for k, v in data_deps if k in extra_values and not k.startswith("derived_")
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
                        for k, v in annotation_deps if k in analysis_values
                    },
                )
            )
            annotation_analysis_extra.append(
                ProjectAnnotationAnalyticsExtra(
                    du_hash=du_hash,
                    frame=frame,
                    **{
                        k: _extra(k, v)
                        for k, v in annotation_deps if k in extra_values and not k.startswith("derived_")
                    },
                )
            )

        return (
            data_analysis,
            data_analysis_extra,
            annotation_analysis,
            annotation_analysis_extra
        )