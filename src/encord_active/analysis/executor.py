from concurrent.futures import ThreadPoolExecutor
from inspect import getfullargspec

import orjson

from encord_active.analysis.metric import MetricDependencies, OneObjectMetric
from encord_active.analysis.types import (
    ClassificationMetadata,
    EmbeddingTensor,
    MetricKey,
    MetricResult,
    ObjectMetadata,
)
from encord_active.analysis.util.torch import (
    data_unit_to_classification_meta,
    data_unit_to_object_meta,
    pillow_to_tensor,
)
from encord_active.lib.common.iterator import DatasetIterator
from encord_active.lib.project.project_file_structure import ProjectFileStructure

from .config import AnalysisConfig


class Executor:
    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config

    def execute(self, pfs: ProjectFileStructure):
        pass


class SlowDumbExecutor(Executor):
    def execute(self, pfs: ProjectFileStructure):
        project_meta = orjson.loads(pfs.project_meta.read_text())
        project_hash: str = project_meta["project_hash"]

        iterator = DatasetIterator(pfs.project_dir)
        metric_results: dict[MetricKey,  MetricDependencies] = {}

        for data_unit, img in iterator.iterate(desc="Simple Metrics"):
            # ======= STAGE 1: Compute analysis without any dependencies ====== #
            if img is None:
                raise ValueError("Image was none, so can't compute metrics")

            # === FRAME Metrtics === #
            key = MetricKey(project_hash=project_hash, label_hash=iterator.label_hash, data_unit_hash=iterator.du_hash, frame=iterator.frame, annotation=None)
            img_res: MetricDependencies = metric_results.setdefault(key, {})
            torch_image = pillow_to_tensor(img)
            objects = data_unit_to_object_meta(data_unit, partial_key=key)
            classifications = data_unit_to_classification_meta(data_unit, partial_key=key)


            # TODO come back here and fix this part
            input_args = {
                "deps": img_res,
                "image": torch_image,
                "mask": None,
                "": objects,
            }

            for metric in self.config.analysis:
                if isinstance(metric, OneObjectMetric):
                    continue
                arg_keys, *_ = getfullargspec(metric.calculate)
                args = {k: input_args.get(k) for k in arg_keys}
                res = metric.calculate(
                    image=torch_image,
                    objects=objects,
                    classifications=torch_classifications,
                    mask=None,
                    image_deps= img_res,
                    obj_deps={},
                )
                img_res[metric.ident] = res

            # === OBJECT Metrics === #
            object_results: dict[MetricKey,  dict[str, MetricResult | EmbeddingTensor]] = {}
            for obj, (obj_key, tobj) in zip(data_unit.get("labels", {}).get("objects", []), objects.items()):
                assert obj["objectHash"] == tobj.object_hash
                object_res: dict[str, MetricResult | EmbeddingTensor] = object_results.setdefault(obj_key, {})

                for metric in self.config.analysis:
                    input_args = {
                        "deps": object_res,
                        "image": torch_image,
                        "mask": tobj.mask,
                        "objs": 
                        "": torch_objects,
                    }



            # === CLASSITICATION Metrics === #
            """
            # Nothing to do for classifications in this stage atm
            torch_classifications = data_unit_to_object_meta(data_unit)
            all_torch_classifications.update(torch_classifications)

            clf_results: dict[MetricKey,  dict[str, MetricResult | EmbeddingTensor]] = {}
            for obj, tclf in zip(data_unit.get("labels", {}).get("classifications", []), torch_classifications.values()):
                assert obj["classificationHash"] == tclf.classification_hash
                key = metric_key(annotation=)
                object_res: dict[str, MetricResult | EmbeddingTensor] = {}

                for metric in self.config.analysis:
                    # object_res[metric.ident] = metric.execute(...)
                    ...
            """

        # ======= STAGE 2: Build indices and other "O(n^2)" operations ====== #
        for metric_result in metric_results.values():
            for metric in self.config.derived_embeddings:
                metric.setup_embedding_index(metric_results)
                metric_result[metric.ident] = metric.execute(metric_result)

        # ======= STAGE 3: Derive final metrics based on indices ====== #
        for metric_result in metric_results.values():
            for metric in self.config.derived_metrics:
                metric_result[metric.ident] = metric.execute(metric_result)

    def analyse_image(self, image, data_unit: dict, analysis: BaseEvaluation) -> AnalysisResult:
        ...

    def derive_embeddings(self, metric_db: None) -> None:
        pass

    def derive_metric(self, analysis_result: AnalysisResult) -> AnalysisResult:
        ...
