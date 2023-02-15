import inspect
import logging
import os
from importlib import import_module
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Type, Union

import cv2

# NOTE: this statement enusres faiss is imported before torch.
# removing it will trigger segmentation faults in certain metrics
import faiss  # pylint: disable=unused-import
from loguru import logger

from encord_active.lib.common.iterator import DatasetIterator, Iterator
from encord_active.lib.common.utils import fetch_project_info
from encord_active.lib.common.writer import StatisticsObserver
from encord_active.lib.labels.object import ObjectShape
from encord_active.lib.metrics.metric import (
    AnnotationType,
    DataType,
    EmbeddingType,
    Metric,
    MetricType,
    SimpleMetric,
    StatsMetadata,
)
from encord_active.lib.metrics.utils import get_embedding_type
from encord_active.lib.metrics.writer import CSVMetricWriter

logger = logger.opt(colors=True)


def get_metrics(module: Optional[Union[str, list[str]]] = None, filter_func=lambda x: True):
    if module is None:
        module = ["geometric", "heuristic", "semantic"]
    elif isinstance(module, str):
        module = [module]

    return [i for m in module for i in get_module_metrics(m, filter_func)]


def get_module_metrics(module_name: str, filter_func: Callable) -> List:
    if __name__ == "__main__":
        base_module_name = ""
    else:
        base_module_name = __name__[: __name__.rindex(".")] + "."  # Remove "run_all"

    metrics = []
    path = os.path.join(os.path.dirname(__file__), *module_name.split("."))
    for file in os.listdir(path):
        if file.endswith(".py") and not file.startswith("_") and not file.split(".")[0].endswith("_"):
            logging.debug("Importing %s", file)
            clsmembers = inspect.getmembers(
                import_module(f"{base_module_name}{module_name}.{file.split('.')[0]}"), inspect.isclass
            )
            for cls in clsmembers:
                if (
                    (issubclass(cls[1], SimpleMetric) and cls[1] != SimpleMetric)
                    or (issubclass(cls[1], Metric) and cls[1] != Metric)
                ) and filter_func(cls[1]):
                    metrics.append((f"{base_module_name}{module_name}.{file.split('.')[0]}", f"{cls[0]}"))

    return metrics


def is_metric_matching_embedding(embedding_type: EmbeddingType, metric: Metric):
    if metric.metadata.annotation_type is None or isinstance(metric.metadata.annotation_type, list):
        return embedding_type == get_embedding_type(metric.metadata.title, metric.metadata.annotation_type)
    else:
        return embedding_type == get_embedding_type(metric.metadata.title, [metric.metadata.annotation_type])


def get_metrics_by_embedding_type(embedding_type: EmbeddingType):
    metrics = map(load_metric, get_metrics())
    return [metric for metric in metrics if is_metric_matching_embedding(embedding_type, metric)]


def run_metrics_by_embedding_type(embedding_type: EmbeddingType, **kwargs):
    metrics = get_metrics_by_embedding_type(embedding_type)
    execute_metrics(metrics, **kwargs)


def run_all_heuristic_metrics():
    run_metrics(filter_func=lambda x: x.metadata.metric_type == MetricType.HEURISTIC)


def run_all_image_metrics():
    run_metrics(filter_func=lambda x: x.metadata.data_type == DataType.IMAGE)


def run_all_polygon_metrics():
    run_metrics(filter_func=lambda x: x.metadata.annotation_type in [AnnotationType.OBJECT.POLYGON, AnnotationType.ALL])


def run_all_prediction_metrics(**kwargs):
    # Return all metrics that apply to objects.
    def filter(m: Type[Metric]):
        # TODO: find a better way to resolve this, only leaf children of `Metric` don't expect arguments
        at = m().metadata.annotation_type  # type: ignore
        if isinstance(at, list):
            for t in at:
                if isinstance(t, ObjectShape):
                    return True
            return False
        else:
            return isinstance(at, ObjectShape)

    run_metrics(filter_func=filter, **kwargs)


def run_metrics(filter_func: Callable[[Metric], bool] = lambda x: True, **kwargs):
    metrics = list(map(load_metric, get_metrics(filter_func=filter_func)))
    execute_metrics(metrics, **kwargs)


def load_metric(module_classname_pair: Tuple[str, str]) -> Metric:
    return import_module(module_classname_pair[0]).__getattribute__(module_classname_pair[1])()


def _write_meta_file(cache_dir: Path, metric: Union[Metric, SimpleMetric], stats: StatisticsObserver):
    meta_file = (cache_dir / "metrics" / f"{metric.metadata.get_unique_name()}.meta.json").expanduser()
    metric.metadata.stats = StatsMetadata.from_stats_observer(stats)

    with meta_file.open("w") as f:
        f.write(metric.metadata.json())


def _execute_metrics(cache_dir: Path, iterator: Iterator, metrics: list[Metric]):
    for metric in metrics:
        logger.info(f"Running metric <blue>{metric.metadata.title}</blue>")
        unique_metric_name = metric.metadata.get_unique_name()

        stats = StatisticsObserver()
        with CSVMetricWriter(cache_dir, iterator, prefix=unique_metric_name) as writer:
            writer.attach(stats)

            try:
                metric.execute(iterator, writer)
            except Exception as e:
                logging.critical(e, exc_info=True)

        _write_meta_file(cache_dir, metric, stats)


def _execute_simple_metrics(cache_dir: Path, iterator: Iterator, metrics: list[SimpleMetric]):
    if len(metrics) == 0:
        return
    logger.info(f"Running metrics <blue>{', '.join(metric.metadata.title for metric in metrics)}</blue>")
    csv_writers = [CSVMetricWriter(cache_dir, iterator, prefix=metric.metadata.get_unique_name()) for metric in metrics]
    stats_observers = [StatisticsObserver() for _ in metrics]
    for csv_w, stats in zip(csv_writers, stats_observers):
        csv_w.attach(stats)
    for data_unit, img_pth in iterator.iterate():
        if img_pth is None:
            continue
        try:
            image = cv2.imread(img_pth.as_posix())
            for metric, csv_w in zip(metrics, csv_writers):
                rank = metric.execute(image)
                csv_w.write(rank)
        except Exception as e:
            logging.critical(e, exc_info=True)
    for metric, stats in zip(metrics, stats_observers):
        _write_meta_file(cache_dir, metric, stats)


@logger.catch()
def execute_metrics(
    selected_metrics: Sequence[Union[Metric, SimpleMetric]],
    data_dir: Path,
    iterator_cls: Type[Iterator] = DatasetIterator,
    use_cache_only: bool = False,
    **kwargs,
):
    project = None if use_cache_only else fetch_project_info(data_dir)
    iterator = iterator_cls(data_dir, project=project, **kwargs)
    cache_dir = iterator.update_cache_dir(data_dir)

    simple_metrics = [m for m in selected_metrics if isinstance(m, SimpleMetric)]
    metrics = [m for m in selected_metrics if isinstance(m, Metric)]

    if metrics:
        _execute_metrics(cache_dir, iterator, metrics)
    if simple_metrics:
        _execute_simple_metrics(cache_dir, iterator, simple_metrics)
