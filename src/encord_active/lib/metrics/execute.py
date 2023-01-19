import inspect
import json
import logging
import os
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Type, Union

from encord.project_ontology.object_type import ObjectShape
from loguru import logger

from encord_active.lib.common.iterator import DatasetIterator, Iterator
from encord_active.lib.common.utils import fetch_project_info
from encord_active.lib.common.writer import StatisticsObserver
from encord_active.lib.metrics.metric import (
    AnnotationType,
    DataType,
    Metric,
    MetricType,
)
from encord_active.lib.metrics.writer import CSVMetricWriter


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
                if issubclass(cls[1], Metric) and cls[1] != Metric and filter_func(cls[1]):
                    metrics.append((f"{base_module_name}{module_name}.{file.split('.')[0]}", f"{cls[0]}"))

    return metrics


def run_all_heuristic_metrics():
    run_metrics(filter_func=lambda x: x.METRIC_TYPE == MetricType.HEURISTIC)


def run_all_image_metrics():
    run_metrics(filter_func=lambda x: x.DATA_TYPE == DataType.IMAGE)


def run_all_polygon_metrics():
    run_metrics(filter_func=lambda x: x.ANNOTATION_TYPE in [AnnotationType.OBJECT.POLYGON, AnnotationType.ALL])


def run_all_prediction_metrics(**kwargs):
    # Return all metrics that apply to objects.
    def filter(m: Metric):
        at = m.ANNOTATION_TYPE
        if isinstance(at, list):
            for t in at:
                if isinstance(t, ObjectShape):
                    return True
            return False
        else:
            return isinstance(at, ObjectShape)

    run_metrics(filter_func=filter, **kwargs)


def run_metrics(filter_func: Callable = lambda x: True, **kwargs):
    metrics = list(map(load_metric, get_metrics(filter_func=filter_func)))
    execute_metrics(metrics, **kwargs)


def load_metric(module_classname_pair: Tuple[str, str]) -> Metric:
    return import_module(module_classname_pair[0]).__getattribute__(module_classname_pair[1])()


def __get_value(o):
    if isinstance(o, (float, int, str)):
        return o
    if isinstance(o, Enum):
        return __get_value(o.value)
    if isinstance(o, (list, tuple)):
        return [__get_value(v) for v in o]
    return None


def __get_object_attributes(obj: Any):
    metric_properties = {v.lower(): __get_value(getattr(obj, v)) for v in dir(obj)}
    metric_properties = {k: v for k, v in metric_properties.items() if (v is not None or k == "annotation_type")}
    return metric_properties


logger = logger.opt(colors=True)


@logger.catch()
def execute_metrics(
    metrics: List[Metric],
    data_dir: Path,
    iterator_cls: Type[Iterator] = DatasetIterator,
    use_cache_only: bool = False,
    **kwargs,
):
    project = None if use_cache_only else fetch_project_info(data_dir)
    iterator = iterator_cls(data_dir, project=project, **kwargs)
    cache_dir = iterator.update_cache_dir(data_dir)

    for metric in metrics:
        logger.info(f"Running Metric <blue>{metric.TITLE.title()}</blue>")
        unique_metric_name = metric.get_unique_name()

        stats = StatisticsObserver()
        with CSVMetricWriter(cache_dir, iterator, prefix=unique_metric_name) as writer:
            writer.attach(stats)

            try:
                metric.execute(iterator, writer)
            except Exception as e:
                logging.critical(e, exc_info=True)

        # Store meta-data about the scores.
        meta_file = (cache_dir / "metrics" / f"{unique_metric_name}.meta.json").expanduser()

        with meta_file.open("w") as f:
            json.dump(
                {
                    **__get_object_attributes(metric),
                    **__get_object_attributes(stats),
                },
                f,
                indent=2,
            )
