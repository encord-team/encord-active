import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, List, Type, Union

from loguru import logger

from encord_active.lib.common.iterator import DatasetIterator, Iterator
from encord_active.lib.common.metric import Metric
from encord_active.lib.common.utils import fetch_project_info
from encord_active.lib.common.writer import CSVMetricWriter, StatisticsObserver

logger = logger.opt(colors=True)


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


@logger.catch()
def perform_test(
    metrics: Union[Metric, List[Metric]],
    data_dir: Path,
    iterator_cls: Type[Iterator] = DatasetIterator,
    use_cache_only: bool = False,
    **kwargs,
):
    all_tests: List[Metric] = metrics if isinstance(metrics, list) else [metrics]

    project = None if use_cache_only else fetch_project_info(data_dir)
    iterator = iterator_cls(data_dir, project=project, **kwargs)
    cache_dir = iterator.update_cache_dir(data_dir)

    for metric in all_tests:
        logger.info(f"Running Metric <blue>{metric.TITLE.title()}</blue>")
        unique_metric_name = metric.get_unique_name()

        stats = StatisticsObserver()
        with CSVMetricWriter(cache_dir, iterator, prefix=unique_metric_name) as writer:
            writer.attach(stats)

            try:
                metric.test(iterator, writer)
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
