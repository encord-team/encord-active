from loguru import logger

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.metrics.metric import (
    AnnotationType,
    DataType,
    Metric,
    MetricType,
)
from encord_active.lib.metrics.writer import CSVMetricWriter

logger = logger.opt(colors=True)


class AnnotationTimeMetric(Metric):
    def __init__(self):
        super().__init__(
            title="Annotation Time",
            short_description="Ranks annotations by the time it took to make them.",
            long_description=r"""Ranks annotations by the time it took to make them.

If no logs are available for a particular object, it will get score 0.""",
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.ALL,
        )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        found_any = False

        for data_unit, img_pth in iterator.iterate(desc="Computing annotation times"):
            for obj in data_unit["labels"]["objects"]:
                object_label_logs = iterator.get_label_logs(object_hash=obj["objectHash"])

                if not object_label_logs:
                    writer.write(0.0, obj, description="No logs available.")
                    continue

                times = map(lambda x: x.get("time_taken", None) or 0.0, object_label_logs)
                writer.write(sum(times) / 1000, obj, description="Annotation time (seconds).")
                found_any = True

        if not found_any:
            logger.info("<blue>[Note]</blue> Couldn't get any label logs. All objects will have score 0.")
