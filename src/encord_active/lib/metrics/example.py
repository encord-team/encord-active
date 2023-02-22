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


class ExampleMetric(Metric):
    def __init__(self):
        super().__init__(
            title="Example Title",
            short_description="Assigns same value and description to all objects.",
            long_description=r"""For long descriptions, you can use Markdown to _format_ the text.

For example, you can make a
[hyperlink](https://memegenerator.net/instance/74454868/europe-its-the-final-markdown)
to the awesome paper that proposed the method.

Or use math to better explain such method:
$$h_{\lambda}(x) = \frac{1}{x^\intercal x}$$
""",
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=[
                AnnotationType.OBJECT.BOUNDING_BOX,
                AnnotationType.OBJECT.ROTATABLE_BOUNDING_BOX,
                AnnotationType.OBJECT.POLYGON,
            ],
        )

    def execute(self, iterator: Iterator, writer: CSVMetricWriter):
        valid_annotation_types = {annotation_type.value for annotation_type in self.metadata.annotation_type}

        logger.info("My custom logging")

        # Preprocessing happens here.
        # You can build/load databases of embeddings, compute statistics, etc

        for data_unit, img_pth in iterator.iterate(desc="Progress bar description"):
            # Frame level score (data quality)
            writer.write(1337, description="Your description of the score [can be omitted]")

            for obj in data_unit["labels"].get("objects", []):
                # Label (object/classification) level score (label/model prediction quality)
                if not obj["shape"] in valid_annotation_types:
                    continue

                # This is where you do the actual inference.
                # Some convenient properties associated with the current data.
                # ``iterator.label_hash`` the label hash of the current data unit
                # ``iterator.du_hash`` the hash of the current data unit
                # ``iterator.frame`` the frame of the current data unit
                # ``iterator.num_frames`` the total number of frames in the label row.

                # Do your thing (inference)
                # Then
                writer.write(42, labels=obj, description="Your description of the score [can be omitted]")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from encord_active.lib.metrics.execute import execute_metrics

    path = sys.argv[1]
    execute_metrics([ExampleMetric()], data_dir=Path(path))
