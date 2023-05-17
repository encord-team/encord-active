---
sidebar_position: 5
---

# Writing Custom Quality Metric

**Guide on how to write your own Custom Quality Metrics**
:::tip
If you are more comfortable using notebooks, we have a
[Google Colab notebook](https://colab.research.google.com/drive/1tAqGGSY0sZfwec2Vp4ThvgLKIefy3-4b?usp=sharing)
for writing your own metric.
:::

Create a new python file in the `<your_custom_metrics_folder>` directory and use the template provided in
`libs/encord_active/metrics/example.py`. The subdirectory within `libs/encord_active/metrics` is dictated by what
information the metric employs:

- **Geometric:** Metrics related to the geometric properties of annotations.
  This includes size, shape, location etc.
- **Semantic:** Metrics based on the _contents_ of some image, video or annotation.
  This includes embedding distances, image uncertainties etc.
- **Heuristic:** Any other metrics. For example, brightness, sharpness, object counts, etc.

You can use the following template to get started with writing your own metric.
Your implementation should call `writer.write(<object_score>, <object>)` for every object in the iterator **OR** use `writer.write(<frame_score>)` for every data unit in the iterator.

```python
from loguru import logger

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.metrics.metric import AnnotationType, DataType, Metric, MetricType
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
            doc_url='link/to/documentation', # This is optional, if a link is given, it can be accessed from the app
            metric_type=MetricType.HEURISTIC,
            data_type=DataType.IMAGE,
            annotation_type=[AnnotationType.OBJECT.BOUNDING_BOX, AnnotationType.OBJECT.ROTATABLE_BOUNDING_BOX, AnnotationType.OBJECT.POLYGON],
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
```

Before running your own custom metric, make sure that you have `project_meta.yaml` file in the project data folder.

To run your metric from the root directory, use:

```shell
# within venv
python your_metric_file.py /path/to/your/data/dir
```

You can check the generated metric file in your `<data root dir>/metrics`, its name should be `<hash>_example_title.csv` .
When you have run your metric, you can visualize your results by running:

```shell
# within venv
encord-active visualize
```

Now, you can improve your data/labels/model by choosing your own custom metric in the app.
