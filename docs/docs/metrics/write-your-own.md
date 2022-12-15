---
sidebar_position: 5
---

# Write Your Own Metric

Create a new python file in the `<your_custom_metrics_folder>` directory and use the template provided in
`libs/encord_active/metrics/example.py`. The subdirectory within `libs/encord_active/metrics` is dictated by what
information the metric employs:

- **Geometric:** Metrics related to the geometric properties of annotations.
  This includes size, shape, location etc.
- **Semantic:** Metrics based on the _contents_ of some image, video or annotation.
  This includes embedding distances, image uncertainties etc.
- **Heuristic:** Any other metrics. For example, brightness, sharpness, object counts, etc.

You can use the following template to get started with writing your own metric.
Your implementation should call `writer.write(<object_score>, <object>)` for every object in the iterator or use `writer.write(<frame_score>)` for every data unit in the iterator.

```python
from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.metric import AnnotationType, DataType, MetricType, Metric
from encord_active.lib.common.writer import CSVMetricWriter

class ExampleMetric(Metric):
    TITLE = "Example Title"
    METRIC_TYPE = MetricType.HEURISTIC
    DATA_TYPE = DataType.IMAGE
    ANNOTATION_TYPE = [AnnotationType.OBJECT.BOUNDING_BOX, AnnotationType.OBJECT.POLYGON]
    SHORT_DESCRIPTION = "A short description of your metric."
    LONG_DESCRIPTION = "A longer and more detailed description. " \
                       "I can use Markdown to _format_ the text."

    def test(self, iterator: Iterator, writer: CSVMetricWriter):
        valid_annotation_types = {annotation_type.value for annotation_type in self.ANNOTATION_TYPE}

        for data_unit, img_pth in iterator.iterate(desc="Progress bar description"):
            # Frame level score (data quality)
            writer.write(1337, description="Your description for the frame [can be omitted]")
            for obj in data_unit["labels"].get("objects", []):
                # Label (object/classification) level score (label / model prediction quality)
                if not obj["shape"] in valid_annotation_types:
                    continue

                # This is where you do the actual inference.
                # Some convenient properties associated with the current data.
                # ``iterator.label_hash`` the label hash of the current data unit
                # ``iterator.du_hash`` the data unit hash of
                # ``iterator.frame`` the frame of the current data unit hash of
                # ``iterator.num_frame`` the total number of frames in the label row.

                # Do your thing (inference)
                # ...
                # Then
                writer.write(42, labels=obj, description="Your description of the score [can be omitted]")

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from encord_active.lib.common.tester import perform_test

    path = sys.argv[1]
    perform_test(ExampleMetric(), data_dir=Path(path))
```

Before running your own custom metric, make sure that you have `project_meta.yaml` file in the project data folder.

To run your metric from the root directory, use:

```shell
(ea-venv)$ python your_metric_file.py /path/to/your/data/dir
```

You can check the generated metric file in your `<data root dir>/metrics`, its name should be `<hash>_example_title.csv` .
When you have run your metric, you can visualise your results by running:

```shell
(ea-venv)$ encord-active visualise
```

Now, you can improve your data/labels/model by choosing your own custom metric in the app.
