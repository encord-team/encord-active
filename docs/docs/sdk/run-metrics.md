---
sidebar_position: 2
---

# Quality metric execution

**Learn how to execute built-in and custom quality metrics**

Quality metrics calculation using Python code provides the same functionality as executing the [`metric run`][cli-metric-run] command in the CLI.

## Running built-in metrics

To run all metrics available by default in Encord Active, you can use the following code snippet:

```python
from pathlib import Path
from encord_active.lib.metrics.execute import run_metrics

project_path = Path("/path/to/your/project/root")
run_metrics(data_dir=project_path, use_cache_only=True)
```

The `run_metrics` function also allows you to filter which metrics to run by providing a filter function:

```python
options = dict(data_dir=project_path, use_cache_only=True)
run_metrics(filter_func=lambda m: m().metadata.title == "<query>", **options)
```

### Compute only data or label metrics

The `run_metrics_by_embedding_type` utility function allows you to run predefined subsets of metrics:

```python
from encord_active.lib.metrics.execute import run_metrics_by_embedding_type
from encord_active.lib.metrics.types import EmbeddingType

run_metrics_by_embedding_type(EmbeddingType.IMAGE, **options)
run_metrics_by_embedding_type(EmbeddingType.OBJECT, **options)
run_metrics_by_embedding_type(EmbeddingType.CLASSIFICATION, **options)
```

## Running Custom Metrics

If you have already [written a custom metric][write-custom-metric], let's call it `SuperMetric`, then you can execute it on a project of your choosing with the following code:

```python
from pathlib import Path
from encord_active.lib.metrics.execute import execute_metrics
from super_metric import SuperMetric

project_path = Path("/path/to/your/project")
execute_metrics([SuperMetric()], data_dir=project_path)
```

:::tip

The CLI allows to [register metrics][cli-metric-add] to the project to easily execute them afterwards.

:::

[write-custom-metric]: ../metrics/write-your-own
[cli-metric-add]: ../cli#metric-add
[cli-metric-run]: ../cli#metric-run
