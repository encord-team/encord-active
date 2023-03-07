---
sidebar_position: 2
title: Running Quality Metrics
---

There are a couple of ways to compute quality metrics via code.

:::info

This is equivalent to running the [`metric run` command][metrics-run-cli-docs] from the CLI.

:::

## Running Builtin Metrics

To run all metrics available via Encord Active, you can to the following:

```python
from pathlib import Path

from encord_active.lib.metrics.execute import run_metrics

project_path = Path("/path/to/your/project/root")
run_metrics(data_dir=project_path, use_cache_only=True)
```

The `run_metrics` function also allows you to filter which metrics to run by providing a filter function:

```python
options = dict(
    data_dir=project_path,
    use_cache_only=True,
)
run_metrics(
    filter_func=lambda m: m().metadata.title == "<query>",
    **options,
)
```

### Running Data or Label Metrics only

There is a utility function you can use to run targeted subsets of metrics:

```
from encord_active.lib.metrics.execute import (
    run_metrics_by_embedding_type,
)
from encord_active.lib.metrics.metric import EmbeddingType

run_metrics_by_embedding_type(EmbeddingType.IMAGE, **options)
run_metrics_by_embedding_type(EmbeddingType.OBJECT, **options)
run_metrics_by_embedding_type(EmbeddingType.CLASSIFICATION, **options)
```

[metrics-run-cli-docs]: ../cli/metric-management#metric-run

## Running Custom Metrics

If you have already [written a custom metric][write-custom-metric] - let's call it `SuperMetric`, then you can execute it on a projects with the following code:

```python
from pathlib import Path

from encord_active.lib.metrics.execute import execute_metrics

from super_metric import SuperMetric

project_path = Path("/path/to/your/project")
execute_metrics([SuperMetric()], data_dir=project_path)
```

:::tip

The CLI also has a way to [registering metrics][cli-metric-add] to the project to easily execute them afterwards.

:::

[write-custom-metric]: ../metrics/write-your-own
[cli-metric-add]: ../cli/metric-management#metric-add
