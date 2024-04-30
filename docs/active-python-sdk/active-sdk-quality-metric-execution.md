---
title: "Quality metric execution"
slug: "active-sdk-quality-metric-execution"
hidden: false
metadata: 
  title: "Quality metric execution"
  description: "Compute quality metrics using Python code for CLI-equivalent functionality. Learn built-in & custom metric execution. | Encord"
  image: 
    0: "https://files.readme.io/d548c25-image_16.png"
createdAt: "2023-07-14T16:05:36.319Z"
updatedAt: "2023-08-11T13:43:54.172Z"
category: "65a71bbfea7a3f005192d1a7"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/local_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

**Learn how to execute built-in and custom quality metrics**

There are a couple of ways to compute quality metrics via code.

Quality metrics calculation using Python code provides the same functionality as executing the [`metric run`](https://docs.encord.com/docs/active-cli#run) command in the CLI.

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

## Running custom metrics

If you have already [written a custom metric](https://docs.encord.com/docs/active-write-custom-quality-metrics), let's call it `SuperMetric`, then you can execute it on a project of your choosing with the following code:

```python
from pathlib import Path
from encord_active.lib.metrics.execute import execute_metrics
from super_metric import SuperMetric

project_path = Path("/path/to/your/project")
execute_metrics([SuperMetric()], data_dir=project_path)
```

> 👍 Tip
> The CLI allows you to [register metrics](https://docs.encord.com/docs/active-cli#add) to the project to easily execute them afterwards.