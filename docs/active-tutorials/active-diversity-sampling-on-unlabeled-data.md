---
title: "Running Diversity Based Acquisition Function on Unlabeled Data"
slug: "active-diversity-sampling-on-unlabeled-data"
hidden: false
metadata: 
  title: "Running Diversity Based Acquisition Function on Unlabeled Data"
  description: "Tutorial: Run Clustering-based Diversity Sampling to Rank Images. Enhance acquisition function. Boost image ranking with diversity sampling."
  image: 
    0: "https://files.readme.io/3889a75-image_16.png"
createdAt: "2023-07-11T16:27:41.992Z"
updatedAt: "2023-08-09T12:34:28.153Z"
category: "65a71bbfea7a3f005192d1a7"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/local_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

In this tutorial, you will see how to run a clustering-based diversity sampling acquisition function to rank the images.


> ℹ️ Note
> This tutorial assumes that you have [installed](https://docs.encord.com/docs/active-oss-install) `encord-active`.

## 1. Inference on the unlabeled data

First, terminate any running Encord Active app before running any code. Then, get the root directory path of the encord-active project and copy it to `data_dir` variable below. You will use [Image Diversity](https://docs.encord.com/docs/active-data-quality-metrics#image-diversity) metric to rank the images. Image Diversity metric simply clusters the dataset according to the number of classes in the <<glossary:Ontology>> and selects equal number of samples from each cluster. Your project may consist of both labeled and unlabeled examples and you may want to run this acquisition function only on the unlabeled data; therefore, you will set `skip_labeled_data` to `True`.

```python
from pathlib import Path
from encord_active.lib.metrics.semantic.image_diversity import ImageDiversity
from encord_active.lib.metrics.execute import execute_metrics

data_dir = Path("/path/to/encord-active/project")
acquisition_func = ImageDiversity()
execute_metrics([acquisition_func], data_dir=data_dir, use_cache_only=True, skip_labeled_data=True)
```

## 2. Refresh metric files

After executing the acquisition function. It should output two new files in the metrics folder of the root project folder. We need to update the metric information in the project to reflect the changes in the UI:

```python
from encord_active.lib.metrics.io import get_metric_metadata
from encord_active.lib.metrics.metadata import fetch_metrics_meta, update_metrics_meta
from encord_active.lib.project.project_file_structure import ProjectFileStructure

project_fs = ProjectFileStructure(data_dir)
metrics_meta = fetch_metrics_meta(project_fs)
metrics_meta[acquisition_func.metadata.title]= get_metric_metadata(acquisition_func)
update_metrics_meta(project_fs, metrics_meta)
project_fs.db.unlink(missing_ok=True)
```

Now, open the encord-active app using the following CLI command in the project or its root folder:

```shell
encord-active start
```

Go to Data Quality -> Explorer, and choose Image Diversity from the metric drop-down menu. You will see the examples sorted according to the image diversity function. From now on, you can select the first N samples and:

1. create a new project to label based off these samples.
2. export the selected samples using Actions tab and use them in you own label annotation pipeline.