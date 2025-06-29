---
title: "Filtering"
slug: "active-sdk-filtering"
hidden: false
metadata: 
  title: "Filtering"
  description: "Master data filtering in Encord Active: Optimize insights, remove noise, prioritize tasks. Use filters for focused analysis."
  image: 
    0: "https://files.readme.io/1556a4c-image_16.png"
createdAt: "2023-07-14T16:05:36.402Z"
updatedAt: "2023-08-09T16:17:22.090Z"
category: "65a71bbfea7a3f005192d1a7"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/local_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

**Learn how to apply filters to your data and labels**

To filter your data and labels based on metrics, you can utilize the `MergedMetrics` dataframe.

```python
from pathlib import Path

import pandas as pd
from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.project.project_file_structure import ProjectFileStructure

project_path = Path("/path/to/your/project/root")
pfs = ProjectFileStructure(project_path)
with DBConnection(pfs) as conn:
    metrics: pd.DataFrame = MergedMetrics(conn).all().reset_index()
```

The resulting dataframe will contain all your data and labels, along with the associated metrics computed on them.

Here's an example of the column names that you might see in the dataframe:

```
Index(['identifier', 'url', 'Green Value', 'Sharpness', 'Uniqueness',
       'Randomize Images', 'Red Value', 'Area', 'Aspect Ratio',
       'Brightness', 'Blue Value', 'Contrast', 'Classification Quality',
       'description', 'object_class', 'annotator', 'frame', 'tags'], dtype='object')
```

Based on this dataframe, you can apply various filtering operations using pandas to select the data and labels that meet your criteria.

Here's an example code that builds upon the previous code and filters the dataframe to find images with very low brightness:

```python
filtered_metrics = metrics[metrics["Brightness"] < 0.2]
```

In this code, the dataframe `metrics` is filtered to include only the rows where the value in the `Brightness` column is lower than 0.2. Customize the filter condition to suit your requirements and perform additional analysis or processing on the filtered dataframe as desired.

If you want to obtain the url to the data item (image) corresponding to a specific row, you can use the following utility function:

```python
from encord_active.lib.common.data_utils import url_to_file_path
from encord_active.lib.common.image_utils import key_to_data_unit

metric_row = metrics.iloc[0]
image_url = url_to_file_path(
    key_to_data_unit(metric_row["identifier"], pfs)[0].signed_url,
    project_path
)
```