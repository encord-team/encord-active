---
sidebar_position: 4
title: Filtering
---

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
Index(['identifier', 'url', 'Green Values', 'Sharpness', 'Image Singularity',
       'Blur', 'Random Values on Images', 'Red Values', 'Area', 'Aspect Ratio',
       'Brightness', 'Blue Values', 'Contrast', 'Image-level Annotation Quality',
       'description', 'object_class', 'annotator', 'frame', 'tags'], dtype='object')
```

Based on this dataframe, you can apply various filtering operations using pandas to select the data and labels that meet your criteria.

Here's an example code that builds upon the previous code and filters the dataframe to find images with very low brightness:

```python
filtered_metrics = metrics[metrics["Brightness"] < 0.2]
```

In this code, the dataframe `metrics` is filtered to include only the rows where the value in the `Brightness` column is lower than 0.2.
Customize the filter condition to suit your requirements and perform additional analysis or processing on the filtered dataframe as desired.

:::tip

To obtain the url to the data item (image) corresponding to a specific row, you can use the following utility function:

```python
from encord_active.lib.common.data_utils import url_to_file_path
from encord_active.lib.common.image_utils import key_to_data_unit

metric_row = metrics.iloc[0]
image_url = url_to_file_path(
    key_to_data_unit(metric_row["identifier"], pfs)[0].signed_url,
    project_path
)
```

:::