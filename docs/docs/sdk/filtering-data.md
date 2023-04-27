---
sidebar_position: 4
title: Filtering Data
---

To filter your data and labels based on metrics, you use the `MergedMetrics` dataframe.

```python
from pathlib import Path

import pandas as pd
from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.project.project_file_structure import ProjectFileStructure

project_path = Path("/path/to/your/project/root")
DBConnection.set_project_file_structure(ProjectFileStructure(project_path))

metrics: pd.DataFrame = MergedMetrics().all().reset_index()

print(metrics.columns)
```

This dataframe will have all your data and labels listed with all the associated metrics computed on them:

```
Index(['identifier', 'url', 'Green Values', 'Sharpness', 'Image Singularity',
       'Blur', 'Random Values on Images', 'Red Values', 'Area', 'Aspect Ratio',
       'Brightness', 'Blue Values', 'Contrast',
       'Image-level Annotation Quality', 'description', 'object_class',
       'annotator', 'frame', 'tags'],
      dtype='object')
```

Based on this data frame, you can do any filter you might like using pandas.

To get the path to the data item (image) that a specific row corresponds to, you can use this utility function:

```python
from encord_active.lib.project import ProjectFileStructure
from encord_active.lib.common.image_utils import key_to_data_unit

fs = ProjectFileStructure(project_path)

metric_row = metrics.iloc[0]
image_url = key_to_data_unit(metric_row["identifier"], fs).path

print(image_url)
```
