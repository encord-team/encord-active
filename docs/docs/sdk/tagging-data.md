---
sidebar_position: 3
title: Tagging
---

Tagging is done in two steps.

1. Creating the tag.
2. Tagging the data or labels.

Below we describe how to do this with code.

## Creating Tags

To create new tags, you follow this template:

```python
from pathlib import Path

from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.tags import Tag, Tags, TagScope

project_path = Path("/path/to/your/project/root")
DBConnection.set_project_path(project_path)

# Getting tags
tags = Tags()
existing_tags = tags.all()

# Creating tags
data_tag = Tag("Data tag name", TagScope.DATA)
label_tag = Tag("Label tag name", TagScope.DATA)

tags.create_tag(data_tag)
tags.create_tag(label_tag)
# or
tags.create_multiple([data_tag, label_tag])
```

Note that there are two different scopes to which you can tag your data.
One it the `TagScope.DATA` score which applies tags to the images them self.
The second one it the `TagScope.LABEL`, which applies to individual labels.
As such the `TagScope.LABEL` can be present multiple times in the same image if there are multiple labels with that tag.

## Tagging Data

Tagging data is based on unique identifiers of the data you are tagging.
For data tagging, identifier is a composition of `label_hash`, `data_hash`, and `frame`.

To iterate over the data in a project and add tags is to follow this structure.

```python
from pathlib import Path

import pandas as pd

from encord_active.lib.common.iterator import DatasetIterator
from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.merged_metrics import MergedMetrics

DBConnection.set_project_path(project_path)
metrics = MergedMetrics()

for data_unit, image_path in DatasetIterator(project_path).iterate():
    identifier = f"{iterator.label_hash}_{iterator.data_hash}_{iterator.frame:05d}"
    tags = metrics.get_row(identifier).tags
    tags.append(data_tag)  # Only use TagScope.DATA tags here.
    metrics.update_tags(identifier, tags)
```

## Tagging Labels

Tagging labels is very similar to tagging data.
You will need to append the `classificationHash` or the `objectHash` for the label you are tagging.

To iterate over the labels in a project and add tags follow this structure.

```python
from pathlib import Path

import pandas as pd

from encord_active.lib.common.iterator import DatasetIterator
from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.merged_metrics import MergedMetrics

DBConnection.set_project_path(project_path)
metrics = MergedMetrics()

for data_unit, image_path in DatasetIterator(project_path).iterate():
    # For bounding boxes and polygons
    for obj in data_unit.get("labels", {}).get("objects", []):
        obj_hash = obj["objectHash"]
        identifier = f"{iterator.label_hash}_{iterator.data_hash}_{iterator.frame:05d}_{obj_hash}"

        tags = metrics.get_row(identifier).tags
        tags.append(label_tag)  # Only use TagScope.LABEL tags here.
        metrics.update_tags(identifier, tags)

    # For frame-level classifications
    for obj in data_unit.get("labels", {}).get("classifications", []):
        clf_hash = obj["classificationHash"]
        identifier = f"{iterator.label_hash}_{iterator.data_hash}_{iterator.frame:05d}_{clf_hash}"

        tags = metrics.get_row(identifier).tags
        tags.append(label_tag)  # Only use TagScope.LABEL tags here.
        metrics.update_tags(identifier, tags)
```
