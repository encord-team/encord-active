---
sidebar_position: 3
title: Tagging
---

## Scopes

Note that there are two different scopes to which you can apply tags.
One is the `TagScope.DATA` scope which applies tags to the images.
The second one is the `TagScope.LABEL`, which applies to individual labels.
The `TagScope.LABEL` can be present multiple times in the same image if there are multiple labels with that tag.

## Tagging Data

Tagging data is based on unique identifiers of the data you are tagging.
For data tagging, the identifier is a composition of `label_hash`, `data_hash`, and `frame`.

To iterate over the data in a project and add tags is to follow this structure.

```python
from pathlib import Path

from encord_active.lib.common.iterator import DatasetIterator
from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.project.project_file_structure import ProjectFileStructure

project_path = Path("/path/to/your/project/root")
with DBConnection(ProjectFileStructure(project_path)) as conn:
    metrics = MergedMetrics(conn).all()

iterator = DatasetIterator(project_path)
for data_unit, image in iterator.iterate():
    identifier = f"{iterator.label_hash}_{iterator.du_hash}_{iterator.frame:05d}"
    tags = metrics.get_row(identifier).tags
    tags.append(data_tag)  # Only use TagScope.DATA tags here.
    metrics.update_tags(identifier, tags)
```

## Tagging Labels

Tagging labels is very similar to tagging data.
You will need to append the `classificationHash` or the `objectHash` for the label you are tagging to the identifier - as done below.

```python
from pathlib import Path

from encord_active.lib.common.iterator import DatasetIterator
from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.project.project_file_structure import ProjectFileStructure

project_path = Path("/path/to/your/project/root")
with DBConnection(ProjectFileStructure(project_path)) as conn:
    metrics = MergedMetrics(conn).all()

iterator = DatasetIterator(project_path)
for data_unit, image in iterator.iterate():
    # For bounding boxes and polygons
    for obj in data_unit.get("labels", {}).get("objects", []):
        obj_hash = obj["objectHash"]
        identifier = f"{iterator.label_hash}_{iterator.du_hash}_{iterator.frame:05d}_{obj_hash}"

        tags = metrics.get_row(identifier).tags
        tags.append(label_tag)  # Only use TagScope.LABEL tags here.
        metrics.update_tags(identifier, tags)

    # For frame-level classifications
    for obj in data_unit.get("labels", {}).get("classifications", []):
        clf_hash = obj["classificationHash"]
        identifier = f"{iterator.label_hash}_{iterator.du_hash}_{iterator.frame:05d}_{clf_hash}"

        tags = metrics.get_row(identifier).tags
        tags.append(label_tag)  # Only use TagScope.LABEL tags here.
        metrics.update_tags(identifier, tags)
```
