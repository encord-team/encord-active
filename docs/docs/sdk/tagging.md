---
sidebar_position: 3
title: Tagging
---

**Learn the meaning of scopes for tags and how to tag your data**

Tagging data using Python code provides the same functionality as using the [_Tag_ feature][tagging-guide] in the application.

## Scopes

When applying tags in your project, it's important to understand the two different scopes to which tags can be applied: `TagScope.DATA` and `TagScope.LABEL`.

* The `TagScope.DATA` scope allows you to apply tags to the images themselves.
  Tags applied in this scope describe characteristics or attributes that apply to the entire image as a whole.
  For example, you can tag an image with labels such as "outdoor," "sunset," or "high-resolution" to describe its overall properties.

* The `TagScope.LABEL` scope allows you to apply tags to individual labels within an image.
  This means that you can assign specific tags to each label, capturing unique attributes or properties associated with that particular label.
  For instance, you can tag a label as "cat" or "dog" to identify the specific objects present in an image.

It's worth noting that the `TagScope.LABEL` can be present multiple times in the same image if there are multiple labels with the same tag. This provides flexibility in tagging multiple labels that share a common characteristic.

By utilizing these scopes, you can apply tags at different levels of granularity, ensuring precise and specific tagging of data in your AI project.

## Tagging images

Each image is assigned unique identifiers based on the `label_hash`, `data_hash`, and `frame`.
This enables you to easily reference and tag specific images as needed.

To effectively tag your images, follow the structure below:

```python
from pathlib import Path

from encord_active.lib.common.iterator import DatasetIterator
from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.db.tags import Tag, TagScope
from encord_active.lib.project.project_file_structure import ProjectFileStructure

project_path = Path("/path/to/your/project/root")
with DBConnection(ProjectFileStructure(project_path)) as conn:
    metrics = MergedMetrics(conn)

    new_tag = Tag(name="custom tag", scope=TagScope.DATA)
    iterator = DatasetIterator(project_path)
    for data_unit, image in iterator.iterate():
        identifier = f"{iterator.label_hash}_{iterator.du_hash}_{iterator.frame:05d}"
        tags = metrics.get_row(identifier).tags[0]  # Indexing to get the pd.Series content
        tags.append(new_tag)  # Only add TagScope.DATA tags here.
        metrics.update_tags(identifier, tags)
```

## Tagging labels

Tagging labels follows a similar process to tagging images.
In addition to the `label_hash`, `data_hash`, and `frame` of the corresponding image, you will need the hash of the label you are tagging. Specifically, for classifications, you will need the `classificationHash`, and for objects, you will need the `objectHash`.

To tag labels effectively, use the following approach:
```python
from pathlib import Path

from encord_active.lib.common.iterator import DatasetIterator
from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.db.tags import Tag, TagScope
from encord_active.lib.project.project_file_structure import ProjectFileStructure

project_path = Path("/path/to/your/project/root")
with DBConnection(ProjectFileStructure(project_path)) as conn:
    metrics = MergedMetrics(conn)

    iterator = DatasetIterator(project_path)
    new_object_tag = Tag(name="an object tag", scope=TagScope.LABEL)
    new_classification_tag = Tag(name="a classification tag", scope=TagScope.LABEL)
    for data_unit, image in iterator.iterate():
        # For tagging objects
        for obj in data_unit.get("labels", {}).get("objects", []):
            obj_hash = obj["objectHash"]
            identifier = f"{iterator.label_hash}_{iterator.du_hash}_{iterator.frame:05d}_{obj_hash}"

            tags = metrics.get_row(identifier).tags[0]  # Indexing to get the pd.Series content
            tags.append(new_object_tag)  # Only add TagScope.LABEL tags here.
            metrics.update_tags(identifier, tags)

        # For tagging frame-level classifications
        for obj in data_unit.get("labels", {}).get("classifications", []):
            clf_hash = obj["classificationHash"]
            identifier = f"{iterator.label_hash}_{iterator.du_hash}_{iterator.frame:05d}_{clf_hash}"

            tags = metrics.get_row(identifier).tags[0]  # Indexing to get the pd.Series content
            tags.append(new_classification_tag)  # Only add TagScope.LABEL tags here.
            metrics.update_tags(identifier, tags)
```

[tagging-guide]: ../user-guide/tagging
