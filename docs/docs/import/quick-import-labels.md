---
sidebar_position: 2
---

# Quick import labels

To import your custom labels as well, you will have to implement a `LabelTransformer` that transforms your custom labels to Encord Active labels.
The interface for label transforms look as follows:

```python
from pathlib import Path
from typing import List

from encord_active.lib.labels.label_transformer import (
    BoundingBoxLabel,
    ClassificationLabel,
    DataLabel,
    LabelTransformer,
    PolygonLabel
)


class MyTransformer(LabelTransformer):
    def from_custom_labels(self, label_files: List[Path], data_files: List[Path]) -> List[DataLabel]:
        # your implementation goes here
        ...
```

A label transformer will take as input a list of label files and a list of data files (the files we provide to the `init_local_project` below) and needs to return a list of `DataLabel`s.
A `DataLabel` is a pair of a data file and one of `ClassificationLabel`, `BoundingBoxLabel`, and `PolygonLabel`.
There are multiple examples of how to define such transformers in our [examples directory][gh-transformer-examples].

_Coming soon_ - FREDERIK
