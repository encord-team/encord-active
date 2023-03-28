---
sidebar_position: 1
---

# Quick import data & labels

**Grep arbitrary images from within a dataset directory.**

If you have an image dataset stored on your local system already, you can initialise a project from that dataset with the `init` command.
The command will automatically run all the existing metrics on your data.

The main argument is the root of your dataset directory.

```shell
encord-active init /path/to/dataset
```

:::tip

If you add the `--dryrun` option:

```shell
#  after command      👇    before the path
encord-active init --dryrun /path/to/dataset
```

No project will be initialised but all the files that would be included will be listed.
You can use this for making sure that Encord Active will include what you expect before starting the actual initialisation.

:::

You have some additional options to tailor the initialisation of the project for your needs.

## Including Labels

If you want to include labels as well, this is also an option.
To do so, you will have to define how to parse your labels.
You do this by implementing the [`LabelTransformer`][gh-label-transformer] interface.

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

Here is an example of inferring classifications from the file structure of the images.
Let's say you have your images stored in the following structure:

```
/path/to/data_root
├── cat
│   ├── 0.jpg
│   └── ...
├── dog
│   ├── 0.jpg
│   └── ...
└── horse
    ├── 0.jpg
    └── ...

```

Your implementation would look similar to:

```python
# classification_transformer.py
from pathlib import Path
from typing import List

from encord_active.lib.labels.label_transformer import (
    ClassificationLabel,
    DataLabel,
    LabelTransformer,
)


class ClassificationTransformer(LabelTransformer):
    def from_custom_labels(self, _, data_files: List[Path]) -> List[DataLabel]:
        return [DataLabel(f, ClassificationLabel(class_=f.parent.name)) for f in data_files]
```

And the CLI command:

```shell
encord-active init --transformer classification_transformer.py /path/to/data_root
```

:::tip

More concrete examples for, e.g., bounding boxes and polygons, are included in our [example directory][gh-transformer-examples] on GitHub.

:::

[gh-transformer-examples]: https://github.com/encord-team/encord-active/blob/main/examples/label-transformers
