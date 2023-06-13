---
sidebar_position: 1
---

# Quick import data & labels

**Grep arbitrary images from within a dataset directory**


If you already have an image dataset stored locally, you can initialize a project from that dataset using the `init` command.
This command will automatically execute all the built-in metrics on your data, setting up the project accordingly.

The main argument is the path to the local dataset directory.

```shell
encord-active init /path/to/dataset
```

:::tip

To simulate the creation of a project without actually performing any action, use the `--dryrun` option.

```shell
encord-active init --dryrun /path/to/dataset
```

This option provides a detailed list of all the files that would be included in the project, along with a summary.
It allows you to verify the project content and ensure that everything is set up correctly before proceeding.

:::

There are various options available to customize the initialization of your project according to your specific requirements.
For a comprehensive list of these options, please refer to the [Command Line Interface][init-command-cli] (CLI) documentation.

## Including Labels

If you want to include labels as well, this is also an option.
To do so, you will have to define how to parse your labels.
You do this by implementing the [`LabelTransformer`][gh-label-transformer-interface] interface.

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

More concrete examples for bounding boxes, polygons and other label types are included in our [example directory][gh-transformer-examples] on GitHub.

:::

[init-command-cli]: ../cli#init
[gh-label-transformer-interface]: https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/labels/label_transformer.py#L61-L79
[gh-transformer-examples]: https://github.com/encord-team/encord-active/blob/main/examples/label-transformers
