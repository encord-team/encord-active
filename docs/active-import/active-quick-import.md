---
title: "Quick import data & labels"
slug: "active-quick-import"
hidden: false
createdAt: "2023-07-21T14:06:06.472Z"
updatedAt: "2023-07-31T11:23:32.263Z"
category: "65a71bbfea7a3f005192d1a7"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/local_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

---

**Create a project using images from a dataset directory of your choice**


If you already have an image dataset stored locally, you can initialize a project from that dataset using the `init` command. This command will automatically execute all the built-in metrics on your data, setting up the project accordingly.

The main argument is the path to the local dataset directory.

```shell
encord-active init /path/to/dataset
```

> 👍 Tip
> To simulate the creation of a project without actually performing any action, use the `--dryrun` option.
>
> ```shell
> encord-active init --dryrun /path/to/dataset
> ```
>
> This option provides a detailed list of all the files that would be included in the project, along with a summary. It allows you to verify the project content and ensure that everything is set up correctly before proceeding.

There are various options available to customize the initialization of your project according to your specific requirements. For a comprehensive list of these options, please refer to the [Command Line Interface](https://docs.encord.com/docs/active-cli#init) (CLI) documentation.

## Including labels

If you want to include labels as well, this is also an option. To do so, you'll have to define how to parse your labels by implementing the [`LabelTransformer`](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/labels/label_transformer.py#L61-L79) interface.

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

> 👍 Tip
> More concrete examples for bounding boxes, polygons and other label types are included in our [example directory](https://github.com/encord-team/encord-active/blob/main/examples/label-transformers) on GitHub.