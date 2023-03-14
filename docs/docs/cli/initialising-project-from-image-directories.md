---
sidebar_position: 3
---

# Initialising from Image Directory

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

If you want to include labels as well, this is also an option. To do so, you will have to define how to parse your labels.
You do this by implementing the [`LabelTransformer`][gh-label-transformer] interface.

```python
from pathlib import Path

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

Your implementation would look similar to this:

```python
from pathlib import Path

from encord_active.lib.labels.label_transformer import (
    ClassificationLabel,
    DataLabel,
    LabelTransformer
)


class MyTransformer(LabelTransformer):
    def from_custom_labels(self, label_files: List[Path], data_files: List[Path]) -> List[DataLabel]:
        out: List[DataLabel] = []
        for data_file in data_files:
            out.append(DataLabel(data_file, ClassificationLabel(class_=data_file.parent.name))
        return out
```

:::tip

More concrete examples for, e.g., bounding boxes and polygons, are included in our [example directory][gh-transformer-examples] on GitHub.

:::

## CLI Options

#### `--data-glob` (or `-dg`)

> **Default:** `"**/*.jpg"`, `"**/*.jpeg"`, `"**/*.png"`, `"**/*.tiff"`

Glob patterns used to choose data files, i.e., images.
You can specify multiple options if you wish to include files from specific subdirectories.
For example,

```shell
encord-active init -dg "val/*.jpg" -dg "test/*.jpg" /path/to/dataset
```

would match `jpg` files in the `val` and `test` directories but not in the `train` directory of the following file structure:

```
/path/to/dataset
├── train
│   ├── 0.jpg
│   └── ...
├── val
│   ├── 0.jpg
│   └── ...
└── test
    ├── 0.jpg
    └── ...
```

The matched files will be passed to the [`LabelTransformer.from_custom_labels`][gh-label-transformer] method for you to parse and transform into [`DataLabel`][gh-data-label] objects -- if you use the `--transformer` option.

#### `--label-glob` (or `-lg`)

> **Default:** `None`

Glob patterns used to choose data files, i.e., files that contain label information.
You can specify multiple options if you wish to include files from specific subdirectories.

For example,

```shell
encord-active init -lg "val/*.txt" -lg "test/*.txt" /path/to/dataset
```

would match `txt` files in the `val` and `test` directories but not in the `train` directory of the following file structure:

```
/path/to/dataset
├── train
│   ├── 0.jpg
│   ├── 0.txt
│   └── ...
├── val
│   ├── 0.jpg
│   ├── 0.txt
│   └── ...
└── test
    ├── 0.jpg
    ├── 0.txt
    └── ...
```

The files will be passed to the [`LabelTransformer.from_custom_labels`][gh-label-transformer] method for you to parse and transform into [`DataLabel`][gh-data-label] objects.

#### `--target` (or `-t`)

> **Default:** the current working directory

Directory where the project would be saved.
The project will be stored in a directory within the target directory (see the `--name` option for details).

#### `--name` (or `-n`)

> **Default:** the name of the data directory prepended with `"[EA] "`

Name to give the new Encord Active project directory.

For example, if you run

```
encord-active init --target foo/bar --name baz /path/to/dataset
```

A new directory named `baz` will be generated in the `foo/bar` directory containing the project files.

#### `--symlinks`

> **Default:** `False`

Use symlinks instead of copying images to the Encord Active project directory.
This will save you a lot of space on your system.

:::warning

If you later move the data that the symlinks are pointing to, the Encord Active project will stop working - unless you manually update the symlinks.

:::

#### `--transformer`

> **Default:** `None`

A file path to a python implementation of the [`LabelTransformer`][gh-label-transformer] interface.
This file can contain one or more implementations and you will be able to choose interactively from the UI which ones you would like to apply during your project initialization.

:::tip

To see some reference implementations, please visit our [GitHub examples directory][gh-transformer-examples].

:::

#### `--dryrun`

> **Default:** `False`

Print the files that will be imported WITHOUT actually creating a project.
This option is helpful if you want to validate that you are actually importing the correct data.

[gh-label-transformer]: https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/labels/label_transformer.py#61
[gh-data-label]: https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/labels/label_transformer.py#56
[gh-transformer-examples]: https://github.com/encord-team/encord-active/blob/main/examples/label-transformers
