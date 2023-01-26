---
sidebar_position: 3
---

# Initialise from Image Directory

If you have an image dataset stored on your local system already, you can initialise an Encord Project from that dataset with the `init` command.
The command will automatically run all the existing metrics on your data.

:::info

This only works for images at the moment. We are planning to extend this to videos as well.

:::

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

### Options

#### `--glob` (or `-g`)

> **Default:** `"**/*.jpg"`, `"**/*.png"`, `"**/*.jpeg"`, `"**/*.tiff"`.

Glob patterns are used to choose files.
You can specify multiple options if you wish to include files from specific subdirectories.
For example,

```shell
encord-active init -g "val/*.jpg" -g "test/*.jpg" /path/to/dataset
```

would match `jpg` files in the `val` and `test` directories but not in the `train` direstory of the folloing file structure:

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

#### `--target` (or `-t`)

> **Default:** the current working directory.

Directory where the project would be saved.
The project will be stored in a directory within the target directory (see the `--name` option for details).

#### `--name` (or `-n`)

> **Default:** the name of the data directory prepended with `"[EA] "`.

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

#### `--dryrun`

> **Default:** `False`

Print the files that will be imported WITHOUT actually creating a project.
This option is helpful if you want to validate that you are actually importing the correct data.
