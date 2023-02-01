---
sidebar_position: 3
---

# Initializing from Image Directory

> Grep arbitrary images from within a dataset directory.

If you have images stored in a (potentially nested) directory, you can grep all the images and initialize an Encord Active project from these images.
To do this run

```shell
encord-active init /path/to/image/dir
```

By default, the command will find all `.jpg`, `.jpeg`, `.png`, and `.tiff` files.
If you want to change this, you can use the `--glob` option to target your file selection.

:::tip

The `--dryrun` flag will print the matched files without actually running the import.

:::
:::tip

The `--symlinks` flag will initialize the project symlinks to your files without copying them to save you disk space.

:::

The `init` command will import all the found images into an Encord Active project stored in a directory within your current working directory.
Afterwards, you can run

```shell
encord-active visualize
```

This will let you choose your newly imported project and open the app.

:::info

For the full documentation of initializing a project from an image directory, please see [here](/cli/initialising-project-from-image-directories).

:::
