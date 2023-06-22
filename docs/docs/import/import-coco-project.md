---
sidebar_position: 4
---

# From COCO project

**Create a project using a dataset and annotations stored in COCO format from your local file system**

:::info
Make sure you have installed Encord Active with the `coco` [extras](../installation#coco-extras).
:::

If you have an existing project on your local machine in the COCO data format, you can import it using the following command:

```shell
encord-active import project --coco -i ./images -a ./annotations.json
```

This command will create a new Encord Active project within a fresh folder in the current working directory.
The project will include the specified images and annotations.

:::info
The input of the command above assumes the following structure but it is not limited to it:

```
coco-project-foo
├── images
│   ├── 00000001.jpeg
│   ├── 00000002.jpeg
│   ├── ...
└── annotations.json
```

You have the flexibility to provide any path for each of the arguments, as long as the first argument represents a directory containing images, and the second argument is an annotations file that adheres to the [COCO data format][coco-data-format].
:::

Running the importer will do the following things.

1. Create a local Encord Active project.
2. Compute all the [metrics](/category/quality-metrics).

:::info

Step 1 will by default make a hard copy of the images used in your dataset.
**Optionally**, you can add the `--symlinks` argument

```shell
encord-active import project --coco -i ./images -a ./annotations.json --symlinks
```

to tell Encord Active to use symlinks instead of copying files. But be aware that **if you later move or delete your original image files, Encord Active will stop working!**

:::

The whole flow might take a while depending on the size of the original dataset.

When the process is done, follow the printed instructions to launch the app with the [visualize](../cli#visualize) command.

:::info

If you wish to make the project available on the Encord platform, please consult the [Export section](../user-guide/exporting#export-to-encord) for instructions on how to accomplish this.

:::


[coco-data-format]: https://cocodataset.org/#format-data
