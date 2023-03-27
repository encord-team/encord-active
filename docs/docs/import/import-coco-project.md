---
sidebar_position: 4
---

# Importing COCO project

**Import a dataset and annotations which are stored in COCO format on your local file system.**

:::info
Make sure you have installed Encord Active with the `coco` [extras](../../installation#coco-extras).
:::

If you already have a project on your local machine using the COCO data format, you can import that project with the following command:

```shell
encord-active import project --coco -i ./images -a ./annotations.json
```

This will create a new Encord Active project inside a new directory in you current working directory.

:::info
The command above assumes the following structure but is not limited to it.

```

coco-project-foo
├── images
│   ├── 00000001.jpeg
│   ├── 00000002.jpeg
│   ├── ...
└── annotations.json

```

You can provide any path to each of the arguments as long as the first one is a directory of images and the second is an annotations file following the [COCO data format](https://cocodataset.org/#format-data).
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

The whole flow might take a while depending on the size of your dataset.
Bare with us, it is worth the wait.

When the process is done, follow the printed instructions to open the app or see more details [here](../cli#visualize).

:::info

If you want the dataset to be available on the Encord platform, please refer to the [Export documentation](../user-guide/filter_export#export-to-encord).

:::
