---
sidebar_position: 3
---

# Importing Coco Project

:::info
Make sure you have installed Encord Active with the `coco` [extras](/installation#coco-extras).
:::

If you already have a project on your local machine using the COCO data format, you can import that project with the following command:

```shell
(ea-venv)$ encord-active import project --coco ./images ./annotations.json
```

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

If this is your first time using Encord Active, you will need the path to your private `ssh-key` associated with Encord and a projects directory where all projects should be stored.
Don't worry! The script will guide you on the way if you don't know it already.

Running the importer will do the following things.

1. Create a dataset.
2. Create an ontology.
3. Create a project.
4. Create a local Encord Active project.
5. Compute all the [metrics](category/metrics).

:::info

- Steps 1-3 are on the Encord platform. This is a dependency that we intend to make optional in the future.
- Step 4 will make a hard copy of the images used in your dataset. In the future, we expect to support symlinks or just pointers to the original images.

:::

The whole flow might take a while depending on the size of your dataset.
Bare with us, it is worth the wait.

When the process is done, follow the printed instructions to open the app or see more details in the [Open Encord Active](/cli/open-encord-active) page.
