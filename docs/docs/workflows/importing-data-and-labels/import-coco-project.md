---
sidebar_position: 2
---

# Importing COCO Project

> Import a dataset and annotations which are stored in COCO format on your local file system.

This requires you to have a COCO project on your local machine with images and an annotations JSON file.

To import the project, run

```shell
encord-active import project --coco -i ./images -a ./annotations.json
```

This will create a new Encord Active project in a new directory in you current working directory.
Afterwards, you can run

```shell
encord-active visualize
```

This will let you choose your newly imported project and open the app.

:::info

For the full documentation of importing COCO projects, please see [here](/cli/import-coco-project).

:::
