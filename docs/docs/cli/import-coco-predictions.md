---
sidebar_position: 5
---

# Importing Coco Predictions

:::info
Make sure you have installed Encord Active with the `coco` [extras](/installation#coco-extras).
:::

:::note
This command assume that you have imported you project using the [COCO importer](/cli/import-coco-project) and you are inside the projects directory.
:::

Importing COCO predictions is currently the easiest way to import predictions to Encord Active.

You need to have a results JSON file following the [COCO results format](https://cocodataset.org/#format-results) and run the following command on it:

```shell
encord-active import predictions --coco results.json
```

After the execution is done you should be ready to view your [model assertions metrics](/category/model-assertions).
