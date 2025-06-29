---
title: "Import a COCO project"
slug: "active-import-coco-project"
hidden: false
metadata: 
  title: "Import a COCO project"
  description: "Start a project in Encord Active: Utilize local COCO format datasets and annotations. Streamlined project creation."
  image: 
    0: "https://files.readme.io/675f89d-image_16.png"
createdAt: "2023-07-11T16:27:41.928Z"
updatedAt: "2023-08-11T12:43:00.436Z"
category: "65a71bbfea7a3f005192d1a7"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/local_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

**Create a project using a dataset and annotations stored in COCO format from your local file system**

> ℹ️ Note
> Make sure you have installed Encord Active with the `coco` [extras](https://docs.encord.com/docs/active-oss-install#coco-extras).


If you have an existing project on your local machine in the COCO data format, you can import it using the following command:

```shell
encord-active import project --coco -i ./images -a ./annotations.json
```

This command will create a new Encord Active project within a fresh folder in the current working directory. The project will include the specified images and annotations.

> ℹ️ Note
> The input of the command above assumes the following structure, but it is not limited to it:
> 
> ```
> coco-project-foo
> ├── images
> │   ├── 00000001.jpeg
> │   ├── 00000002.jpeg
> │   ├── ...
> └── annotations.json
> ```
> 
> You have the flexibility to provide any path for each of the arguments, as long as the first argument represents a directory containing images, and the second argument is an annotations file that adheres to the [COCO data format](https://cocodataset.org/#format-data).

Running the importer will do the following things.

1. Create a local Encord Active project.
2. Compute all the [metrics](https://docs.encord.com/docs/active-quality-metrics).

> ℹ️ Note
> Step 1 will by default make a hard copy of the images used in your dataset.
> **Optionally**, you can add the `--symlinks` argument
> 
> ```shell
> encord-active import project --coco -i ./images -a ./annotations.json --symlinks
> ```
> 
> to tell Encord Active to use symlinks instead of copying files. But be aware that **if you later move or delete your original image files, Encord Active will stop working!**

The whole flow might take a while depending on the size of the original dataset. When the process is done, follow the printed instructions to launch the app with the [start][ea-cli-start] CLI command.

[//]: # (TODO show the note when the export section shows how to export data and labels to encord)
[//]: # (> ℹ️ Note)
[//]: # (> If you wish to make the project available on the Encord platform, please consult the [Export section]&#40;https://docs.encord.com/docs/active-exporting#export-to-the-encord-platform&#41; for instructions on how to accomplish this.)


[ea-cli-start]: https://docs.encord.com/docs/active-cli#start