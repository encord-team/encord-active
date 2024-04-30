---
title: "Project initialization"
slug: "active-sdk-project-initialization"
hidden: false
metadata: 
  title: "Project Initialization"
  description: "Start your project right: Initialize with or without labels using Python code. Quick setup for optimal results."
  image: 
    0: "https://files.readme.io/422fb92-image_16.png"
createdAt: "2023-07-14T16:05:36.364Z"
updatedAt: "2023-08-09T16:22:25.711Z"
category: "65a71bbfea7a3f005192d1a7"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/local_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

**Learn how to initialize a project with and without labels**

Project initialization using Python code provides the same functionality as running the [`init`](https://docs.encord.com/docs/active-cli#init) command in the CLI. If you prefer using the CLI, you can refer to the [Quick import data & labels](https://docs.encord.com/docs/active-quick-import) guide.

## Steps to initialize a project without labels

[block:embed]
{
  "url": "https://colab.research.google.com/github/encord-team/encord-active/blob/main/examples/initialization-project-without-labels-using-python.ipynb",
  "title": "Initialization of a project without labels in Encord Active using Python",
  "favicon": "https://cdn.simpleicons.org/googlecolab/#F9AB00",
  "image": "https://cdn.simpleicons.org/googlecolab/#F9AB00",
  "provider": "colab.research.google.com",
  "href": "https://colab.research.google.com/github/encord-team/encord-active/blob/main/examples/initialization-project-without-labels-using-python.ipynb"
}
[/block]

1. Choose the images you want to import.
  For example, you can use the `glob` function to find all the files with the ".jpg" extension in your current working directory, including subdirectories. Customize the logic as needed to include specific files.

  ```python
  from pathlib import Path

  image_files = list(Path.cwd().glob("**/*.jpg"))
  ```

2. Specify the directory where you want to store the Encord Active project.
  It is recommended to keep multiple projects within the same directory for easy navigation within the UI.
  
  ```python
  projects_dir = Path("path/to/where/you/store/projects")
  ```

3. Create the project.
  Note that the `symlinks` option determines whether files are copied (`symlinks=False`) or referenced using symlinks to save disk space (`symlinks=True`).

  ```python
  from encord_active.lib.project.local import init_local_project
  
  project_path = init_local_project(
      files=image_files,
      target=projects_dir,
      project_name="<your project title>",
      symlinks=True,
  )
  ```

4. Run a metric to ensure proper functioning of the UI.
  
  ```python
  from encord_active.lib.metrics.execute import execute_metrics
  from encord_active.lib.metrics.heuristic.img_features import AreaMetric
  
  execute_metrics(
      selected_metrics=[AreaMetric()],
      data_dir=project_path,
      use_cache_only=True,
  )
  ```

  Alternatively, you can run all metrics that do not depend on any labels using the following code snippet:
  
  ```python
  from encord_active.lib.metrics.execute import run_metrics_by_embedding_type
  from encord_active.lib.metrics.types import EmbeddingType

  run_metrics_by_embedding_type(
      EmbeddingType.IMAGE,
      data_dir=project_path,
      use_cache_only=True,
  )
  ```

After completing these steps, you can launch the application with the [start][ea-cli-start] CLI command and access the project:

```shell
encord-active start -t "path/to/where/you/store/projects"
```

## Steps to initialize a project with labels

[block:embed]
{
  "url": "https://colab.research.google.com/github/encord-team/encord-active/blob/main/examples/initialization-project-with-labels-using-python.ipynb",
  "title": "Initialization of a project with labels in Encord Active using Python",
  "favicon": "https://cdn.simpleicons.org/googlecolab/#F9AB00",
  "image": "https://cdn.simpleicons.org/googlecolab/#F9AB00",
  "provider": "colab.research.google.com",
  "href": "https://colab.research.google.com/github/encord-team/encord-active/blob/main/examples/initialization-project-with-labels-using-python.ipynb"
}
[/block]

If you have previously defined a `LabelTransformer` as explained in the [Quick import data & labels](https://docs.encord.com/docs/active-quick-import#including-labels) guide, you can utilize it in the project initialization process. To include labels, you need to provide the transformer object and the corresponding label files to the `init_local_project` function.

1. Choose the images and label files you want to import.
  For example, you can use the `glob` function to find all the files with the ".jpg" extension and label files with the ".json" extension in your current working directory, including subdirectories. Customize the logic as needed to include specific files.

  ```python
  from pathlib import Path

  image_files = list(Path.cwd().glob("**/*.jpg"))
  label_files = list(Path.cwd().glob("**/*.json"))
  ```

2. Define a class that implements the [`LabelTransformer`][gh-label-transformer-interface] interface and handles the parsing of labels.
  For instance, you can refer to the implementation of the [`BBoxTransformer`][gh-bbox-transformer] class. Instantiate this class to utilize it for including labels in your project.

  ```python
  label_transformer = BBoxTransformer()
  ```

  > 👍 Tip
  > Check out more label transformer examples in the [examples section][gh-transformer-examples] of Encord Active's GitHub repository.
  

3. Specify the directory where you want to store the Encord Active project.
  It is recommended to keep multiple projects within the same directory for easy navigation within the UI.
  
  ```python
  projects_dir = Path("path/to/where/you/store/projects")
  ```

4. Create the project.
  Note that the `symlinks` option determines whether files are copied (`symlinks=False`) or referenced using symlinks to save disk space (`symlinks=True`).

  ```python
  from encord_active.lib.project.local import init_local_project
  
  project_path = init_local_project(
      files = image_files,
      target = projects_dir,
      project_name = "<your project title>",
      symlinks = True,
      label_transformer=label_transformer,
      label_paths=label_files,
  )
  ```

5. Run a metric to ensure proper functioning of the UI.
  
  ```python
  from encord_active.lib.metrics.execute import execute_metrics
  from encord_active.lib.metrics.heuristic.img_features import AreaMetric
  
  execute_metrics(
      selected_metrics=[AreaMetric()],
      data_dir=project_path,
      use_cache_only=True,
  )
  ```

  Alternatively, you can run all metrics related to labels using the following code snippet:
  
  ```python
  from encord_active.lib.metrics.execute import run_metrics_by_embedding_type
  from encord_active.lib.metrics.types import EmbeddingType
  
  run_metrics_by_embedding_type(
      EmbeddingType.OBJECT,
      data_dir=project_path,
      use_cache_only=True,
  )

  run_metrics_by_embedding_type(
      EmbeddingType.CLASSIFICATION,
      data_dir=project_path,
      use_cache_only=True,
  )
  ```

After completing these steps, you can launch the application by using the following CLI command and access the project:

```shell
encord-active start -t "path/to/where/you/store/projects"
```

[ea-cli-start]: https://docs.encord.com/docs/active-cli#start
[gh-label-transformer-interface]: https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/labels/label_transformer.py#L61-L79 
[gh-bbox-transformer]: https://github.com/encord-team/encord-active/blob/main/examples/label-transformers/bounding-boxes/bbox_transformer.py
[gh-transformer-examples]: https://github.com/encord-team/encord-active/blob/main/examples/label-transformers