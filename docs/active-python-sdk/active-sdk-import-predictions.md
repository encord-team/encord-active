---
title: "Import predictions"
slug: "active-sdk-import-predictions"
hidden: false
createdAt: "2023-08-07T15:36:31.990Z"
updatedAt: "2023-08-11T13:41:50.319Z"
category: "65a71bbfea7a3f005192d1a7"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/local_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

**Learn how to import predictions into Encord Active**

> 🚧 Caution
> When running an importer, any previously imported predictions will be overwritten!

[//]: # (> When running an importer, any previously imported predictions will be overwritten! To ensure the ability to revert to previous iterations, it is important to [version your projects]&#40;https://docs.encord.com/docs/active-versioning&#41;. )

# General import

To import predictions into Encord Active, make sure you have a list of `Prediction` objects ready and follow these steps:
1. Verify that all `Prediction` objects are constructed correctly. If you are unsure about building `Prediction` objects, refer to the [Import model predictions](https://docs.encord.com/docs/active-import-model-predictions) guide, which provides details on how to create predictions for bounding boxes, polygons, masks, and classifications.
2. Use the following code snippet to import the predictions into Encord Active:

```python
from pathlib import Path

from encord_active.lib.db.predictions import Prediction
from encord_active.lib.model_predictions.importers import import_predictions
from encord_active.lib.project import Project

project_dir = Path("/path/to/your/project/directory/")
predictions: list[Prediction] = ...  # Your list of predictions

import_predictions(Project(project_dir), predictions)
```

# Custom imports

## COCO

Simplify importing COCO predictions into Encord Active using `import_coco_predictions()` from `encord_active.lib.model_predictions.importers`, which migrates predictions from the <a href="https://cocodataset.org/#format-results" target="_blank">COCO results format</a><img src="https://storage.googleapis.com/docs-media.encord.com/static/img/icons/external_link_icon.png" width="20" /> to an Encord Active project.

For smooth migration, the `import_coco_prediction` function requires a mappings between the object classes of both formats and the IDs of the images.

The <<glossary:Ontology>> mapping should follow a specific format, where the keys correspond to the `featureNodeHash` of the objects in the project ontology, and the values correspond to their respective COCO class names.

```
{
  # featureNodeHash: class_name
  "OTk2MzM3": "pedestrian",
  "NzYyMjcx": "cyclist",
  "Nzg2ODEx": "car"
}
```

By default, the `import_coco_predictions` function will attempt to read the ontology mapping from a JSON file named `ontology_mapping.json` located in the same directory as the predictions' file. If this file is not present, the user should provide the ontology mapping explicitly.

The image mapping should also follow a specific format, with the keys corresponding to the IDs of the images in the COCO file, and the values corresponding to their respective <<glossary:data unit>> hashes.

```
{
  # image_id: data_unit_hash
  1: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
  2: "ffffffff-gggg-hhhh-iiii-jjjjjjjjjjjj"
}
```

Similarly, the `import_coco_predictions` function will attempt to read the image mapping from a JSON file named `image_mapping.json` located in the same directory as the predictions' file. If this file is not present, the user should provide the image mapping explicitly.

When importing COCO predictions, ensure that both the <<glossary:Ontology>> mapping and image mapping are available to accurately associate the predictions with the appropriate ontology objects and data units.

```python
from pathlib import Path

from encord_active.lib.model_predictions.importers import import_coco_predictions
from encord_active.lib.project import Project

project_dir = Path("/path/to/your/project/directory/")
predictions_json = Path("/path/to/your/predictions.json")

import_coco_predictions(Project(project_dir), predictions_json)
```

> ℹ️ Note
> Remember to provide the ontology mapping and image mapping if they are not available in the same directory as the predictions' file.

Refer to the <a href="https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/model_predictions/importers.py#L102-L129" target="_blank">method's documentation</a><img src="https://storage.googleapis.com/docs-media.encord.com/static/img/icons/external_link_icon.png" width="20" /> for `import_coco_predictions` to learn more about the optional parameters.

## KITTI

The KITTI dataset is a widely-used computer vision dataset for benchmarking and evaluating autonomous driving systems. Due to the popularity of this dataset, many researchers and developers have adopted its label format for their own datasets and applications. Encord Active enhances your experience by migrating such labels to the Encord format.

> 🚧 Caution
> The following approach **only works for bounding boxes**.

Simplify importing KITTI predictions into Encord Active using `import_kitti_predictions()` from `encord_active.lib.model_predictions.importers`, which migrates predictions from the KITTI label files format stored in TXT or CSV files to an Encord Active project.

For smooth migration, the `import_kitti_predictions` function requires a specific file structure and a mapping between the object classes in the project's <<glossary:Ontology>> and the corresponding KITTI class names.

Ensure the predictions' directory comply with the following file structure:

```
predictions
├── example_image.txt
├── example_video__0.csv
├── example_video__1.csv
├── ...
├── ontology_mapping.json (optional)
└── other_files_and_folders (optional)
```

Where each prediction file is named with the format `<data_unit_title>[__<frame>]`, where `<data_unit_title>` is the title of the <<glossary:data unit>>, and `<frame>` (optional) represents the frame number in the video / <<glossary:image group>> / <<glossary:image sequence>>. This naming convention allows the `import_kitti_predictions` function to interpret and associate predictions with the correct data units in the Encord Active project.

The ontology mapping should follow a specific format, where the keys correspond to the `featureNodeHash` of the <<glossary:bounding box>> objects in the project's ontology, and the values correspond to their respective KITTI class names.

```
{
  # featureNodeHash: class_name
  "OTk2MzM3": "pedestrian",
  "NzYyMjcx": "cyclist",
  "Nzg2ODEx": "car"
}
```

By default, the `import_kitti_predictions` function will attempt to read the <<glossary:Ontology>> mapping from a JSON file named `ontology_mapping.json` located in the predictions' directory. If this file is not present, the user should provide the ontology mapping explicitly.

When importing KITTI predictions, ensure that both the file structure and ontology mapping are valid and available to accurately associate the predictions with the appropriate ontology objects and data units.

```python
from pathlib import Path

from encord_active.lib.model_predictions.importers import import_kitti_predictions
from encord_active.lib.project import Project

project_dir = Path("/path/to/your/project/directory/")
predictions_dir = Path("/path/to/your/predictions/directory/")

import_kitti_predictions(Project(project_dir), predictions_dir)
```

> ℹ️ Note
> Remember to provide the <<glossary:Ontology>> mapping if it is not available in the predictions' directory.

You can further filter the predictions files using the `file_name_regex` optional parameter, allowing you to select only specific prediction files based on their names. Additionally, if needed, you can provide a custom function `file_path_to_data_unit_func` that matches file names with the corresponding data units in Encord, allowing you to precisely identify the data units for each prediction. Refer to the <a href="https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/model_predictions/importers.py#L139-L167" target="_blank">method's documentation</a><img src="https://storage.googleapis.com/docs-media.encord.com/static/img/icons/external_link_icon.png" width="20" /> for `import_kitti_predictions` to learn more about the optional parameters.

### Label Files Format

The KITTI importer supports the label format described <a href="https://docs.nvidia.com/tao/tao-toolkit-archive/tlt-20/tlt-user-guide/text/preparing_data_input.html#label-files" target="_blank">here</a><img src="https://storage.googleapis.com/docs-media.encord.com/static/img/icons/external_link_icon.png" width="20" /> with the addition of a column corresponding to the model confidence.

An example:

```
car 0.00 0 0.00 587.01 173.33 614.12 200.12 0.00 0.00 0.00 0.00 0.00 0.00 0.00 97.85
cyclist 0.00 0 0.00 665.45 160.00 717.93 217.99 0.00 0.00 0.00 0.00 0.00 0.00 0.00 32.65
pedestrian 0.00 0 0.00 423.17 173.67 433.17 224.03 0.00 0.00 0.00 0.00 0.00 0.00 0.00 3.183
```

Columns are:

- `class_name`: str
- ~~`truncation`: float~~ ignored
- ~~`occlusion`: int~~ ignored
- ~~`alpha`: float~~ ignored
- `xmin`: float
- `ymin`: float
- `xmax`: float
- `ymax`: float
- ~~`height`: float~~ ignored
- ~~`width`: float~~ ignored
- ~~`length`: float~~ ignored
- ~~`location_x`: float~~ ignored
- ~~`location_y`: float~~ ignored
- ~~`location_z`: float~~ ignored
- ~~`rotation_y`: float~~ ignored
- `confidence`: float

> ℹ️ Note
> The columns flagged as `ignored` have to appear in the label format but their values will be ignored.

## Instance segmentation masks

> 🚧 Caution
> The following approach will **transform segmentations into simple polygons**.

Simplify importing predictions stored as `.png` masks of shape `[height, width]`, where each pixel value correspond to a class, into Encord Active using `import_mask_predictions()` from `encord_active.lib.model_predictions.importers`, which migrates predictions from segmentation masks to an Encord Active project.

For smooth migration, the `import_mask_predictions` function requires a specific file structure and a mapping between the object classes in the project's ontology and the corresponding IDs in the masks.

Ensure the predictions' directory comply with the following file structure:

```
predictions
├── example_image.png
├── example_video__0.png
├── example_video__1.png
├── ...
├── ontology_mapping.json (optional)
└── other_files_and_folders (optional)
```

Where each prediction file is named with the format `<data_unit_title>[__<frame>]`, where `<data_unit_title>` is the title of the <<glossary:data unit>>, and `<frame>` (optional) represents the frame number in the video / <<glossary:image group>> / <<glossary:image sequence>>. This naming convention allows the `import_mask_predictions` function to interpret and associate predictions with the correct data units in the Encord Active project.

The ontology mapping should adhere to a specific format, with the keys corresponding to the `featureNodeHash` of the polygon objects in the project's ontology, and the values corresponding to their respective IDs in the masks.

```
{
    # featureNodeHash: pixel_value
    "OTk2MzM3": 1,  # "pedestrian"
    "NzYyMjcx": 2,  # "cyclist",
    "Nzg2ODEx": 3,  # "car"
    # Note: value: 0 is reserved for "background"
}
```

By default, the `import_mask_predictions` function will attempt to read the ontology mapping from a JSON file named `ontology_mapping.json` located in the predictions' directory. If this file is not present, the user should provide the ontology mapping explicitly.

When importing mask predictions, ensure that both the file structure and ontology mapping are valid and available to accurately associate the predictions with the appropriate ontology objects and data units.

```python
from pathlib import Path

from encord_active.lib.model_predictions.importers import import_mask_predictions
from encord_active.lib.project import Project

project_dir = Path("/path/to/your/project/directory/")
predictions_dir = Path("/path/to/your/predictions/directory/")

import_mask_predictions(Project(project_dir), predictions_dir)
```

> ℹ️ Note
> Remember to provide the ontology mapping if it is not available in the predictions' directory.

You can further filter the predictions files using the `file_name_regex` optional parameter, allowing you to select only specific prediction files based on their names. Additionally, if needed, you can provide a custom function `file_path_to_data_unit_func` that matches file names with the corresponding data units in Encord, allowing you to precisely identify the data units for each prediction. Refer to the <a href="https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/model_predictions/importers.py#L178-L207" target="_blank">method's documentation</a><img src="https://storage.googleapis.com/docs-media.encord.com/static/img/icons/external_link_icon.png" width="20" /> for `import_mask_predictions` to learn more about the optional parameters.

> 🚧 Caution
> For each imported file, every "self-contained" contour will be interpreted as an individual prediction. For example, the following mask will be interpreted as three objects: two from class 1 and one from class 2.
> 
> ```
> ┌───────────────────┐
> │0000000000000000000│
> │0011100000000000000│
> │0011110000002222000│
> │0000000000002222000│
> │0000111000002200000│
> │0000111000002200000│
> │0000111000000000000│
> │0000000000000000000│
> └───────────────────┘
> ```
> 
> Also, the confidence of the predictions will be set to 1.