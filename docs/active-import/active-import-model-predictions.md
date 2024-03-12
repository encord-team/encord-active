---
title: "Import model predictions"
slug: "active-import-model-predictions"
hidden: true
metadata: 
  title: "Import model predictions"
  description: "Enhance Encord Active with model predictions. Integrate for visualizations, evaluation, labeling insights. Boost system performance."
  image: 
    0: "https://files.readme.io/091496f-image_16.png"
createdAt: "2023-07-14T16:16:03.504Z"
updatedAt: "2023-08-11T13:41:50.171Z"
category: "65a71bbfea7a3f005192d1a7"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/local_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

**Incorporate model predictions into Encord Active**

By incorporating machine learning model predictions, Encord Active expands its capabilities to provide visualizations, model evaluation, identification of failure modes, error detection in labeling, prioritization of high-value data for relabeling, and other valuable insights, enhancing the overall performance of the system.

> ℹ️ Note
> If you possess predictions in the [COCO Results format](https://cocodataset.org/#format-results), you can conveniently navigate to the [Import COCO predictions](#coco-predictions) subsection.


To import your model predictions into Encord Active you need to:

1. [Cover the basics](#covering-the-basics).
2. [Prepare a `.pkl` file to be imported](#prepare-a-pkl-file-to-be-imported).
3. [Import the predictions via the CLI](#import-the-predictions-via-the-cli).

If you are familiar with [`data_hash`](#uniquely-identifying-data-units) and [`featureNodeHash`](#uniquely-identifying-predicted-classes), you can safely skip to [2. Prepare a `.pkl` File to be Imported](#prepare-a-pkl-file-to-be-imported). Please note that when specifying the `class_id` for a prediction, Encord Active expects the associated `featureNodeHash` from the Encord <<glossary:Ontology>> as id.

> 👍 Tip
> In the [SDK section](https://docs.encord.com/docs/active-sdk-import-predictions), you will also find ways to import predictions [KITTI files](https://docs.encord.com/docs/active-sdk-import-predictions#kitti) and [directories containing mask files](https://docs.encord.com/docs/active-sdk-import-predictions#instance-segmentation-masks).


## Covering the basics

Before diving into the details, there are a couple of things that need to be covered.

> ℹ️ Note
> All commands used from this point onward assume that the current working directory is the project folder. If it isn't, please either navigate to it or utilize the `--target` option available in each command.


### Uniquely identifying data units

At Encord, every <<glossary:data unit>> has a `data_hash` which uniquely defines it. To view the mapping between the `data_hash` values in your Encord project and the corresponding filenames that were uploaded, execute the following CLI command:

```shell
encord-active print data-mapping
```

Once you have selected the project for which you want to generate the mapping, it will display a JSON object resembling the following structure, consisting of key-value pairs (`data_hash`, `data_file_name`):

```
{
  "c115344f-6869-4608-a4b8-644241fea10c": "image_1.jpg",
  "5973f5b6-d284-4a71-9e7e-6576aa3e56cb": "image_2.jpg",
  "9f4dae86-cad4-42f8-bb47-b179eb2e4886": "video_1.mp4"
  ...
}

```

> 👍 Tip
> To store the data mapping as `data_mapping.json` in the current working directory, run:
> 
> ```shell
> encord-active print --json data-mapping
> ```

Please note that in the case of image groups, each individual image within the group has its own unique `data_hash`, whereas videos have a single `data_hash` representing the entire video. As a consequence, predictions for videos will also need a `frame` to uniquely define where the prediction belongs.

> 🚧 Caution
> When you are preparing predictions for import, you need to have the `data_hash` and potentially the `frame` available.


### Uniquely identifying predicted classes

The second thing you will need during the preparation of predictions for import, is the `class_id` for each prediction. The `class_id` tells Encord Active which class the prediction is associated with.

The `class_id` values in an Encord project are determined by the `featureNodeHash` attribute associated with labels in the Encord <<glossary:Ontology>>. You can conveniently print the class names and corresponding `class_id` values of your project ontology via the CLI:

```shell
encord-active print ontology
```

Once you have selected the project for which you want to generate the mapping, it will display a JSON object resembling the following structure, consisting of key-value pairs (`label_name`, `class_id`):

```json
{
  "objects": {
    "cat": "OTK8MrM3",
    "dog": "Nr52O8Ex",
    "horse": "MjkXn2Mx"
  },
  "classifications": {...}
}
```

As classifications with nested and/or checklist attributes (e.g. `has a dog? yes/no -> explain why?`) are represented once for each attribute answer, it's necessary to uniquely identify each <<glossary:classification>> and corresponding answer. This requires utilizing the respective <<glossary:classification>>, attribute and option hashes from the <<glossary:Ontology>>.

```json
{
  "objects": {...},
  "classifications": {
    "horses": {
      "feature_hash": "55eab8b3",
      "attribute_hash": "d446851e",
      "option_hash": "376b9761"
    },
    "cats": {
      "feature_hash": "55eab8b3",
      "attribute_hash": "d446851e",
      "option_hash": "d8e85460"
    },
    "dogs": {
      "feature_hash": "55eab8b3",
      "attribute_hash": "d446851e",
      "option_hash": "e5264a59"
    }
  }
}
```

> 👍 Tip
> To store the ontology as `ontology_output.json` in the current working directory, run:
> 
> ```shell
> encord-active print --json ontology
> ```


## Prepare a `.pkl` file to be imported

Now, you can prepare a pickle file (`.pkl`) to be imported by Encord Active. You can do this by building a list of `Prediction` objects. A prediction object holds a unique identifier of the <<glossary:data unit>> (the `data_hash` and potentially `frame`), the `class_id`, a model `confidence` score, the actual prediction `data`, and the `format` of that data.

### Creating a `Prediction` label

Below are examples illustrating how to create a label. Click a section to expose the details for each of the four supported types.

<details>
<summary>Classification</summary>

```python
prediction = Prediction(
    data_hash="<data_hash>",
    frame = 3,  # optional frame for videos
    confidence = 0.8,
    classification=FrameClassification(
        feature_hash="<class_id>",
        # highlight-start
        attribute_hash="<attribute_hash>",
        option_hash="<option_hash>",
        # highlight-end
    ),
)
```

> 👍 Tip
> To find the three hashes, we can inspect the ontology by running:
> 
> ```shell
> encord-active print ontology
> ```
</details>


<details>
<summary>Bounding Box</summary>

You should specify your `BoundingBox` with relative coordinates and dimensions.
That is:

- `x`: x-coordinate of the top-left corner of the box divided by the image width
- `y`: y-coordinate of the top-left corner of the box divided by the image height
- `w`: box pixel width / image width
- `h`: box pixel height / image height

```python
from encord_active.lib.db.predictions import BoundingBox, Prediction, Format, ObjectDetection

prediction = Prediction(
    data_hash="<data_hash>",
    frame = 3,  # optional frame for videos
    confidence = 0.8,
    object=ObjectDetection(
        feature_hash="<class_id>",
        # highlight-start
        format = Format.BOUNDING_BOX,
        data = BoundingBox(x=0.2, y=0.3, w=0.1, h=0.4),
        # highlight-end
    ),
)
```

> 👍 Tip
> If you don't have your <<glossary:bounding box>> represented in relative terms, you can convert it from pixel values like this:
> 
> ```python
> img_h, img_w = 720, 1280  # the image size in pixels
> BoundingBox(x=10/img_w, y=25/img_h, w=200/img_w, h=150/img_h)
> ```

</details>


<details>
<summary>Segmentation mask</summary>

You should specify masks as binary `numpy` arrays of size $height \times width$ and `dtype=np.uint8`.

```python
from encord_active.lib.db.predictions import Prediction, Format

prediction = Prediction(
    data_hash =  "<data_hash>",
    frame = 3,  # optional frame for videos
    confidence = 0.8,
    object=ObjectDetection(
        feature_hash="<class_id>",
        # highlight-start
        format = Format.MASK,
        data = mask
        # highlight-end
    ),
)
```

</details>

<details>
<summary>Polygon</summary>

You should specify your `Polygon` with relative coordinates as a `numpy` array of shape $num\_points \times 2$. That is, an array of relative (`x`, `y`) coordinates:

- `x`: relative x-coordinate of each point of the polygon (pixel coordinate / image width).
- `y`: relative y-coordinate of each point of the polygon (pixel coordinate / image height).

```python
from encord_active.lib.db.predictions import Prediction, Format
import numpy as np

polygon = np.array([
    # x    y
    [0.2, 0.1],
    [0.2, 0.4],
    [0.3, 0.4],
    [0.3, 0.1],
])

prediction = Prediction(
    data_hash =  "<data_hash>",
    frame = 3,  # optional frame for videos
    confidence = 0.8,
    object=ObjectDetection(
        feature_hash="<class_id>",
        # highlight-start
        format = Format.POLYGON,
        data = polygon
        # highlight-end
    ),
)
```

> 👍 Tip
> If you have your <<glossary:polygon>> represented in absolute terms of pixel locations, you can convert it to relative terms like this:
> 
> ```python
> img_h, img_w = 720, 1280  # the image size in pixels
> polygon = polygon / np.array([[img_w, img_h]])
> ```

Notice the double braces `[[img_w, img_h]]` to get an array of shape `[1, 2]`.

<details>

### Creating the pickle file

Now you are ready to create the pickle file. You can select the appropriate snippet based on your prediction format from above and paste it in the code below.

Pay attention to the highlighted line, as it specifies the location where the `.pkl` file will be stored.

```python showLineNumbers
import pickle
from encord_active.lib.db.predictions import Prediction, Format

predictions_to_store = []

for prediction in my_predictions:  # Iterate over your predictions
    predictions_to_store.append(
        # PASTE appropriate prediction snippet from above
    )

# highlight-next-line
with open("/path/to/predictions.pkl", "wb") as f:
    pickle.dump(predictions_to_store, f)
```

In the above code snippet, you will have to fetch `data_hash`, `class_id`, etc., from the for loop in line 5.

## Import the predictions via the CLI

To import the predictions into Encord Active, execute the following command in the CLI:

```shell
encord-active import predictions /path/to/predictions.pkl
```

This will import your predictions into Encord Active and run all the [metrics](https://docs.encord.com/docs/active-quality-metrics) on your predictions.

## Easy imports

Encord Active streamlines the import of well-known model prediction formats allowing for easy integration of diverse model types into the system.

The following subsections outline simplified methods to import popular formats, bypassing the previous 3-step process.

### COCO predictions

> ℹ️ Note
> Make sure you have installed Encord Active with the `coco` [extras](https://docs.encord.com/docs/active-oss-install#coco-extras).
>
> This command assumes that you have imported your project using the [COCO importer](https://docs.encord.com/docs/active-cli#project) and that the current working directory is the project folder.

Importing COCO predictions is currently the easiest way to import predictions into Encord Active.

You need to have a results JSON file following the [COCO results format](https://cocodataset.org/#format-results) and run the following command on it:

```shell
encord-active import predictions --coco results.json
```

> ℹ️ Note
> Make sure that the annotation coordinates in the COCO results file are not normalized (not scaled into [0-1]).

After the execution is done, you are ready to [evaluate your model performance](https://docs.encord.com/docs/active-evaluate-detection-models).