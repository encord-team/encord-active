---
title: "Easy active learning on MNIST"
slug: "active-easy-active-learning-mnist"
hidden: false
metadata: 
  title: "Easy active learning on MNIST"
  description: "Tutorial: Integrate Random Forest in Encord Active for optimal data labeling. MNIST dataset guide. Train, rank, and sample efficiently."
  image: 
    0: "https://files.readme.io/bcab15c-image_16.png"
createdAt: "2023-07-11T16:27:42.048Z"
updatedAt: "2023-08-09T12:36:27.203Z"
category: "65a71bbfea7a3f005192d1a7"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/local_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

In this tutorial, you will see how to plug a Random Forest model in to Encord Active and use it to select the best data to label next on the MNIST dataset. You will go through the following steps:

1. [Download the MNIST sandbox project](#1-download-the-mnist-sandbox-project).
2. [Train the model with labeled data from the project](#2-train-the-model-with-labeled-data-from-the-project).
3. [Run the acquisition function powered by the model to rank the project data](#3-run-the-acquisition-function-powered-by-the-model-to-rank-the-project-data).
4. [Rank and sample the data to label next](#4-rank-and-sample-the-data-to-label-next).


> ℹ️ Note
> This tutorial assumes that you have [installed](https://docs.encord.com/docs/active-oss-install) `encord-active`.


## 1. Download the MNIST sandbox project

Download the data by running the following CLI command:

```shell
encord-active download --project-name "[open-source][test]-mnist-dataset"
```

When the process is done, the MNIST test dataset is ready to be used.

From now on, the tutorial is hands-on with python code, so we need a reference to the folder where the project was downloaded.

```python
from pathlib import Path
from encord_active.lib.project.project_file_structure import ProjectFileStructure

project_path = Path("/path/to/project/directory")
project_fs = ProjectFileStructure(project_path)
```

## 2. Train the model with labeled data from the project

It's a common scenario to start spinning the active learning cycle using a model trained with some initial data.
Let's select [sklearn.ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) as the base model.

```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 500)
```

We need to wrap the model with a `BaseModelWrapper` in order to interface the model's behaviour with the one expected by the acquisition functions.
The two main functionalities wrapped around the model are:

1. Prepare the input data to be ingested by the model (`prepare_data(..)`), and
2. Be able to obtain predicted probabilities of data samples (`_predict_proba(..)`).

Encord Active has a built-in wrapper for _scikit-learn_ classifiers (`SKLearnModelWrapper`), so let's use it.

```python
from typing import List
import numpy as np
from PIL import Image
from encord_active.lib.common.active_learning import get_data, get_data_hashes_from_project
from encord_active.lib.metrics.acquisition_functions import SKLearnModelWrapper

def transform_image_data(images: List[Image]) -> List[np.ndarray]:
    return [np.asarray(image).flatten() / 255 for image in images]

w_model = SKLearnModelWrapper(forest)

data_hashes = get_data_hashes_from_project(project_fs, subset_size=5000)
X, y = get_data(project_fs, data_hashes, class_name="digit")
X = transform_image_data(X)

w_model._model.fit(X, y)
```

## 3. Run the acquisition function powered by the model to rank the project data

Encord Active provides multiple acquisition functions ready to be used with the wrapped model.

We use an acquisition function called `Entropy` that measures the average level of “uncertainty” in the model's predicted probabilities.
The higher the entropy, the more “uncertain” the model.

```python
from encord_active.lib.common.active_learning import get_metric_results
from encord_active.lib.metrics.acquisition_functions import Entropy
from encord_active.lib.metrics.execute import execute_metrics

acq_func = Entropy(w_model)

execute_metrics([acq_func], data_dir=project_fs.project_dir, use_cache_only=True)

acq_func_results = get_metric_results(project_fs, acq_func)
```

> ℹ️ Note
> We use `Entropy` in this tutorial but Encord Active has multiple acquisition functions that can be inspected [here](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/acquisition_metrics/acquisition_functions.py).


## 4. Rank and sample the data to label next

As soon as the acquisition function finishes its execution through all the data samples, we proceed to rank them.

```python
from encord_active.lib.common.active_learning import get_n_best_ranked_data_samples

batch_size_to_label = 100 # amount of data samples selected to label next
data_to_label_next, scores = get_n_best_ranked_data_samples(
    acq_func_results,
    batch_size_to_label,
    rank_by="desc",
    exclude_data_hashes=data_hashes)
```

The output variable `data_to_label_next` contains the hashes of the best ranked data samples.
Now you can proceed to label these samples and enable your own active learning pipeline.

## Summary

This section concludes the end-to-end example on easy active learning on the MNIST dataset using Random Forest. We covered training a Random Forest model, wrapping the model to match Encord Active requirements on models, selecting and running acquisition functions over the data, and choosing the best data to label next.

Now, you should have a good idea about how Encord Active can be used to run your active learning pipeline while enabling smart selection of the data for labeling.