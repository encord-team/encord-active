---
title: "Touring the Quickstart dataset"
slug: "active-touring-quickstart-dataset"
hidden: false
metadata: 
  title: "Touring the Quickstart dataset"
  description: "Comprehensive tutorials: Encord Active use cases - MNIST active learning, COCO & Quickstart dataset tours, diversity sampling."
  image: 
    0: "https://files.readme.io/af10c8f-image_16.png"
createdAt: "2023-07-11T16:27:42.034Z"
updatedAt: "2023-08-11T13:50:23.892Z"
category: "65a71bbfea7a3f005192d1a7"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/local_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

In this tutorial, we will dive into the quickstart dataset and show you some cool features of Encord Active. You will go through the following steps:

1. [Opening the quickstart dataset](#1-opening-the-quickstart-dataset).
2. [Finding and tagging outliers](#2-finding-and-tagging-outliers).
3. [Figuring out what metrics influence model performance](#3-figuring-out-what-metrics-influence-model-performance).


> ℹ️ Note
> This tutorial assumes that you have [installed](https://docs.encord.com/docs/active-oss-install) `encord-active`.


## 1. Opening the quickstart dataset

To open the quickstart dataset run:

```shell
encord-active quickstart
```

Encord Active downloads the dataset and opens the UI in your browser.

![Encord Active Landing page](https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/active_landing-page.png)

> ℹ️ Note
> If the terminal just seems to get stuck and nothing happens, try visiting http://localhost:8000 in your browser.


### About the dataset

The Quickstart dataset contains images and labels for 200 random samples from the [COCO 2017 validation set](https://cocodataset.org/#download) with a pre-trained [MASK R-CNN RESNET50 FPN V2](https://arxiv.org/abs/1703.06870) model.

## 2. Finding and tagging outliers

First, we will find and tag image outliers.

### Identifying metrics with outliers

When you open Encord Active, you will start on the landing page.

Click the `quickstart` project. The "Summary" page for the project appears:

![Data Quality Summary Page](https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/active-quickstart-data-quality-summary.png)

On the Summary > Data page, you can see all the data outliers that Encord Active automatically found based on all the [Quality Metrics](https://docs.encord.com/docs/active-quality-metrics) that were computed for the images.

> 👍 Tip
> You check the metric of distribution for outliers using the Metrics Distribution graph and selecting the outlier from the drop-down. Good places to start could be the "Brightness" and "Sharpness" entries.

On the Summary > Annotations page, you can see annotation outliers that Encord Active automatically found.

![Annotation Quality Summary Page](https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/active-quickstart-annotations-quality-summary.png)

### Tagging outliers

To tag an image identified as an outlier, go to the Explorer page and select one or more images. The *TAG* button is enabled. Click the *TAG* button and specify a new tag.

Once the tag is created, you can add the tag to the images by selecting images and clicking the *TAG* button and selecting a tag from the list of tags.

![Tag an image](https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/active-quickstart-data-quality-tagging.png)

> 👍 Tip
> Use the Explorer scatter plot graph, the filter feature (click the *FILTERS* button repeatedly to add multiple filters including tags or labels) and queries (queries are only available from the web-app) to specify the images you want to tag.
> ![Filters](https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/active-quickstart-data-quality-tagging_02.png)

Once you are satisfied with your tagged subset of images, you can move on to exporting.

> ℹ️ Note
> Multiple subsets can be created in the web-app.

<!--- 

## 3. Exporting samples for relabeling or other actions

Suppose you have now tagged all the images that you would like to export for relabeling or other actions.
Then, you go to the _Actions_ -> _Filter & Export_ tab in the left sidebar and filter by the tag that you created.

![Select images with the `label_duplication` tag](https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/quickstart-data-quality-export.png)

_Note how your data has been filtered to only the rows that you tagged._

Now you can _Generate COCO file_, _Clone_ the data into a new Encord project, or send the data for _Review_, _Relabel_, or _Augmentation_ tasks. 

--->

## 3. Figuring out what metrics influence model performance

Encord Active also allows you to figure out which metrics influence your model performance the most.
In this section, we'll go through a subset of those:

- [The high level view of model performance](#the-high-level-view-of-model-performance).
- [Inspecting model performance for a specific metric](#inspecting-model-performance-for-a-specific-metric).

### The high level view of model performance

#### mAP and mAR scores

First, navigate to the _Predictions_ > _Summary_ page where you find multiple insights into your model performance.

The first section displays the _mean Average Precision (mAP)_,  _mean Average Recall (mAR)_, _true positive (TP)_, _false positive (FP)_, and _false negative (FN)_ of your model based on the IOU threshold set in the top of the page.

![mAP and mAR scored](https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/active-quickstart-metrics.png)

Dragging the _IOU_ slider changes the scores.
You can also choose to see the aggregate score for certain classes by selecting them in the drop-down to the left.

#### Metric importance and correlation

Scrolling down the _Summary_ page, the importances and correlations of your model performance display as functions of metrics.

![Metric Importances](https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/active-quickstart-metrics-importance.png)

![Metric Correlation](https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/active-quickstart-metrics-importance.png)

From this overview, you can see that, for example "Confidence" has a high importance for the model performance.

<!---

> ℹ️ Note
> The "(P)" and the "(F)" in the labels of the plots indicate whether the metric was run on predictions or frames, respectively.

--->


Next, we can jump to the _Metric Performance_ page and take a closer look at exactly how the model performance is affected by this metric. However, we want to show you the rest of this page prior to doing this.

You can skip straight ahead to the [Inspecting Model Performance for a Specific Metric](#inspecting-model-performance-for-a-specific-metric) if you are too curious to wait.

Before jumping into specific metrics, we want to show you the decomposition of the model performance based on individual classes. Scrolling down the _Summary_ page, the Per Class average precision, average recall, and precision recall curve scores for each individual class appears.

<!---
![AP and AR scores decomposed based on classes](https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/quickstart-metrics-subset.png)

> ℹ️ Note
> We've chosen a subset of classes [`person`, `cat`, `dog`, `bus`] in the settings in the top of the page to make the plots more digestible.


- From the left plot (Mean scores), we can see the AP and the AR for each individual class.
- The right plot shows the precision-recall curves for each of the classes.
- From these plots, we learn that the model performs better on the `dog` class than it does on, e.g., the `bus` class.
- To learn for which instances the model works and for which it doesn't, you can look in the _True Positives_, _False Positives_, and _False Negatives_ tabs to see concrete instances of the three success/failure modes.

--->

### Inspecting model performance for a specific metric

Using the _Metric Performance_ and _Explorer_ pages you can see how specific metrics affect the model performance:

1. Go to _Predictions_ > _Metric Performance_.
2. Select the "Confidence" metric from the _Metric_ drop-down list.

![Performance by Metric page](https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/active_performance-metric-confidence.png)

The plot shows the precision and the false negative rate as a function of the selected metric; "Confidence" in this case.

3. Go to _Predictions_ > _Explorer_.
4. Filter the data based on a data or prediction metric and the prediction outcome.

> ℹ️ Note
> Queries are only available in the web-app version of Active.


<!---

We can see how when the model predictions are small in terms of the absolute area, then the true-positive rate is low (bad), while larger predictions are more often correct (good).

Similarly, when labels are small, the false negative rate is high (bad), while larger labels are less likely to be issued by the model (good).

The performance graphs can be showed by each metric and are easy to interact with using the cursor and scroll function:

![Performance by Metric page gif](https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/quickstart-performance-by-metric.gif)

--->

## Summary

This concludes the tour of the quickstart dataset. In this tutorial we covered opening the quickstart dataset, finding image outliers, and analysing the performance of an off-the-shelf object detection model on the dataset. By now, you should have a good idea about how Encord Active can be used to understand your data, labels, and model.

### Next steps

- We have only covered a few of the page in the app briefly in this tutorial. 
- To learn more about concrete actionable steps you can take to improve your model performance, we suggest that you have a look at the [Workflow section](https://docs.encord.com/docs/annotate-workflows-and-templates).
- If you want to learn more about the existing metrics or want to build your own metric function, the [Metrics section](https://docs.encord.com/docs/active-quality-metrics) is where you should continue reading.
- Finally, we have also included some in-depth descriptions the [Command Line Interface](https://docs.encord.com/docs/active-cli).