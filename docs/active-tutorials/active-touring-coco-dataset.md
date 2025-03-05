---
title: "Touring the COCO Sandbox dataset"
slug: "active-touring-coco-dataset"
hidden: false
metadata: 
  title: "Touring the COCO Sandbox dataset"
  description: "Explore Encord Active with COCO Sandbox dataset: Download, browse, flag errors, analyze metrics for model enhancement"
  image: 
    0: "https://files.readme.io/d3d8971-image_16.png"
category: "65a71bbfea7a3f005192d1a7"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/local_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

In this tutorial, you will see some cool features of Encord Active based on the Coco sandbox dataset.
You will go through the following steps:

1. [Downloading the dataset](#1-downloading-the-dataset)
2. [Opening Encord Active in your browser](#2-opening-encord-active-in-your-browser)
3. [Finding and flagging label errors](#3-finding-and-flagging-label-errors)
4. [Figuring out what metrics influence model performance](#4-figuring-out-what-metrics-influence-model-performance)

> ℹ️ Note
> This tutorial assumes that you have [installed](https://docs.encord.com/docs/active-oss-install) `encord-active`.

## 1. Downloading the dataset

Download the data by running this command

```shell
encord-active download
```

The script asks you to choose a project, navigate the options with <kbd>↑</kbd> and <kbd>↓</kbd> and hit <kbd>enter</kbd>.

Now `encord-active` will download your data.

## 2. Opening Encord Active in your browser

When the download process is done, follow the printed instructions to launch the app with the [start][ea-cli-start] CLI command:

```shell
cd /path/to/downloaded/project
encord-active start
```

> ℹ️ Note
> If the terminal seems stuck and nothing happens, try visiting http://localhost:8000 in your browser.


## 3. Finding and flagging label errors

You will carry out this process in two steps:

1. [Identifying metrics with label errors](#identifying-metrics-with-label-errors)
2. [Tagging label errors](#tagging-label-errors)

<!---
3. \[**Optional**\] [Exporting labels for relabeling](#exporting-labels-for-relabeling)
--->

### Identifying metrics with label errors

1. Open Encord Active (from the web-app or from a local installation):

   ![Encord Active Landing page](https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/active_landing-page-new.png)

   ![Encord Active Landing (Quickstart) page](https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/active_landing-page.png)

2. Select a project.

   > ℹ️ Note
   > If a project does not exist in Encord Active, create one (in the web-app) or import one.

Go to the _Summary_ > _Annotation_ page.
The page should look like this:

![Annotation Quality Summary Page](https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/active-quickstart-annotations-quality-summary.png)

On the Summary page, you will find all the outliers that Encord Active automatically found based on all the [metrics](https://docs.encord.com/docs/active-quality-metrics) that were computed for the labels.

Go to the _Explorer_ page.

Select "Annotation Duplicates" and scroll down the page.

The page should look similar to this:

![Annotation Duplicates](https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/active-annotation-duplicate.png)

The page shows how this metric was computed, how many outliers were found and some of the most severe outliers.

If you hover the mouse over the image with the orange, you can click the expand button as indicated here:

![Expand the image](https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/active-expand-image-oranges.png)

Clicking the button provides a larger view of the images and detailed information about the image.

<div style={{ textAlign: "center" }}>
  <img
    src="https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/duplicated-annotation-orange.jpeg"
    alt="Duplicated annotations"
    width={300}
  />
  <p>Notice the duplicated annotations.</p>
</div>

Hit <kbd>Esc</kbd> to exit the full screen view.

If you take a closer look at the annotations in the other displayed images, you will notice the same issue.

<div
  style={{
    display: "table",
    borderCollapse: "collapse",
    width: "100%",
    margin: "1em",
  }}
>
  <div style={{ textAlign: "center", display: "table-cell", padding: "5px" }}>
    <img
      src="https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/duplicated-annotation-skiing.jpeg"
      alt="Duplicated annotations"
      width={300}
      style={{ display: "block", width: "100%", height: "auto" }}
    />
  </div>
  <div style={{ textAlign: "center", display: "table-cell", padding: "5px" }}>
    <img
      src="https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/duplicated-annotation-broccoli.jpeg"
      alt="Duplicated annotations"
      width={300}
      style={{ display: "block", width: "100%", height: "auto" }}
    />
  </div>
  <div style={{ textAlign: "center", display: "table-cell", padding: "5px" }}>
    <img
      src="https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/duplicated-annotation-fridge.jpeg"
      alt="Duplicated annotations"
      width={300}
      style={{ display: "block", width: "100%", height: "auto" }}
    />
  </div>
</div>

> 👍 Tip
> You can find other sources of label errors by inspecting the other tabs. Good places to start could be the "Absolute Area" and "Aspect Ratio" label metrics.


### Tagging label errors

To tag the images with the identified label errors, select the images and click the TAG button and provide the name for the new tag.

![Add new tag](https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/active-label-duplicate-tag-add.png)

<!---

### Exporting labels for relabeling

Suppose you have now tagged all the duplicated annotations that you would like to export for relabeling.
Then, you go to the _Actions_ -> _Filter & Export_ tab in the left sidebar and filter by the tag that you created (as indicated by the purple arrows).

![Select images with the `label_duplication` tag](https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/export-duplication-tag.png)

_Note how your data has been filtered to only the rows that you tagged._

Now you can _Generate COCO file_, _Clone_ the data into a new Encord project, or send the data for _Review_, _Relabel_, or _Augmentation_ tasks.

--->

## 4. Figuring out what metrics influence model performance

Encord Active also allows you to figure out which metrics influence your model performance the most.
In this section, we'll go through a subset of those:

- [The high level view of model performance](#the-high-level-view-of-model-performance).
- [Inspecting model performance for a specific metric](#inspecting-model-performance-for-a-specific-metric).

### The high level view of model performance

#### mAP and mAR scores

The first section displays the _mean Average Precision (mAP)_,  _mean Average Recall (mAR)_, _true positive (TP)_, _false positive (FP)_, and _false negative (FN)_ of your model based on the IOU threshold set in the top of the page.

![mAP and mAR scored](https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/active-map-mar-fragment.png)

Dragging the IOU slider changes the scores.
You can also choose to see the aggregate score for certain classes by selecting them in the drop-down to the left.

#### Metric importance and correlation

Scrolling down the _Summary_ page, the importance and correlations of your model performance display as functions of metrics.

![Metric Importance](https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/active-importance-and-correlations.png)
![Metric  Correlation](https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/active-importance-and-correlations_02.png)

From this overview, you can see that, for example "Confidence" has a high importance for the model performance.

<!---
> ℹ️ Note
> The "(P)" and the "(F)" in the labels of the plots indicate whether the metric was run on predictions or frames, respectively.
--->

Next, we can jump to the _Metric Performance_ page and take a closer look at exactly how the model performance is affected by this metric. However, we want to show you the rest of this page prior to doing this.

You can skip straight ahead to the [Inspecting Model Performance for a Specific Metric](#inspecting-model-performance-for-a-specific-metric) if you are too curious to wait.

Before jumping into specific metrics, we want to show you the decomposition of the model performance based on individual classes. Scrolling down the _Summary_ page, the Per Class average precision, average recall, and precision recall curve scores for each individual class appears.

### Inspecting model performance for a specific metric

Using the _Metric Performance_ and _Explorer_ pages you can see how specific metrics affect the model performance:

1. Go to _Predictions_ > _Metric Performance_.
2. Select the "Confidence" metric from the _Metric_ drop-down list.

![Performance by Metric page](https://storage.googleapis.com/docs-media.encord.com/static/img/tutorials/active-performance-by-object-area-coco.png)

The plot shows the precision and the false negative rate as a function of the selected metric; "Confidence" in this case.

3. Go to _Predictions_ > _Explorer_.
4. Filter the data based on a data or prediction metric and the prediction outcome.

> ℹ️ Note
> Queries are only available in the web-app version of Active.

## Summary

This concludes the tour around Encord Active with the COCO Sandbox dataset. By now, you should have a good idea about how you can improve both your data, labels, and models by the insights you get from Encord Active.

## Next steps

- We've only covered each page in the app briefly in this tutorial.
- To learn more about concrete actionable steps you can take to improve your model performance, we suggest that you have a look at the [Workflow section](https://docs.encord.com/docs/annotate-workflows-and-templates).
- If you want to learn more about the existing metrics or want to build your own metric function, the [Quality Metrics section](https://docs.encord.com/docs/active-quality-metrics) is where you should continue reading.
- Finally, we have also included some in-depth descriptions the [Command Line Interface](https://docs.encord.com/docs/active-cli).


[ea-cli-start]: https://docs.encord.com/docs/active-cli#start