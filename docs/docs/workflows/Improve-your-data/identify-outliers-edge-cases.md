---
sidebar_position: 3
---

# Find Outliers

**Find outliers in your dataset using Encord Active's Data Quality tab**


With Encord Active, you can quickly find image outliers for pre-defined metrics, custom metrics, and label classes. 
Encord Active finds outliers using precomputed [Interquartile ranges](/pages/data-quality/summary).

 `Prerequisites:` Dataset  

### Setup
If you haven't installed Encord Active, run:

```shell
python3.9 -m venv ea-venv
source ea-venv/bin/activate
# within venv
pip install encord-active
```

In this workflow we will be using the BDD validation dataset:

```shell
# run download command
encord-active download
Loading prebuilt projects ...
[?] Choose a project: [open-source][validation]-bdd-dataset (229.8 mb)
   [open-source][validation]-coco-2017-dataset (1145.2 mb)
   [open-source][test]-limuc-ulcerative-colitis-classification (316.0 mb)
   [open-source]-covid-19-segmentations (55.6 mb)
 > [open-source][validation]-bdd-dataset (229.8 mb)
```

After downloading the dataset we visualise it:
```shell
# open the UI
cd path/to/[open-source][validation]-bdd-dataset
encord-active visualise
```

## Steps

### 1. Find outliers
Navigate to the _Data Quality_ > _Summary_ tab. Here, the [metrics](/category/metrics) will be presented as expandable panes. 

Click on a metric to get deeper insight into _moderate outliers_ and _severe outliers_. The most severe outliers are presented first in the pane.

Use the slider to navigate your data from most severe outlier to least severe.

![data-quality-outliers.png](../../images/data-quality-outliers.png)

### 2. Tag outliers
When you have identified outliers of interest, use the [tagging](/tags) or [bulk tagging](/tags) feature to save a group of images.

![data-quality-outliers-tagging.png](../../images/data-quality-outliers-tagging.png)

After creating a tagged image group, you can access it at the bottom of the left sidebar in the _Actions_ tab.

### 3. Act on outliers
Within the _Actions_ tab, click _Filter dataframe on_ and select _tags_. Next, choose the tags you would like to export, relabel, or delete from your dataset.

   * To generate a new COCO file click _Generate COCO file_.
   * To export images in CSV click _download filtered data_.
   * To review images from your dataset click _Review_.
   * To relabel images please contact Encord to hear more.
   * To augment similar images please contact Encord to hear more.

![data-quality-outliers-action.png](../../images/data-quality-outliers-action.png)

