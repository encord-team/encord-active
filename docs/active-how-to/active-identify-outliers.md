---
title: "Finding data outliers"
slug: "active-identify-outliers"
hidden: true
metadata: 
  title: "Finding data outliers"
  description: "Discover data outliers with Encord Active: Identify outliers using Interquartile ranges. Streamline data analysis."
  image: 
    0: "https://files.readme.io/7ecdcf8-image_16.png"
createdAt: "2023-07-11T16:27:42.224Z"
updatedAt: "2023-08-09T12:56:55.246Z"
category: "6480a3981ed49107a7c6be36"
---

With Encord Active, you can quickly find data and label outliers for pre-defined metrics, custom metrics, and label classes. Encord Active finds outliers using precomputed Interquartile ranges.

## Setup

If you haven't installed Encord Active, visit [installation](https://docs.encord.com/docs/active-oss-install). In this workflow we will be using the BDD validation dataset.

## Data outliers

### 1. Find outliers

Navigate to the _Data Quality_ > _Summary_ tab. Here, the [Quality Metrics](https://docs.encord.com/docs/active-quality-metrics) will be presented as expandable panes.

Click on a metric to get deeper insight into _moderate outliers_ and _severe outliers_. The most severe outliers are presented first in the pane.

Use the slider to navigate your data from most severe outlier to least severe.

![data-quality-outliers.png](https://storage.cloud.google.com/docs-media.encord.com/static/img/images/data-quality-outliers.png)

### 2. Tag outliers

When you have identified outliers of interest, use the [tags](https://docs.encord.com/docs/active-tagging) or [bulk tagging](https://docs.encord.com/docs/active-tagging#bulk-tagging) feature to save a group of images.

![data-quality-outliers-tagging.png](https://storage.cloud.google.com/docs-media.encord.com/static/img/images/data-quality-outliers-tagging.png)

After creating a tagged <<glossary:image group>>, you can access it at the bottom of the left sidebar in the _Actions_ tab.

### 3. Act on outliers

Within the _Actions_ tab, click _Filter dataframe on_ and select _tags_. Next, choose the tags you would like to export, relabel, augment, review, or delete from your dataset.

![data-quality-outliers-action.png](https://storage.cloud.google.com/docs-media.encord.com/static/img/images/data-quality-outliers-action.png)

## Label outliers

### Steps

### 1. Find outliers

Navigate to the _Label Quality_ > _Summary_ tab. Here each [Quality Metric](https://docs.encord.com/docs/active-quality-metrics) will be presented as an expandable panes.

![label-quality-outliers.png](https://storage.cloud.google.com/docs-media.encord.com/static/img/images/label-quality-outliers.png)

You can click on a metric to get a deeper insight into _moderate outliers_ and _severe outliers_. Severe outliers are presented first in the pane.

### 2. Tag outliers

Next, you can use the slider to navigate your data from most severe outlier to least severe.

![label-quality-outliers-slider.png](https://storage.cloud.google.com/docs-media.encord.com/static/img/images/label-quality-outliers-slider.png)

When you have identified outliers of interest use the [tags](https://docs.encord.com/docs/active-tagging) or [bulk tagging](https://docs.encord.com/docs/active-tagging#bulk-tagging) feature to select a group of images. After creating a tagged <<glossary:image group>>, you can access it at the bottom of the left sidebar in the _Actions_ tab.

![label-quality-outliers-tagging.png](https://storage.cloud.google.com/docs-media.encord.com/static/img/images/label-quality-outliers-tagging.png)

Within the _Actions_ tab, click _Filter data frame on_ and select _tags_. Next, choose the tags you would like to export, relabel, augment, review, or delete from your dataset.

![label-quality-outliers-action.png](https://storage.cloud.google.com/docs-media.encord.com/static/img/images/label-quality-outliers-action.png)
