---
sidebar_position: 1
---

# Working with Quality Metrics

**Overview and guide on how to work with Quality Metrics**

## Overview

Quality metrics constitute the foundation of Encord Active. They are additional parametrizations added onto your data, labels, and models; they are ways of indexing your data, labels, and models in semantically interesting and relevant ways.

Encord Active (EA) is designed to compute, store, inspect, manipulate, and utilize quality metrics for a wide array of functionality. It hosts a library of these quality metrics, and importantly allows you to customize by writing your own “Quality Metrics” to calculate/compute QMs across your dataset.

We have split the metrics into three main categories:

- **Data Quality Metrics:** For analysing and working with your image, sequence or video data. The metrics operate on images or individual video frames and are heuristic in the sense that they depend on the image content without labels.
  - Examples metrics: Area, Brightness, Blur, Green value.
  
- **Label Quality Metrics:** For analysing and working with your labels. The metrics can operate on the geometries of objects like bounding boxes, polygons, segmentations, and polylines and the heuristics of classifications.
  - Examples metrics: Object Aspect Ratio, Occlusion, Object Count.

- **Model Quality Metrics:** For analysing and working with your image and labels with an imported machine learning model. The metrics operate in various different ways, some are based on model predictions and other on active learning acquisition functions.
  - Examples metrics: Entropy, Smallest Margin, Least Confident.


  ## Guide to work with Quality Metrics

  _Coming soon_
