---
title: "Label quality metrics"
slug: "active-label-quality-metrics"
hidden: false
metadata: 
  title: "Label quality metrics"
  description: "Enhance label quality with geometries in Encord Active. Optimize annotations for bounding boxes, polygons, and polylines."
  image: 
    0: "https://files.readme.io/9b2a03a-image_16.png"
createdAt: "2023-07-21T09:09:02.139Z"
updatedAt: "2023-08-11T13:41:50.217Z"
category: "6480a3981ed49107a7c6be36"
---

Label quality metrics operate on the geometries of objects like <<glossary:bounding box>>es, <<glossary:polygon>>s and <<glossary:polyline>>s.

## Access Label Quality Metrics

Label Quality Metrics are used for sorting data, filtering data, and data analytics.

<!---

| Title                                                                                                                                                                                        | Metric Type | Data Type                                                                                                          |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|--------------------------------------------------------------------------------------------------------------------|
| [Absolute Area](#absolute-area) - <small>Computes object size in amount of pixels.</small>                                                                                 | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Aspect Ratio](#aspect-ratio) - <small>Computes aspect ratios of objects.</small>                                                                                              | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Broken Object Tracks](#broken-object-tracks) - <small>Identifies broken object tracks based on object overlaps.</small>                              | `sequence`  | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Border Proximity](#border-proximity) - <small>Ranks annotations by how close they are to image borders.</small>                                   | `image`     | `bounding box`, `point`, `polygon`, `polyline`, `rotatable bounding box`, `skeleton`                               |
| [Classification Quality](#classification-quality) - <small>Compares image classifications against similar images.</small>                                                    | `image`     | `radio`                                                                                                            |
| [Inconsistent Object Class](#inconsistent-object-class) - <small>Looks for overlapping objects with different classes (across frames).</small> | `sequence`  | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Inconsistent Track ID](#inconsistent-track-id) - <small>Looks for overlapping objects with different track-ids (across frames).</small> | `sequence`  | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Label Duplicates](#label-duplicates) - <small>Ranks labels by how likely they are to represent the same object.</small>                                                      | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Missing Objects](#missing-objects) - <small>Identifies missing objects based on object overlaps.</small>                              | `sequence`  | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Object Classification Quality](#object-classification-quality) - <small>Compares object annotations against similar image crops.</small>                                                            | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Occlusion Risk](#occlusion-risk) - <small>Tracks objects and detect outliers in videos.</small>                                                                                 | `sequence`  | `bounding box`, `rotatable bounding box`                                                                           |
| [Polygon Shape Anomaly](#polygon-shape-anomaly) - <small>Calculates potential outliers by polygon shape.</small>                                                                         | `image`     | `polygon`                                                                                                          |
| [Polygon Shape Similarity (video)](#FIX THIS LINK) - <small>Ranks objects by how similar they are to their instances in previous frames of a video.</small>                                          | `sequence`  | `polygon`                                                                                                          |
| [Randomize Objects](#randomize-objects) - <small>Assigns a random value between 0 and 1 to objects.</small>                                                                    | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Relative Area](#relative-object-size) - <small>Computes object size as a percentage of total image size.</small>                                                                 | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |

--->

| Title                                                                                                                                                                                        | Metric Type | Ontology Type                                                                                                          |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|--------------------------------------------------------------------------------------------------------------------|
| [Absolute Area](#absolute-area) - <small>Computes object size in amount of pixels.</small>                                                                                 | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Aspect Ratio](#aspect-ratio) - <small>Computes aspect ratios of objects.</small>                                                                                              | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Blue Value](#blue-value) - <small>Ranks annotated objects by how blue the average value of the object is.</small>                                                                                 | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Brightness](#brightness) - <small>Ranks annotated objects by their brightness.</small>                                                                                 | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Border Proximity](#border-proximity) - <small>Ranks annotations by how close they are to image borders.</small>                                   | `image`     | `bounding box`, `point`, `polygon`, `polyline`, `rotatable bounding box`, `skeleton`                               |
| [Broken Object Tracks](#broken-object-tracks) - <small>Identifies broken object tracks based on object overlaps.</small>                              | `sequence`, `video`  | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Brightness](#brightness) - <small>Ranks annotated objects by their brightness.</small>                                                                                 | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Confidence](#confidence) - <small>The confidence that an object was annotated correctly.</small>                                                                                 | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Contrast](#contrast) - <small>Ranks annotated objects by their contrast.</small>                                                                                 | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Classification Quality](#classification-quality) - <small>Compares image classifications against similar images.</small>                                                    | `image`     | `radio`                                                                                                            |
| [Green Value](#green-value) - <small>Ranks annotated objects by how green the average value of the object is.</small>                                                                                 | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Height](#height) - <small>Ranks annotated objects by the height of the object.</small>                                                                                 | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Inconsistent Object Class](#inconsistent-object-class) - <small>Looks for overlapping objects with different classes (across frames).</small> | `sequence`, `video`  | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Inconsistent Track ID](#inconsistent-track-id) - <small>Looks for overlapping objects with different track-ids (across frames).</small> | `sequence`, `video`  | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Label Duplicates](#label-duplicates) - <small>Ranks labels by how likely they are to represent the same object.</small>                                                      | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Missing Objects](#missing-objects) - <small>Identifies missing objects based on object overlaps.</small>                              | `sequence`, `video`  | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Object Classification Quality](#object-classification-quality) - <small>Compares object annotations against similar image crops.</small>                                                            | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Occlusion Risk](#occlusion-risk) - <small>Tracks objects and detect outliers in videos.</small>                                                                                 | `sequence`, `video`  | `bounding box`, `rotatable bounding box`                                                                           |
| [Polygon Shape Anomaly](#polygon-shape-anomaly) - <small>Calculates potential outliers by polygon shape.</small>                                                                         | `image`     | `polygon`                                                                                                          |
| [Randomize Objects](#randomize-objects) - <small>Assigns a random value between 0 and 1 to objects.</small>                                                                    | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Red Value](#red-value) - <small>Ranks annotated objects by how red the average value of the object is.</small>                                                                                 | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Relative Area](#relative-area) - <small>Computes object size as a percentage of total image size.</small>                                                                 | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Sharpness](#sharpness) - <small>Ranks annotated objects by their sharpness.</small>                                                                                 | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Width](#width) - <small>Ranks annotated objects by the width of the object.</small>                                                                                 | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |

**To access Label Quality Metrics for Explorer:**

1. Click a Project from the _Active_ home page.

2. Click **Explorer**.

3. Click **Labels**.

4. Sort and filter the tabular data.

5. Click the plot diagram icon.

6. Sort and filter the embedding plot data.

**To access Label Quality Metrics for analytics:**

1. Click a Project from the _Active_ home page.

2. Click **Analytics**.

3. Click **Annotations**.

4. Select the quality metric you want to view from the _2D Metrics view_ or _Metrics Distribution_ graphs.

## Absolute Area

Computes object size in amount of pixels.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/geometric/object_size.py).

## Aspect Ratio

Computes aspect ratios (**width/height**) of objects. 

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/geometric/object_size.py).

## Blue Value  

Ranks annotated objects by how blue the average value of the object is.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py).

## Brightness  
Ranks annotated objects by their brightness. Brightness is computed as the average (normalized) pixel value across each object. 

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py).

## Broken Object Tracks

Identifies broken object tracks by comparing object overlaps based on a running window.

**Example:**

If objects of the same class overlap in three consecutive frames (_i-1_, _i_, and _i+1_) but do not share object hash, the frames are flagged as a potentially broken track.

![Broken Object Tracks example](https://storage.googleapis.com/docs-media.encord.com/static/img/active/filters-labels-metrics/broken-object-tracks.PNG)

`CAT:2` is marked as potentially having a wrong track id.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/missing_objects_and_wrong_tracks.py).

## Border Proximity

This metric ranks annotations by how close they are to image borders.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/geometric/image_border_closeness.py).

## Confidence

The confidence score (α) is a measure of a machine learning model's certainty that a given prediction is accurate. The higher the confidence score, the more certain a model is about its prediction.

Manual labels are always assigned α = 100%, while label predictions created using models and automated methods such as interpolation have a confidence score below 100% (α < 100%).

Values for this metric are calculated as labels are fetched from Annotate.

> ℹ️ Note
> While arguably not making much sense when annotated by a human, this value is very important for objects that were automatically labeled.

## Contrast

Ranks annotated objects by their contrast. Contrast is computed as the standard deviation of the pixel values. 

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py).

## Classification Quality

This metric creates embeddings from images. Then, these embeddings are used to build nearest neighbor graph. Similar embeddings' classifications are compared against each other.

We calculate the embeddings of each image, (for example, change 3xNxM dimensional images to 1xD dimensional vectors using a neural network architecture). Then for each embedding (or image) we look at the **50** nearest neighbors and compare its annotation with the neighboring annotations. 

For example, let's say the current image is annotated as **A** but only _20_ out of _50_ of its neighbors are also annotated as **A**. The rest are annotated differently. That gives us a score of _20/50_ = _0.4_. A score of 1 means that the annotation is very reliable because very similar images are annotated the same. As the score gets closer to the zero, the annotation is not reliable.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/semantic/img_classification_quality.py).

## Green Value  
Ranks annotated objects by how green the average value of the object is.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py).

## Height 
Ranks annotated objects by the height of the object.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py).

## Inconsistent Object Class

This algorithm looks for overlapping objects in consecutive frames that have different classes.

**Example:**

![Inconsistent Object Class example](https://storage.googleapis.com/docs-media.encord.com/static/img/active/filters-labels-metrics/inconsistent-object-track.PNG)

`Dog:1` is flagged as potentially the wrong class, because `Dog:1` overlaps with `CAT:1`.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/high_iou_changing_classes.py).

## Inconsistent Track ID

This algorithm looks for overlapping objects with different track-ids. Overlapping objects with different track-ids are flagged as potential inconsistencies in tracks.

**Example:**

![Inconsistent Track ID example](https://storage.googleapis.com/docs-media.encord.com/static/img/active/filters-labels-metrics/inconsistent-track-id.PNG)

`Cat:2` is flagged as potentially having a broken track, because track ids `1` and `2` do not match.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/high_iou_changing_classes.py).

## Label Duplicates

Ranks labels by how likely they are to represent the same object.

> [Jaccard similarity coefficient](https://en.wikipedia.org/wiki/Jaccard_index) is used to measure closeness of two annotations.

**Example 1:**

An annotator accidentally labels the same thing in a frame twice.

An annotator labeled the same orange twice in a frame. Look carefully at both images and you can see that there are two slightly different labels around the orange.

![Duplicate labels example 1](https://storage.googleapis.com/docs-media.encord.com/static/img/active/filters-labels-metrics/label_duplicates_01.png)


**Example 2:**

Sometimes the same type of things in a frame are very close to each other and the annotator does not know if the things should be annotated separately or as a group so they do both. Or perhaps the annotator labels all the things in a group and sometimes they label each individual thing, or they label the group and each individual thing in the group.

An annotator labeled a group of oranges and then labeled individual oranges in the group.

![Duplicate labels example 2](https://storage.googleapis.com/docs-media.encord.com/static/img/active/filters-labels-metrics/label_duplicates_02.png)

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/geometric/annotation_duplicates.py)

## Missing Objects

Identifies missing objects by comparing object overlaps based on a running window.

**Example:**

If an intermediate frame (frame _i_) does not include an object in the same region, as the two surrounding frames (_i-1_ and _i+1_), the frame is flagged.

![Missing Objects example](https://storage.googleapis.com/docs-media.encord.com/static/img/active/filters-labels-metrics/missing-objects.PNG)

Frame _i_ is flagged as potentially missing an object.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/missing_objects_and_wrong_tracks.py).

## Object Classification Quality

This metric transforms polygons into bounding boxes and an embedding for each <<glossary:bounding box>> is extracted. Then, these embeddings are compared with their neighbors. If the neighbors are annotated/classified differently, a low score is given to the classification.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/semantic/img_object_quality.py).

## Occlusion Risk

This metric collects information related to object size and aspect ratio for each video and finds outliers among them.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/geometric/occlusion_detection_video.py).

## Polygon Shape Anomaly

Computes the Euclidean distance between the polygons' [Hu moments](https://en.wikipedia.org/wiki/Image_moment) for each class and the prototypical class moments.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/geometric/_hu_static.py).

<!---
## Polygon Shape Similarity (video)

Ranks objects by how similar they are to their instances in previous frames (of a video) based on [Hu moments](https://en.wikipedia.org/wiki/Image_moment). The more an object's shape changes, the lower the object's score.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/geometric/hu_temporal.py).
--->

## Red Value
Ranks annotated objects by how red the average value of the object is.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py).

## Relative Area

Computes object size as a percentage of total image size.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/geometric/object_size.py).

## Randomize Objects

Uses a uniform distribution to generate a value between 0 and 1 to each object

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/random.py).

## Sharpness
Ranks annotated objects by their sharpness.

Sharpness is computed by applying a Laplacian filter to each annotated object and computing the variance of the output. In short, the score computes "the amount of edges" in each annotated object.

```python
score = cv2.Laplacian(image, cv2.CV_64F).var()
```

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py).

## Width 
Ranks annotated objects by the width of the object.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py).