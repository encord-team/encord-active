---
sidebar_position: 3
---

# Label Quality Metrics

Label Quality Metrics operate on the geometries of objects like bounding boxes, polygons, and polylines.

| Title                                                                                                                                                                                        | Metric Type | Data Type                                                                                                          |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| ----------- | ------------------------------------------------------------------------------------------------------------------ |
| [Annotation Duplicates](#annotation-duplicates) - <small>Ranks annotations by how likely they are to represent the same object.</small>                                                      | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Annotation closeness to image borders](#annotation-closeness-to-image-borders) - <small>Ranks annotations by how close they are to image borders.</small>                                   | `image`     | `bounding box`, `point`, `polygon`, `polyline`, `rotatable bounding box`, `skeleton`                               |
| [Detect Occlusion in Video](#detect-occlusion-in-video) - <small>Tracks objects and detect outliers.</small>                                                                                 | `sequence`  | `bounding box`, `rotatable bounding box`                                                                           |
| [Frame object density](#frame-object-density) - <small>Computes the percentage of image area that's occupied by objects.</small>                                                             | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Image-level Annotation Quality](#image-level-annotation-quality) - <small>Compares image classifications against similar images.</small>                                                    | `image`     | `radio`                                                                                                            |
| [Inconsistent Object Classification and Track IDs](#inconsistent-object-classification-and-track-ids) - <small>Looks for overlapping objects with different classes (across frames).</small> | `sequence`  | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Missing Objects and Broken Tracks](#missing-objects-and-broken-tracks) - <small>Identifies missing objects and broken tracks based on object overlaps.</small>                              | `sequence`  | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Object Annotation Quality](#object-annotation-quality) - <small>Compares object annotations against similar image crops.</small>                                                            | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Object Area - Absolute](#object-area---absolute) - <small>Computes object area in amount of pixels.</small>                                                                                 | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Object Area - Relative](#object-area---relative) - <small>Computes object area as a percentage of total image area.</small>                                                                 | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Object Aspect Ratio](#object-aspect-ratio) - <small>Computes aspect ratios of objects.</small>                                                                                              | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Object Count](#object-count) - <small>Counts number of objects in the image.</small>                                                                                                        | `image`     | `bounding box`, `checklist`, `point`, `polygon`, `polyline`, `radio`, `rotatable bounding box`, `skeleton`, `text` |
| [Polygon Shape Similarity](#polygon-shape-similarity) - <small>Ranks objects by how similar they are to their instances in previous frames.</small>                                          | `sequence`  | `polygon`                                                                                                          |
| [Random Values on Objects](#random-values-on-objects) - <small>Assigns a random value between 0 and 1 to objects.</small>                                                                    | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Shape outlier detection](#shape-outlier-detection) - <small>Calculates potential outliers by polygon shape.</small>                                                                         | `image`     | `polygon`                                                                                                          |

## Annotation Duplicates

Ranks annotations by how likely they are to represent the same object.

> [Jaccard similarity coefficient](https://en.wikipedia.org/wiki/Jaccard_index)
> is used to measure closeness of two annotations.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/geometric/annotation_duplicates.py)

## Annotation closeness to image borders

This metric ranks annotations by how close they are to image borders.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/geometric/image_border_closeness.py)

## Detect Occlusion in Video

This metric collects information related to object size and aspect ratio for each track
and find outliers among them.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/geometric/occlusion_detection_video.py)

## Frame object density

Computes the percentage of image area that's occupied by objects.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/geometric/object_size.py)

## Image-level Annotation Quality

This metric creates embeddings from images. Then, these embeddings are used to build
nearest neighbor graph. Similar embeddings' classifications are compared against each other.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/semantic/img_classification_quality.py)

## Inconsistent Object Classification and Track IDs

This algorithm looks for overlapping objects in consecutive
frames that have different classes. Furthermore, if classes are the same for overlapping objects but have different
track-ids, they will be flagged as potential inconsistencies in tracks.

**Example 1:**

```
      Frame 1                       Frame 2
┌───────────────────┐        ┌───────────────────┐
│                   │        │                   │
│  ┌───────┐        │        │  ┌───────┐        │
│  │       │        │        │  │       │        │
│  │ CAT:1 │        │        │  │ DOG:1 │        │
│  │       │        │        │  │       │        │
│  └───────┘        │        │  └───────┘        │
│                   │        │                   │
└───────────────────┘        └───────────────────┘
```

`Dog:1` will be flagged as potentially wrong class, because it overlaps with `CAT:1`.

**Example 2:**

```
      Frame 1                       Frame 2
┌───────────────────┐        ┌───────────────────┐
│                   │        │                   │
│  ┌───────┐        │        │  ┌───────┐        │
│  │       │        │        │  │       │        │
│  │ CAT:1 │        │        │  │ CAT:2 │        │
│  │       │        │        │  │       │        │
│  └───────┘        │        │  └───────┘        │
│                   │        │                   │
└───────────────────┘        └───────────────────┘
```

`Cat:2` will be flagged as potentially having a broken track, because track ids `1` and `2` doesn't match.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/high_iou_changing_classes.py)

## Missing Objects and Broken Tracks

Identifies missing objects by comparing object overlaps based
on a running window.

**Case 1:**
If an intermediate frame (frame $i$) doesn't include an object in the same
region, as the two surrounding frames ($i-1$ and $i+1$), it is flagged.

```
      Frame i-1                     Frame i                      Frame i+1
┌───────────────────┐        ┌───────────────────┐        ┌───────────────────┐
│                   │        │                   │        │                   │
│  ┌───────┐        │        │                   │        │  ┌───────┐        │
│  │       │        │        │                   │        │  │       │        │
│  │ CAT:1 │        │        │                   │        │  │ CAT:1 │        │
│  │       │        │        │                   │        │  │       │        │
│  └───────┘        │        │                   │        │  └───────┘        │
│                   │        │                   │        │                   │
│                   │        │                   │        │                   │
└───────────────────┘        └───────────────────┘        └───────────────────┘
```

Frame $i$ will be flagged as potentially missing an object.

**Case 2:**
If objects of the same class overlap in three consecutive frames ($i-1$, $i$, and $i+1$) but do not share object
hash, they will be flagged as a potentially broken track.

```
      Frame i-1                     Frame i                      Frame i+1
┌───────────────────┐        ┌───────────────────┐        ┌───────────────────┐
│                   │        │                   │        │                   │
│  ┌───────┐        │        │  ┌───────┐        │        │  ┌───────┐        │
│  │       │        │        │  │       │        │        │  │       │        │
│  │ CAT:1 │        │        │  │ CAT:2 │        │        │  │ CAT:1 │        │
│  │       │        │        │  │       │        │        │  │       │        │
│  └───────┘        │        │  └───────┘        │        │  └───────┘        │
│                   │        │                   │        │                   │
│                   │        │                   │        │                   │
└───────────────────┘        └───────────────────┘        └───────────────────┘
```

`CAT:2` will be marked as potentially having a wrong track id.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/missing_objects_and_wrong_tracks.py)

## Object Annotation Quality

This metric transforms polygons into bounding boxes
and an embedding for each bounding box is extracted. Then, these embeddings are compared
with their neighbors. If the neighbors are annotated differently, a low score is given to it.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/semantic/img_object_quality.py)

## Object Area - Absolute

Computes object area in amount of pixels.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/geometric/object_size.py)

## Object Area - Relative

Computes object area as a percentage of total image area.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/geometric/object_size.py)

## Object Aspect Ratio

Computes aspect ratios ($width/height$) of objects.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/geometric/object_size.py)

## Object Count

Counts number of objects in the image.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/object_counting.py)

## Polygon Shape Similarity

Ranks objects by how similar they are to their instances in previous frames
based on [Hu moments](https://en.wikipedia.org/wiki/Image_moment). The more an object's shape changes,
the lower its score will be.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/geometric/hu_temporal.py)

## Random Values on Objects

Uses a uniform distribution to generate a value between 0 and 1 to each object

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/random.py)

## Shape outlier detection

Computes the Euclidean distance between the polygons'
[Hu moments](https://en.wikipedia.org/wiki/Image_moment) for each class and
the prototypical class moments.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/geometric/hu_static.py)
