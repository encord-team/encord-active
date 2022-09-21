# Heuristic

Work on images or individual video frames and are heuristic in the sense that they mostly depend on the image content without labels.

| Title                                                                                                                                                                                        | Metric Type | Data Type                 |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| ----------- | ------------------------- |
| [Area](#area) - <small>Ranks images by their area (width\*height).</small>                                                                                                                   | `image`     | `Any`                     |
| [Aspect Ratio](#aspect-ratio) - <small>Ranks images by their aspect ratio (width/height).</small>                                                                                            | `image`     | `Any`                     |
| [Blue Values](#blue-values) - <small>Ranks images by how much blue is present in the image.</small>                                                                                          | `image`     | `Any`                     |
| [Blur](#blur) - <small>Ranks images by their blurriness.</small>                                                                                                                             | `image`     | `Any`                     |
| [Brightness](#brightness) - <small>Ranks images by their brightness.</small>                                                                                                                 | `image`     | `Any`                     |
| [Contrast](#contrast) - <small>Ranks images by their contrast.</small>                                                                                                                       | `image`     | `Any`                     |
| [Green Values](#green-values) - <small>Ranks images by how much green is present in the image.</small>                                                                                       | `image`     | `Any`                     |
| [Inconsistent Object Classification and Track IDs](#inconsistent-object-classification-and-track-ids) - <small>Looks for overlapping objects with different classes (across frames).</small> | `sequence`  | `bounding box`, `polygon` |
| [Missing Objects and Broken Tracks](#missing-objects-and-broken-tracks) - <small>Identifies missing objects and broken tracks based on object overlaps.</small>                              | `sequence`  | `bounding box`, `polygon` |
| [Object Count](#object-count) - <small>Counts number of objects in the image</small>                                                                                                         | `image`     | `Any`                     |
| [Red Values](#red-values) - <small>Ranks images by how much red is present in the image.</small>                                                                                             | `image`     | `Any`                     |
| [Sharpness](#sharpness) - <small>Ranks images by their sharpness.</small>                                                                                                                    | `image`     | `Any`                     |

## Area

This metric ranks images by their area (width\*height).

Area is computed as the product of image width and image height.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/encord_active.metrics/heuristic/img_features.py)

## Aspect Ratio

This metric ranks images by their aspect ratio (width/height).

Aspect ratio is computed as the ratio of image width to image height.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/encord_active.metrics/heuristic/img_features.py)

## Blue Values

This metric ranks images by how much blue is present in the image.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/encord_active.metrics/heuristic/img_features.py)

## Blur

This metric ranks images by their blurriness.

Blurriness is computed by applying a Laplacian filter to each image and computing the
variance of the output. In short, the score computes "the amount of edges" in each
image. Note that this is $1 - \text{sharpness}$.

```python
score = 1 - cv2.Laplacian(image, cv2.CV_64F).var()
```

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/encord_active.metrics/heuristic/img_features.py)

## Brightness

This metric ranks images by their brightness.

Brightness is computed as the average (normalized) pixel value across each image.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/encord_active.metrics/heuristic/img_features.py)

## Contrast

This metric ranks images by their contrast.

Contrast is computed as the standard deviation of the pixel values.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/encord_active.metrics/heuristic/img_features.py)

## Green Values

This metric Ranks images by how much blue is present in the image.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/encord_active.metrics/heuristic/img_features.py)

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

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/encord_active.metrics/heuristic/high_iou_changing_classes.py)

## Missing Objects and Broken Tracks

Identifies missing objects by comparing object overlaps based
on a running window.

**Case 1:**
If an intermediate frame (frame $i$) doesn't include an object in the same
region as in the two surrounding frames ($i-1$ and $i+1$), it is flagged.

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

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/encord_active.metrics/heuristic/missing_objects_and_wrong_tracks.py)

## Object Count

Counts number of objects in the image.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/encord_active.metrics/heuristic/object_counting.py)

## Red Values

This metric ranks images by how much red is present in the image.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/encord_active.metrics/heuristic/img_features.py)

## Sharpness

This metric ranks images by their sharpness.

Sharpness is computed by applying a Laplacian filter to each image and computing the
variance of the output. In short, the score computes "the amount of edges" in each
image.

```python
score = cv2.Laplacian(image, cv2.CV_64F).var()
```

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/encord_active.metrics/heuristic/img_features.py)
