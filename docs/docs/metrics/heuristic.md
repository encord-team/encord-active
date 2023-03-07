# Heuristic

Work on images or individual video frames and are heuristic in the sense that they mostly depend on the image content without labels.

| Title                                                                                                                                                                                        | Metric Type   | Data Type                                                                                                          |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|--------------------------------------------------------------------------------------------------------------------|
| [Area](#area) - <small>Ranks images by their areawidth/height).</small>                                                                                                                      | `image`       |                                                                                                                    |
| [Aspect Ratio](#aspect-ratio) - <small>Ranks images by their aspect ratio (width/height).</small>                                                                                            | `image`       |                                                                                                                    |
| [Blue Values](#blue-values) - <small>Ranks images by how blue the average value of the image is.</small>                                                                                     | `image`       |                                                                                                                    |
| [Blur](#blur) - <small>Ranks images by their blurriness.</small>                                                                                                                             | `image`       |                                                                                                                    |
| [Brightness](#brightness) - <small>Ranks images by their brightness.</small>                                                                                                                 | `image`       |                                                                                                                    |
| [Contrast](#contrast) - <small>Ranks images by their contrast.</small>                                                                                                                       | `image`       |                                                                                                                    |
| [Green Values](#green-values) - <small>Ranks images by how green the average value of the image is.</small>                                                                                  | `image`       |                                                                                                                    |
| [Inconsistent Object Classification and Track IDs](#inconsistent-object-classification-and-track-ids) - <small>Looks for overlapping objects with different classes (across frames).</small> | `sequence`    | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Missing Objects and Broken Tracks](#missing-objects-and-broken-tracks) - <small>Identifies missing objects and broken tracks based on object overlaps.</small>                              | `sequence`    | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Object Count](#object-count) - <small>Counts number of objects in the image</small>                                                                                                         | `image`       | `bounding box`, `checklist`, `point`, `polygon`, `polyline`, `radio`, `rotatable bounding box`, `skeleton`, `text` |
| [Random Values on Images](#random-values-on-images) - <small>Assigns a random value between 0 and 1 to images</small>                                                                        | `image`       |                                                                                                                    |
| [Random Values on Objects](#random-values-on-objects) - <small>Assigns a random value between 0 and 1 to objects</small>                                                                     | `image`       | `bounding box`, `polygon`, `rotatable bounding box`                                                                |
| [Red Values](#red-values) - <small>Ranks images by how red the average value of the image is.</small>                                                                                        | `image`       |                                                                                                                    |
| [Sharpness](#sharpness) - <small>Ranks images by their sharpness.</small>                                                                                                                    | `image`       |                                                                                                                    |


## Area  
Ranks images by their area.

Area is computed as the product of image width and image height ($width \times height$).
      

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py)

## Aspect Ratio  
Ranks images by their aspect ratio.

Aspect ratio is computed as the ratio of image width to image height ($\frac{width}{height}$).
  

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py)

## Blue Values  
Ranks images by how blue the average value of the
                    image is.  

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py)

## Blur  
Ranks images by their blurriness.

Blurriness is computed by applying a Laplacian filter to each image and computing the
variance of the output. In short, the score computes "the amount of edges" in each
image. Note that this is $1 - \text{sharpness}$.

```python
score = 1 - cv2.Laplacian(image, cv2.CV_64F).var()
```
  

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py)

## Brightness  
Ranks images their brightness.

Brightness is computed as the average (normalized) pixel value across each image.
  

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py)

## Contrast  
Ranks images by their contrast.

Contrast is computed as the standard deviation of the pixel values.
  

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py)

## Green Values  
Ranks images by how green the average value of the
                    image is.  

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py)

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

## Object Count  
Counts number of objects in the image.  

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/object_counting.py)

## Random Values on Images  
Uses a uniform distribution to generate a value between 0 and 1 to each image  

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/random.py)

## Random Values on Objects  
Uses a uniform distribution to generate a value between 0 and 1 to each object  

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/random.py)

## Red Values  
Ranks images by how red the average value of the
                    image is.  

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py)

## Sharpness  
Ranks images by their sharpness.

Sharpness is computed by applying a Laplacian filter to each image and computing the
variance of the output. In short, the score computes "the amount of edges" in each
image.

```python
score = cv2.Laplacian(image, cv2.CV_64F).var()
```
  

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py)


