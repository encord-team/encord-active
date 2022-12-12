# Geometric

Operate on the geometries of objects like bounding boxes, polygons, and polylines.

| Title                                                                                                                                                      | Metric Type | Data Type                                                  |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ---------------------------------------------------------- |
| [Annotation Duplicates](#annotation-duplicates) - <small>Ranks annotations by how likely they are to represent the same object</small>                     | `image`     | `bounding box`, `polygon`                                  |
| [Annotation closeness to image borders](#annotation-closeness-to-image-borders) - <small>Ranks annotations by how close they are to image borders.</small> | `image`     | `bounding box`, `point`, `polygon`, `polyline`, `skeleton` |
| [Detect Occlusion in Video](#detect-occlusion-in-video) - <small>Tracks objects and detect outliers</small>                                                | `sequence`  | `bounding box`                                             |
| [Frame object density](#frame-object-density) - <small>Computes the percentage of image area that's occupied by objects</small>                            | `image`     | `bounding box`, `polygon`                                  |
| [Object Area - Absolute](#object-area---absolute) - <small>Computes object area in amount of pixels</small>                                                | `image`     | `bounding box`, `polygon`                                  |
| [Object Area - Relative](#object-area---relative) - <small>Computes object area as a percentage of total image area</small>                                | `image`     | `bounding box`, `polygon`                                  |
| [Object Aspect Ratio](#object-aspect-ratio) - <small>Computes aspect ratios of objects</small>                                                             | `image`     | `bounding box`, `polygon`                                  |
| [Polygon Shape Similarity](#polygon-shape-similarity) - <small>Ranks objects by how similar they are to their instances in previous frames.</small>        | `sequence`  | `polygon`                                                  |
| [Shape outlier detection](#shape-outlier-detection) - <small>Calculates potential outliers by polygon shape.</small>                                       | `image`     | `polygon`                                                  |

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

## Object Area - Absolute

Computes object area in amount of pixels.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/geometric/object_size.py)

## Object Area - Relative

Computes object area as a percentage of total image area.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/geometric/object_size.py)

## Object Aspect Ratio

Computes aspect ratios ($width/height$) of objects.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/geometric/object_size.py)

## Polygon Shape Similarity

This metric ranks objects by how similar they are to their instances in previous frames
based on [Hu moments](https://en.wikipedia.org/wiki/Image_moment). The more an object's shape changes,
the lower its score will be.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/geometric/hu_temporal.py)

## Shape outlier detection

Computes the Euclidean distance between the polygons'
[Hu moments](https://en.wikipedia.org/wiki/Image_moment) for each class and
the prototypical class moments.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/geometric/hu_static.py)
