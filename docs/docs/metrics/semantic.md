# Semantic

Operates with the semantic information of images or individual video frames.

| Title                                                                                                                                    | Metric Type | Data Type                 |
| ---------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ------------------------- |
| [Image-level Annotation Quality](#image-level-annotation-quality) - <small>Compares image classifications against similar images</small> | `image`     | `radio`                   |
| [Object Annotation Quality](#object-annotation-quality) - <small>Compares object annotations against similar image crops</small>         | `image`     | `bounding box`, `polygon` |

## Image-level Annotation Quality

This metric creates embeddings from images. Then, these embeddings are used to build
nearest neighbor graph. Similar embeddings' classifications are compared against each other.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/semantic/img_classification_quality.py)

## Object Annotation Quality

This metric transforms polygons into bounding boxes
and an embedding for each bounding box is extracted. Then, these embeddings are compared
with their neighbors. If the neighbors are annotated differently, a low score is given to it.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/semantic/img_object_quality.py)
