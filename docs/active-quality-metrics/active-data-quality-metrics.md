---
title: "Data quality metrics"
slug: "active-data-quality-metrics"
hidden: false
metadata: 
createdAt: "2023-07-21T09:09:02.294Z"
updatedAt: "2023-08-09T12:32:43.094Z"
category: "6480a3981ed49107a7c6be36"
---
Data quality metrics work on images or individual video frames.

## Access Data Quality Metrics

Data Quality Metrics are used for sorting data, filtering data, and data analytics.

| Title                                                                                                                                          | Metric Type | Ontology Type |
|------------------------------------------------------------------------------------------------------------------------------------------------|-------------|---------------|
| [Area](#area) - <small>Ranks images by their area (width/height).</small>                                                                      | `image`     |               |
| [Aspect Ratio](#aspect-ratio) - <small>Ranks images by their aspect ratio (width/height).</small>                                              | `image`     |               |
| [Blue Value](#blue-value) - <small>Ranks images by how blue the average value of the image is.</small>                                         | `image`     |               |
| [Brightness](#brightness) - <small>Ranks images by their brightness.</small>                                                                   | `image`     |               |
| [Contrast](#contrast) - <small>Ranks images by their contrast.</small>                                                                         | `image`     |               |
| [Diversity](#diversity) - <small>Forms clusters based on the ontology and ranks images from easy samples to annotate to hard samples to annotate.</small>              | `image`     |               |
| [Frame Number](#frame-number) - <small>Selects images based on a specified range.</small>                                                      | `image`     |               |
| [Green Value](#green-value) - <small>Ranks images by how green the average value of the image is.</small>                                     | `image`     |               |
| [Height](#height) - <small>Ranks images by the height of the image.</small>                                                                    | `image`     |               |
| [Object Count](#object-count) - <small>Counts number of objects in the image.</small>                                                                                                        | `image`     | `bounding box`, `checklist`, `point`, `polygon`, `polyline`, `radio`, `rotatable bounding box`, `skeleton`, `text` |
| [Object Density](#object-density) - <small>Computes the percentage of image area that is occupied by objects.</small>              | `image`     | `bounding box`, `polygon`, `rotatable bounding box`                                                                | 
| [Randomize Images](#randomize-images) - <small>Assigns a random value between 0 and 1 to images.</small>                                       | `image`     |               |
| [Red Value](#red-value) - <small>Ranks images by how red the average value of the image is.</small>                                           | `image`     |               |
| [Sharpness](#sharpness) - <small>Ranks images by their sharpness.</small>                                                                      | `image`     |               |
| [Uniqueness](#uniqueness) - <small>Finds duplicate and near-duplicate images.</small>                                                          | `image`     |               |
| [Width](#width) - <small>Ranks images by the width of the image.</small>                                                                       | `image`     |               |

**To access Data Quality Metrics for Explorer:**

1. Click a Project from the _Active_ home page.

2. Click **Explorer**.

3. Click **Data**.

4. Sort and filter the tabular data.

5. Click the plot diagram icon.

6. Sort and filter the embedding plot data.

**To access Data Quality Metrics for analytics:**

1. Click a Project from the _Active_ home page.

2. Click **Analytics**.

3. Click **Data**.

4. Select the quality metric you want to view from the **2D Metrics view** or **Metrics Distribution** graphs.

## Area  

Ranks images by their area. Area is computed as the product of image width and image height (_width x height_).
      
Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py).

## Aspect Ratio  

Ranks images by their aspect ratio. Aspect ratio is computed as the ratio of image width to image height (_width / height_).

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py).

## Blue Value  

Ranks images by how blue the average value of the image is.  

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py).

## Brightness  
Ranks images by their brightness. Brightness is computed as the average (normalized) pixel value across each image. 

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py).

## Contrast  
Ranks images by their contrast. Contrast is computed as the standard deviation of the pixel values. 

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py).

## Diversity

For selecting the first samples to annotate when there are no labels in the project. Choosing simple samples that represent those classes well, gives better results. This metric ranks images from easy samples to annotate to hard samples to annotate. Easy samples have lower scores, while hard samples have higher scores.

### Algorithm

1. [K-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) is applied to image embeddings. The total number of clusters is obtained from the <<glossary:Ontology>> file (if there are both object and image-level information, total object classes are determined as the total cluster number). If no ontology information exists, _K_ is determined as 10.

2. Samples for each cluster are ranked based on their proximity to cluster centers. Samples closer to the cluster centers refer to easy samples.

3. Different clusters are combined in a way that the result is ordered from easy to hard and the number of samples for each class is balanced for the first _N_ samples.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/semantic/image_diversity.py).

## Frame Number

Select a range of images in a video or a sequential group of images.




## Green Value  
Ranks images by how green the average value of the image is. 

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py).

## Height 
Ranks images by the height of the image.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py).

## Object Count

Counts number of objects in the image.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/object_counting.py).

## Object Density

Computes the percentage of image area that is occupied by objects.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/geometric/object_size.py).

## Randomize Images 
Uses a uniform distribution to generate a value between 0 and 1 for each image.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/random.py).

## Red Value
Ranks images by how red the average value of the image is. 

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py).

## Sharpness
Ranks images by their sharpness.

Sharpness is computed by applying a Laplacian filter to each image and computing the variance of the output. In short, the score computes "the amount of edges" in each image.

```python
score = cv2.Laplacian(image, cv2.CV_64F).var()
```

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py).

## Uniqueness

This metric gives each image a score that shows each image's uniqueness.  
- A score of zero means that the image has duplicates in the dataset; on the other hand, a score close to one represents that image is quite unique. Among the duplicate images, we only give a non-zero score to a single image, and the rest will have a score of zero (for example, if there are five identical images, only four will have a score of zero). This way, these duplicate samples can be easily tagged and removed from the project.    
- Images that are near duplicates of each other will be shown side by side. 

### Possible actions

- **To delete duplicate images:**  Set the quality filter to cover only zero values (that ends up with all the duplicate images), then use bulk tagging (for example, with a tag like `Duplicate`) to tag all images.
- **To mark duplicate images:** Near-duplicate images are shown side by side. Navigate through these images and mark whichever is of interest to you.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/semantic/image_singularity.py).

## Width 
Ranks images by the width of the image.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/heuristic/img_features.py).
