---
sidebar_position: 2
---

# Data Quality Metrics

Data Quality Metrics work on images or individual video frames and depend on the image content without labels.

| Title                                                                                                                  | Metric Type                                                | Data Type                                                                                                          |
|------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [Area](#area) - <small>Ranks images by their area (width/height).</small>                                              | `image`                                                    |                                                                                                                    |
| [Aspect Ratio](#aspect-ratio) - <small>Ranks images by their aspect ratio (width/height).</small>                      | `image`                                                    |                                                                                                                    |
| [Blue Values](#blue-values) - <small>Ranks images by how blue the average value of the image is.</small>               | `image`                                                    |                                                                                                                    |
| [Blur](#blur) - <small>Ranks images by their blurriness.</small>                                                       | `image`                                                    |                                                                                                                    |
| [Brightness](#brightness) - <small>Ranks images by their brightness.</small>                                           | `image`                                                    |                                                                                                                    |
| [Contrast](#contrast) - <small>Ranks images by their contrast.</small>                                                 | `image`                                                    |                                                                                                                    |
| [Green Values](#green-values) - <small>Ranks images by how green the average value of the image is.</small>            | `image`                                                    |                                                                                                                    |
| [Image Singularity](#image-singularity) - <small>Finds duplicate and near-duplicate images.</small>                    | `image`                                                    |                                                                                                                    |
| [Image Difficulty](#image-difficulty) - <small>Ranks images from easy samples to hard samples.</small>                 | `image`                                                    |                                                                                                                    |
| [Random Values on Images](#random-values-on-images) - <small>Assigns a random value between 0 and 1 to images.</small> | `image`                                                    |                                                                                                                    |
| [Red Values](#red-values) - <small>Ranks images by how red the average value of the image is.</small>                  | `image`                                                    |                                                                                                                    |
| [Sharpness](#sharpness) - <small>Ranks images by their sharpness.</small>                                              | `image`                                                    |                                                                                                                    |


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

## Image Singularity  

This metric gives each image a score that shows each image's uniqueness.  
- A score of zero means that the image has duplicates in the dataset; on the other hand, a score close to one represents that image is quite unique. Among the duplicate images, we only give a non-zero score to a single image, and the rest will have a score of zero (for example, if there are five identical images, only four will have a score of zero). This way, these duplicate samples can be easily tagged and removed from the project.    
- Images that are near duplicates of each other will be shown side by side. 
### Possible actions
- **To delete duplicate images:** You can set the quality filter to cover only zero values (that ends up with all the duplicate images), then use bulk tagging (e.g., with a tag like `Duplicate`) to tag all images.
- **To mark duplicate images:** Near duplicate images are shown side by side. Navigate through these images and mark whichever is of interest to you.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/semantic/image_singularity.py)

## Image Difficulty

For selecting the first samples to annotate when there is no label in the project, choosing simple samples that 
represent those classes well gives better results.
This metric ranks images from easy samples to hard samples. Easy (hard) samples have lower (higher) scores.  
### Algorithm
1. K-means clustering is applied to image embeddings. The total number of clusters is obtained from the ontology file 
(if there 
are both object and image-level information, total object classes are determined as the total cluster number). 
If no ontology information exists, _K_ is determined as 10.
2. Samples for each cluster are ranked based on their proximity to cluster centers. Samples closer to the cluster 
centers refer to easy samples.
3. Different clusters are combined in a way that the result is ordered from easy to hard and the number of samples for 
each class is balanced for the first _N_ samples.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/semantic/image_difficulty.py)

## Random Values on Images  
Uses a uniform distribution to generate a value between 0 and 1 to each image  

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


