# Semantic

Operates with the semantic information of images or individual video frames.

| Title                                                                                                                                    | Metric Type   | Data Type                                           |
|------------------------------------------------------------------------------------------------------------------------------------------|---------------|-----------------------------------------------------|
| [Image Singularity](#image-singularity) - <small>Finds duplicate and near-duplicate images</small>                                       | `image`       |                                                     |
| [Image-level Annotation Quality](#image-level-annotation-quality) - <small>Compares image classifications against similar images</small> | `image`       | `radio`                                             |
| [Object Annotation Quality](#object-annotation-quality) - <small>Compares object annotations against similar image crops</small>         | `image`       | `bounding box`, `polygon`, `rotatable bounding box` |


## Image Singularity  

This metric gives each image a score that shows each image's uniqueness.  
- A score of zero means that the image has duplicates in the dataset; on the other hand, a score close to one represents that image is quite unique. Among the duplicate images, we only give a non-zero score to a single image, and the rest will have a score of zero (for example, if there are five identical images, only four will have a score of zero). This way, these duplicate samples can be easily tagged and removed from the project.    
- Images that are near duplicates of each other will be shown side by side. 
### Possible actions
- **To delete duplicate images:** You can set the quality filter to cover only zero values (that ends up with all the duplicate images), then use bulk tagging (e.g., with a tag like `Duplicate`) to tag all images.
- **To mark duplicate images:** Near duplicate images are shown side by side. Navigate through these images and mark whichever is of interest to you.
  

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/semantic/image_singularity.py)

## Image-level Annotation Quality  
This metric creates embeddings from images. Then, these embeddings are used to build
    nearest neighbor graph. Similar embeddings' classifications are compared against each other.
          

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/semantic/img_classification_quality.py)

## Object Annotation Quality  
This metric transforms polygons into bounding boxes
    and an embedding for each bounding box is extracted. Then, these embeddings are compared
    with their neighbors. If the neighbors are annotated differently, a low score is given to it.
      

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/semantic/img_object_quality.py)


