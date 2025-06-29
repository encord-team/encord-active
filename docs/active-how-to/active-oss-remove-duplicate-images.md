---
title: "Remove duplicate images"
slug: "active-oss-remove-duplicate-images"
hidden: true
metadata: 
  title: "Remove duplicate images"
  description: "Enhance dataset quality: Detect & remove duplicate images with Encord Active. Mitigate bias, optimize data for models"
  image: 
    0: "https://files.readme.io/05b71a8-image_16.png"
createdAt: "2023-07-11T16:27:42.223Z"
updatedAt: "2023-08-09T16:11:47.793Z"
category: "6480a3981ed49107a7c6be36"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/local_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

**Enhance datasets by detecting and eliminating duplicate and near-duplicate images**

The presence of duplicate or closely similar images can introduce bias in deep learning models. Encord Active provides the capability to identify and eliminate these duplicate or near-duplicate images from datasets. This process contributes to enhancing data quality by removing redundant instances, ultimately leading to improved model performance.

In this workflow, the [`Image Singularity` quality metric](https://docs.encord.com/docs/active-data-quality-metrics#uniqueness) is employed to identify duplicate and near-duplicate images.

## Image singularity (AKA Uniqueness)

The `Image singularity` metric evaluates all images within the dataset and assigns a uniqueness score to each, indicating their distinctiveness.
- The uniqueness score falls within the [0,1] range. A higher score indicates a greater level of image uniqueness.
- A score of zero signifies the presence of at least one identical image within the dataset. For instances with _N_ duplicate images, _N-1_ of them are assigned a score of zero (with only one holding a non-zero score) to facilitate their exclusion from the dataset.
- Near-duplicate images are labeled as `Near-duplicate image` and are presented side by side in the Explorer's grid view. This setup simplifies the decision-making process when selecting which image to keep and which one to remove.

[//]: # (TODO add this option when Description fields made by metrics are accessible in the UI)
[//]: # (- In the context of _N_ duplicate images, _N-1_ images are associated with a single representative image, visible through the `Description` field in the details of each image.)

## Walkthrough

Go to the _Data_ tab within the _Summary_ page and pick the `Image Singularity` quality metric from the drop-down menu in the _Metric Distribution_ section.

![Distribution of data based on Image Singularity scores](https://storage.googleapis.com/docs-media.encord.com/static/img/active/workflows/summary-data-metric-distribution-image-singularity.png)

The chart displays the distribution of data based on the `Image Singularity` scores. The example image illustrates a project containing around 200 duplicate images. 

Proceed to the _Explorer_ page and choose the `Image Singularity` quality metric from the _Order by_ drop-down. This menu is positioned above the natural language search bar and enables data to be organized according to the chosen criteria.

![Ordering data by Image Singularity](https://storage.googleapis.com/docs-media.encord.com/static/img/active/workflows/explorer-order-by-image-singularity.png)

Choose any sample and click its corresponding **Similar items** button. This action will display images similar to the selected one, including any duplicates if they exist.

![Displaying similar images based on the similarity search query](https://storage.googleapis.com/docs-media.encord.com/static/img/active/workflows/explorer-image-similarity-search.png)

### Removing duplicate images

In situations where users aim to eliminate duplicate images from a dataset, they can flag these images and create a subset of the dataset devoid of duplicates.

1. Access the _Explorer_ page and ensure that a data metric is chosen in the _Group by_ dropdown. This steps ensures that the _Explorer_'s grid view shows data items.
2. Tag all images with a data tag, such as `non-duplicate images`, by utilizing the <kbd>SELECT ALL</kbd> button followed by the <kbd>TAG</kbd> button. This operation is known as [bulk tagging](https://docs.encord.com/docs/active-tagging#bulk-tagging). Afterwards, click the <kbd>CLEAR SELECTION</kbd> button to reset the selection.
3. Opt for the `Image Singularity` quality metric within the <kbd>FILTERS</kbd> button. Adjust the range slider for this metric to cover the entire range available. This step involves the [standard filter](https://docs.encord.com/docs/active-filtering#standard-filter-feature).
4. Click the <kbd>SELECT ALL</kbd> button to choose all image duplicates. Then, utilize the <kbd>TAG</kbd> button to remove the `non-duplicate images` tag from this subset. Upon completion, click both the <kbd>RESET FILTERS</kbd> and <kbd>CLEAR SELECTION</kbd> buttons to reset the selections. As a result, the subset labeled with the `non-duplicate images` tag will now exclusively consist of images that are not duplicated.
5. Choose the `Data Tags` option within the <kbd>FILTERS</kbd> button. Ensure that only the `non-duplicate images` tag is selected.
6. Click the <kbd>CREATE PROJECT SUBSET</kbd> button and follow the provided instructions to generate a project containing exclusively non-duplicate images.

Incorporating this workflow into dataset management strategies can significantly enhance data quality, eliminate redundancies, and contribute to more accurate model training and evaluation.

### Removing near-duplicate images

[block:image]
{
  "images": [
    {
      "image": ["https://storage.googleapis.com/docs-media.encord.com/static/img/images/workflows/improve-your-data-and-labels/remove-duplicate-images/remove-duplicate-images-11.png",
      "",
      "An example of near-duplicate image pairs detected with Encord Active"
      ],
      "align": "center",
      "caption": "An example of near-duplicate image pairs detected with Encord Active"
    }
  ]
}
[/block]

Similar to duplicates, near-duplicate images are those where one image slightly differs from another due to shifts, blurriness, or distortion. Consequently, they should also be eliminated from the dataset. However, in this scenario, a decision is required to determine which sample remains and which is discarded. These images possess scores marginally greater than 0 and are displayed alongside one another in the grid view, facilitating easy comparison.

To proceed:
1. Tag all images with a data tag, by utilizing the <kbd>SELECT ALL</kbd> button followed by the <kbd>TAG</kbd> button. Afterwards, click the <kbd>CLEAR SELECTION</kbd> button to reset the selection.
2. To focus on images with remarkably low uniqueness scores, opt for the `Image Singularity` quality metric within the <kbd>FILTERS</kbd> button and adjust the range slider for this metric to cover the range [0,0.05].
3. Examine the images and proceed to remove the tag from images intended for exclusion from the project. 
4. Follow the same export steps as outlined in the _Removing duplicate images_ section.

[//]: # (TODO add this option when Description fields made by metrics are accessible in the UI)
[//]: # (2. Identify images with a description indicating "_Near-duplicate image_." )

With these actions, users can efficiently manage near-duplicate images and improve dataset quality.
