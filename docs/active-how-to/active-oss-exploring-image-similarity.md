---
title: "Exploring image similarity"
slug: "active-oss-exploring-image-similarity"
hidden: false
metadata: 
  title: "Exploring image similarity"
  description: "Enhance data quality with visual similarity search in Encord Active. Detect edge cases, duplicates, and label quality. Streamline dataset management."
  image: 
    0: "https://files.readme.io/7d31a4f-image_16.png"
createdAt: "2023-07-11T16:27:42.192Z"
updatedAt: "2023-08-09T12:46:40.225Z"
category: "65a71bbfea7a3f005192d1a7"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/local_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

**Mine edge cases, duplicates, and check the quality of your labels with visual similarity search**

Frequently, when distinctive characteristics arise within a dataset, identifying similar images becomes crucial (e.g., for relabeling or removal). Detecting these instances assists in assessing the thoroughness of data representation and the accuracy of labels, particularly in situations where certain classes may be underrepresented or labels could be inaccurately assigned. As datasets expand, manual identification of such cases becomes progressively challenging.

Leverage Encord Active's **similarity search** feature to effortlessly locate semantically akin images in your dataset. Upon identifying an edge case or duplicate, applying tags and executing actions such as relabeling or deletion can be performed.

[//]: # (In this workflow, the [COCO Validation 2017 Dataset]&#40;https://docs.encord.com/docs/active-cli#coco-validation-2017-dataset&#41; is used as an example.)

## Steps

1. Access the _Explorer_ page and locate the image or label of interest.
2. Click the **Similar items** button associated with the selected sample. Encord Active will arrange the samples on the _Explorer_ page, showcasing the most semantically similar images first.
  ![Displaying similar images based on the similarity search query](https://storage.googleapis.com/docs-media.encord.com/static/img/active/workflows/explorer-image-similarity-search.png)

> 👍 Tip
> To cancel the similarity search, you can click the <kbd>X</kbd> button located in the top right corner of the chosen sample or the <kbd>RESET FILTERS</kbd> button positioned near the natural language search bar.
