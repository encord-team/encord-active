---
title: "Exploring data and label distributions"
slug: "active-oss-exploring-data-and-label-distributions"
hidden: false
metadata: 
  title: "Exploring data and label distributions"
  description: "Visualize & understand distributions with Encord Active. Optimize models by uncovering missing data and label insights."
  image: 
    0: "https://files.readme.io/8aedbf8-image_16.png"
createdAt: "2023-07-11T16:27:42.230Z"
updatedAt: "2023-08-09T12:43:09.288Z"
category: "65a71bbfea7a3f005192d1a7"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/local_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

Encord Active provides the capability to visually explore data and label distributions via pre-defined metrics, custom metrics, and label classes.

Gaining insights into data distribution across diverse quality metrics allows for the identification of potential data gaps that could influence model performance on outliers or edge cases.

[//]: # (In this workflow, the [COCO Validation 2017 Dataset]&#40;https://docs.encord.com/docs/active-cli#coco-validation-2017-dataset&#41; is used as an example.)

## Static charts

Access the _Summary_ page to utilize the metric distribution charts and explore the _2D Metrics View_ for data and labels.

In the _2D Metrics View_, one metric's values are plotted on the x-axis, while the values of the other metric are represented on the y-axis. This visualization allows for an examination of the relationship between these two metrics and their potential interactions within the data and labels.

The label class distribution can be visualized by selecting the `Label Class` property from the drop-down menu within the Metric Distribution section on the _Label_ tab.

## Interactive exploration

Access the Explorer page and select a quality metric (such as `Brightness` or `Object Annotation Quality`) from the _Order by_ drop-down. This menu is located above the natural language search bar and enables data to be organized according to the chosen criteria.

The dashboard displays data distribution based on the selected metric. Navigating through the ordered dataset is possible by progressing through visualized data items. It is also feasible to perform range selections on the distribution chart or apply the chosen metric as a filter and utilize the slider to define a specific range.
