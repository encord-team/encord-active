---
title: "Quality metrics"
slug: "active-quality-metrics"
hidden: true
metadata: 
  title: "Quality Metrics"
  description: "Evaluate quality with Encord Active metrics. Data, label, model quality analysis. Customize metrics. Enhance computer vision infrastructure."
category: "6480a3981ed49107a7c6be36"
---

Quality metrics evaluate the quality of various components in your computer vision infrastructure, and therefore constitute the foundation of Encord Active. They are additional parameterization options added onto your data, labels, and models; they are ways of indexing your data, labels, and models in semantically interesting and relevant ways.

Encord Active (EA) is designed to compute, store, inspect, manipulate, and use quality metrics for a wide array of functionality. It hosts a library of these quality metrics, and importantly allows you to customize by writing your own “Quality Metrics” to calculate/compute QMs across your dataset.

We have split the metrics into the following categories:

- **Data Quality Metrics:** For analyzing and working with your image, sequence or video data. These metrics operate on images or individual video frames and are heuristic in the sense that they depend on the image content without labels.
  - Example metrics: Area, Brightness, Green Value, Sharpness.
  
- **Label Quality Metrics:** For analyzing and working with your labels. These metrics operate on the geometries of objects, like <<glossary:bounding box>>es, <<glossary:polygon>>s, segmentations and <<glossary:polyline>>s, and the heuristics of classifications.
  - Example metrics: Aspect Ratio, Classification Quality, Occlusion Risk.

---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Clickable Div</title>\n    <style>\n        .clickable-div {\n            display: inline-block;\n            width: 200px;\n            height: 50px;\n            background-color: #ffffff;\n            border: solid;\n            text-align: center;\n            line-height: 50px;\n            color: #000000;\n            text-decoration: none;\n            margin: 10px;\n        }\n\n        .clickable-div:hover {\n            background-color: #ededff;\n        }\n    </style>\n</head>\n<body>\n    <a href=\"https://docs.encord.com/docs/active-data-quality-metrics\" class=\"clickable-div\">Data Quality Metrics</a> <a href=\"https://docs.encord.com/docs/active-label-quality-metrics\" class=\"clickable-div\">Label Quality Metrics</a> <a href=\"https://docs.encord.com/docs/active-write-custom-quality-metrics\" class=\"clickable-div\">Custom Quality Metrics</a>\n</body>\n</html>"
}
[/block]