---
title: "Explore embedding plots"
slug: "active-exploring-embeddings"
hidden: false
metadata: 
  title: "Exploring embedding plots"
  description: "Improve your data curation and active learning workflows with the Embeddings view. Visualize clusters and gain deeper data understanding. Optimize workflows."
  image: 
    0: "https://files.readme.io/566b786-image_16.png"
createdAt: "2023-07-11T16:27:42.145Z"
updatedAt: "2023-08-09T12:45:07.861Z"
category: "6480a3981ed49107a7c6be36"
---
[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/hosted_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

Encord Active incorporates embedding plots — a two-dimensional visualization technique employed to represent intricate, high-dimensional data in a more comprehensible and visually coherent manner. This technique reduces data dimensionality while preserving the inherent structure and patterns within the original data.

The embedding plot aids in identifying interesting/noteworthy clusters, inspecting outliers, and excluding unwanted samples. Accessible on the **Project > Explorer** page, the embedding plot is adaptable to data or labels.

![Vibrant 2D data embedding plot highlighting data patterns and clusters](https://storage.googleapis.com/docs-media.encord.com/static/img/active/workflows/explorer-data-embedding-plot_01.png)

Notice how images are clustered around certain regions. By defining a rectangular area on the plot, users can quickly isolate and analyze data points within that defined region. This approach facilitates the exploration of commonalities among these samples.

Upon selecting a region, the content within the _Explorer_ page will be adjusted accordingly. Various actions can be executed with the chosen group:
- Use [Collections](https://docs.encord.com/docs/active-collections) to tag and group images.
- Investigate the performance of the selected samples within the _Predictions_ page.
- Establish subsets similar to these and then conduct comparisons.

Samples within the data embedding plot lack label information, resulting in uniform coloration across all points. Data points in the label embedding plot are color-coded based on their label classes.

![Vibrant 2D label embedding plot highlighting label patterns and clusters](https://storage.googleapis.com/docs-media.encord.com/static/img/active/workflows/explorer-label-embedding-plot_01.png)

> 👍 Tip
> The embedding plot is adaptable to data or labels. In addition to selecting points within a rectangular area, the label embedding plot offers the functionality to filter data points based on the label classes.

With the label embedding plot, users can:
- Identify classes that are often confused with each other.
- Detect samples with incorrect labeling, such as instances of a different class embedded within a larger cluster of another class.
- Spot outliers and subsequently eliminate them from the dataset.
