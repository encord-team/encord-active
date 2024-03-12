---
title: "Filtering"
slug: "active-oss-filtering"
hidden: false
metadata: 
  title: "Filtering"
  description: "Enhance insights with data filtering in Encord Active: Identify patterns, remove duplicates, improve model behavior. Use standard filters, embedding plots, or natural language search"
  image: 
    0: "https://files.readme.io/9123073-image_16.png"
createdAt: "2023-07-14T16:16:03.504Z"
updatedAt: "2023-08-09T16:19:33.172Z"
category: "65a71bbfea7a3f005192d1a7"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/local_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

**Learn how to filter data in Encord Active**

Filtering data in Encord Active is crucial for various reasons. It enables insights and actionable results on the following key aspects and more:
1. Identification of patterns, trends, or anomalies within a subset of the data.
2. Recognition of duplicates, outliers and inconsistencies.
3. Removal of irrelevant, noisy and erroneous data.
4. Understanding model's behaviour and potential skewness when facing different subsets of the data. 
5. **[Encord project only]** Update tasks status to prioritize some unannotated images in the labeling stage and send labels to be reviewed/fixed, all along with descriptive comments for the project users (e.g. annotators and reviewers).

Encord Active provides three data filtering methods:
1. **Standard filter feature**: This option allows users to refine their search using metadata filters, user-defined tag filters, and metric filters.
2. **Embedding plot**: A two-dimensional visualization technique used by Encord to represent high-dimensional data in a more interpretable form. Can be used to select points within a specific rectangular area, thereby focusing on a particular subset of data points for in-depth analysis.
3. **Natural language search**: Enables users to enter descriptive queries in everyday language, making it easier to find relevant images without the need for specific keywords or complex search parameters.


# Standard filter feature

The standard filter feature offers the following filtering options:
1. **Data points metadata filters**: Filter data points based on metadata attributes such as `Object Class` and `Annotator`, allowing to focus on specific classes or annotations created by particular annotators.
2. **User-defined tag filters**: Apply filters based on user-defined tags, enabling categorization and filtering of data points according to custom tags. 
3. **Metric filters**: Utilize metrics, including built-in ones like `Image Diversity` and `Label Duplicates`, as well as user-defined metrics, to filter data points based on potentially complex properties.

[//]: # (Don't show this section in the ToC. Use H3 heading to make that happen.)
### Steps to use the standard filter feature

1. Go to the Explorer page and locate the filter feature.
  ![Explorer page featuring the highlighted filter feature, allowing users to refine data visualization](https://storage.googleapis.com/docs-media.encord.com/static/img/active/user-guide/explorer-highlight-filter.png)
2. Choose one or more filters from the available options.
3. For numerical filters, specify the threshold range. For categorical filters, select the groups of interest.
  ![Filtering data example with the `Red value` filter applied, narrowing down data points based on a specified threshold](https://storage.googleapis.com/docs-media.encord.com/static/img/active/user-guide/explorer-filter-by-red-values.png)

> 👍 Tip
> Take advantage of one of the UI components to personalize the visualization order of the filtered data based on a specific metric. You can choose to display the data in either ascending or descending order, depending on your preferences.
>
> ![Customize visualization order option in the UI](https://storage.googleapis.com/docs-media.encord.com/static/img/active/user-guide/explorer-component-order-by.png)


# Embedding plot

The embedding plot in Encord Active is a two-dimensional visualization technique used to represent high-dimensional data in a more interpretable and visually accessible form. By reducing the dimensionality of the data, the embedding plot helps preserve the underlying structure and patterns of the original data.

In the plot, each data point is represented as a single point in the two-dimensional space, with proximity indicating similarity and shared characteristics among corresponding high-dimensional data points. This allows selecting points within a specific rectangular area, enabling a focused analysis of a particular subset of data points.

By defining a rectangular area on the plot, users can quickly isolate and examine the data points that fall within that region. The selection can be based on specific criteria or visual observations, allowing further exploration of attributes or additional analysis on the chosen subset.

This interactive functionality enhances the ability to gain deeper insights into underlying patterns and relationships within the selected area, providing a flexible and intuitive way to analyze and understand the data points within the dataset.

Based on the selected option in the _Order by_ drop-down, users can choose to visualize either the embedding plot for the data or the labels.

![Vibrant 2D embedding plot with distinct data points highlighting patterns and clusters](https://storage.googleapis.com/docs-media.encord.com/static/img/active/user-guide/explorer-embedding-plot.png)

> 👍 Tip
> In addition to selecting points within a rectangular area, the label embedding plot offers the functionality to filter data points based on the label classes.


# Natural language search

<details>
<summary> Video Tutorial - Natural language search in EA </summary>

[block:html]
{
  "html": "<div style=\"position: relative; padding-bottom: 56.25%; height: 0;\"><iframe src=\"https://www.loom.com/embed/c7da1b687690403a9d440707b5e38f32?sid=2fd7915a-665b-4fe2-bf37-613f87619fef\" frameborder=\"0\" webkitallowfullscreen mozallowfullscreen allowfullscreen style=\"position: absolute; top: 0; left: 0; width: 100%; height: 100%;\"></iframe></div>"
}
[/block]

</details>

The natural language search feature enables users to enter descriptive queries in everyday language, such as "images that contain baseball items". The system intelligently processes the query and retrieves images that match the description. This feature simplifies and greatly enhances the search experience within Encord Active, allowing finding relevant images without the need for specific keywords or complex search parameters.

![Encord Active's natural language search in action, retrieving relevant images based on descriptive queries](https://storage.cloud.google.com/docs-media.encord.com/static/img/active/user-guide/explorer-natural-language-search.png)

> ℹ️ Note
> The natural language search feature is exclusively available in the hosted version of Encord Active.