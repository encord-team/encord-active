---
title: "Explore image similarity"
slug: "active-exploring-image-similarity"
hidden: false
metadata: 
  title: "Explore image similarity"
  description: "Enhance data quality with visual similarity search in Encord Active. Detect edge cases, duplicates, and label quality. Streamline dataset management."
  image: 
    0: "https://files.readme.io/7d31a4f-image_16.png"
createdAt: "2023-07-11T16:27:42.192Z"
updatedAt: "2023-08-09T12:46:40.225Z"
category: "6480a3981ed49107a7c6be36"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/hosted_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

Frequently, when distinctive characteristics arise within a dataset, identifying similar images becomes crucial (e.g., for relabeling or removal). Detecting these instances assists in assessing the thoroughness of data representation and the accuracy of labels, particularly in situations where certain classes may be underrepresented or labels could be inaccurately assigned. As datasets expand, manual identification of such cases becomes progressively challenging.

Leverage Encord Active's **similarity search** feature to effortlessly locate semantically akin images in your dataset. Upon identifying an edge case or duplicate, applying tags and executing actions such as relabeling or deletion can be performed.

## Quick Tour

All of the sections in the Quick Tour assume that you are already in a Project.

> 👍 Tip
> Choose any image in the Explorer workspace and click its _Similar items_ [!Similarity button](https://storage.googleapis.com/docs-media.encord.com/static/img/active/workflows/similarity-button.png) button. This displays images similar to the selected one, including any duplicates if they exist.

### Explorer

The _Explorer_ page has three areas that can help you find duplicate images in your Project.

<details>

<summary><b>1: Duplicates Shortcut</b></summary>

Found in the _Overview_ tab, any images that have a `Uniqueness` value of 0 to 0.0001 are highlighted as duplicates. You can adjust this value from the _Filter_ tab.

![Duplicates shortcut](https://storage.googleapis.com/docs-media.encord.com/static/img/active/workflows/image-duplicates-qt-01.png)

</details>

<details>

<summary><b>2: Sorting by `Uniqueness`</b></summary>

The entire Project can be sorted by `Uniqueness`. Sort by ascending order to display duplicates first.

![Sorting by `Uniqueness`](https://storage.googleapis.com/docs-media.encord.com/static/img/active/workflows/image-duplicates-qt-02.png)

</details>

<details>

<summary><b>3: Filtering by `Uniqueness`</b></summary>

Filter the entire project using `Uniqueness`. 

Go to **Filter** tab > **Add Filter** > **Data Quality Metrics** > **Uniqueness** A small histogram diagram appears above the filter.

You can then change the filter settings to specify a range closer to 0.

![Filtering by `Uniqueness`](https://storage.googleapis.com/docs-media.encord.com/static/img/active/workflows/image-duplicates-qt-03.png)

</details>

### Analytics

In a Project, go to the _Analytics_ page and pick the `Uniqueness` quality metric for the _Metric Distribution_ section.

![Distribution of data based on Uniqueness scores](https://storage.googleapis.com/docs-media.encord.com/static/img/active/workflows/image-duplicates-qt-anal-01.png)

The chart displays the distribution of data based on the `Uniqueness` scores.