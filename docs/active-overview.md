---
title: "Overview of Encord Active"
slug: "active-overview"
hidden: false
metadata: 
  title: "Overview of Encord Active"
  description: "Optimize models with Encord Active: A cloud-based and open source toolkit for data quality, model enhancement, and failure mode detection. Boost performance."
  image: 
    0: "https://files.readme.io/294aa28-image_16.png"
createdAt: "2023-07-11T16:27:41.844Z"
updatedAt: "2023-08-11T12:43:00.437Z"
category: "6480a3981ed49107a7c6be36"
---

<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1; padding: 10px;">
    <a href="https://github.com/encord-team/encord-active" target="_blank" style="text-decoration:none">
      <img alt="View on Github" src="https://img.shields.io/badge/GitHub-View_Code-blue?label=&logo=github&labelColor=181717">
    </a>
    <a href="https://github.com/encord-team/encord-notebooks" target="_blank" style="text-decoration:none">
      <img alt="Encord Notebooks" src="https://img.shields.io/badge/Encord_Notebooks-blue?logo=github&label=&labelColor=181717">
    </a>
    <a href="https://colab.research.google.com/drive/11iZE1CCFIGlkWdTmhf5XACDojtGeIRGS?usp=sharing" target="_blank" style="text-decoration:none">
      <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
    <a href="https://join.slack.com/t/encordactive/shared_invite/zt-1hc2vqur9-Fzj1EEAHoqu91sZ0CX0A7Q" target="_blank" style="text-decoration:none">
      <img alt="Join us on Slack" src="https://img.shields.io/badge/Join_Our_Community-4A154B?label=&logo=slack&logoColor=white">
    </a>
    <a href="https://twitter.com/encord_team" target="_blank" style="text-decoration:none">
      <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/encord_team?label=%40encord_team&amp;style=social">
    </a>
  </div>
  <div style="flex: 1; padding: 10px;">
    <img alt="Python versions" src="https://img.shields.io/pypi/pyversions/encord-active">
    <a href="https://pypi.org/project/encord-active/" target="_blank" style="text-decoration:none">
      <img alt="PyPi project" src="https://img.shields.io/pypi/v/encord-active">
    </a>
    <a href="https://docs.encord.com/docs/active-contributing" target="_blank" style="text-decoration:none">
      <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-Welcome-blue">
    </a>
    <img alt="Licence" src="https://img.shields.io/github/license/encord-team/encord-active">
    <img alt="Downloads" src="https://static.pepy.tech/badge/encord-active">
  </div>
</div>

<div style="text-align: center;">
  <img alt="Encord Active logo" src="https://storage.googleapis.com/docs-media.encord.com/active-1.png" width="300">
</div>


Encord Active is available in two versions: **Encord Active Cloud** and **Encord Active OS**. Active Cloud is tightly integrated with Encord Annotate, with Active Cloud and Annotate being hosted by Encord. Encord Active OS is an open source toolkit that can be installed on a local computer/server.

[Encord Active Cloud](https://encord.com/encord-active/) and Encord Active OS (open source) are active learning solutions that help you find failure modes in your models, and improve your data quality and model performance.

Use Active Cloud and Active OS to visualize your data, evaluate your models, surface model failure modes, find labeling mistakes, prioritize high-value data for relabeling and more!

[//]: # (![video]&#40;https://storage.googleapis.com/docs-media.encord.com/static/img/gifs/ea-demo.gif&#41;)

## When to use Encord Active?

Encord Active helps you understand and improve your data, labels, and models at all stages of your computer vision journey.

Whether you've just started collecting data, labeled your first batch of samples, or have multiple models in production, Encord Active can help you.

![encord active diagram](https://storage.googleapis.com/docs-media.encord.com/static/img/process-chart-ea.webp)

### Example use cases

To give you a better idea about how Active Cloud and Annotate work together, here are some use cases.

**Data Curation and Label Error Correction**

![Encord Active workflow](https://storage.googleapis.com/docs-media.encord.com/static/img/active/active-workflow-data-curation-label-validation.png)

**Optimize Model Performance**

![Encord Active workflow](https://storage.googleapis.com/docs-media.encord.com/static/img/active/active-workflow-model-optimization.png)

> ℹ️ Note
> Before going any further, you should know what a Collection is in Encord Active Cloud. Collections provide a way to save interesting groups of data units and labels, to support and guide your downstream workflow. For more information on [Collections go here](https://docs.encord.com/docs/active-collections).

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Clickable Div</title>\n    <style>\n        .clickable-div {\n            display: inline-block;\n            width: 200px;\n            height: 50px;\n            background-color: #ffffff;\n            border: solid;\n            text-align: center;\n            line-height: 50px;\n            color: #000000;\n            text-decoration: none;\n            margin: 10px;\n        }\n\n        .clickable-div:hover {\n            background-color: #ededff;\n        }\n    </style>\n</head>\n<body>\n    <a href=\"https://docs.encord.com/docs/active-use-cases\" class=\"clickable-div\">Data Cleansing/Curation</a> <a href=\"https://docs.encord.com/docs/active-use-cases\" class=\"clickable-div\">Label Correction/Validation</a> <a href=\"https://docs.encord.com/docs/active-use-cases\" class=\"clickable-div\">Model/Prediction Evaluation</a>\n</body>\n</html>"
}
[/block]

## What data does Encord Active support?

<ActiveFileSizeBestPractice />

<!--
| Features                         |                 Open Source                 |                                     Hosted                                      |
| :------------------------------- | :-----------------------------------------: | :-----------------------------------------------------------------------------: |
| Data types                       |                `jpg`, `png`                 |                           `jpg`, `png`, `tiff`, `mp4`                           |
| Labels                           | `classification`, `bounding box`, `polygon` | `classification`, `bounding box`, `polygon`, `polyline`, `bitmask`, `key-point` |
| Number of images                 |   25,000 per project (unlimited projects)   |                     500,000 per project (unlimited projects)                    |
| Videos                           |                      -                      |                                 2 hours @ 30fps                                 |
| Data exploration                 |                     ✅                      |                                       ✅                                        |
| Label exploration                |                     ✅                      |                                       ✅                                        |
| Similarity search                |                     ✅                      |                                       ✅                                        |
| Off-the-shelf quality metrics    |                     ✅                      |                                       ✅                                        |
| Custom quality metrics           |                     ✅                      |                                       ✅                                        |
| Data and label tagging           |                     ✅                      |                                       ✅                                        |
| Image duplication detection      |                     ✅                      |                                       ✅                                        |
| Label error detection            |                     ✅                      |                                       ✅                                        |
| Outlier detection                |                     ✅                      |                                       ✅                                        |
| Dataset summaries                |                     ✅                      |                                       ✅                                        |
| Project comparison               |                     ✅                      |                                       ✅                                        |
| Model evaluation                 |                      -                      |                                       ✅                                        |
| Label synchronization            |                      -                      |                                       ✅                                        |
| Natural language search          |                      -                      |                                       ✅                                        |
| Search by image                  |                      -                      |                                       ✅                                        |
| Integration with Encord Annotate |                      -                      |                                       ✅                                        |
| Active learning workflows        |                      -                      |                                       ✅                                        |
| Nested Attributes                |                      -                      |                                   Contact us                                    |
-->


| Features                         |             Active Open Source              |                                  Active Cloud                                   |
| :------------------------------- | :-----------------------------------------: | :-----------------------------------------------------------------------------: |
| Data types                       |                `jpg`, `png`                 |                           `jpg`, `png`, `tiff`, `mp4`                           |
| Labels**<sup>1</sup>**           |`classification`, `bounding box`, `polygon`  | `classification`, `bounding box`, `polygon`, `polyline`, `bitmask`, `key-point` |
| Number of images                 |   25,000 per project (unlimited projects)   |                     500,000 per project (unlimited projects)                    |
| Videos                           |                      -                      |                                 2 hours @ 30fps                                 |
| Data exploration                 |                     ✅                      |                                       ✅                                        |
| Label exploration                |                     ✅                      |                                       ✅                                        |
| Similarity search                |                     ✅                      |                                       ✅                                        |
| Off-the-shelf quality metrics    |                     ✅                      |                                       ✅                                        |
| Custom quality metrics           |                     ✅                      |                                       ✅                                        |
| Data and label tagging           |                     ✅                      |                                       ✅                                        |
| Image duplication detection      |                     ✅                      |                                       ✅                                        |
| Label error detection            |                     ✅                      |                                       ✅                                        |
| Outlier detection                |                     ✅                      |                                       ✅                                        |
| Collections                      |                      -                      |                                       ✅                                        |
| Model evaluation                 |                      -                      |                                       ✅                                        |
| Label synchronization            |                      -                      |                                       ✅                                        |
| Natural language search          |                      -                      |                                       ✅                                        |
| Search by image                  |                      -                      |                                       ✅                                        |
| Integration with Encord Annotate |                      -                      |                                       ✅                                        |
| Nested Attributes                |                      -                      |                                       ✅                                        |
| Custom metadata                  |                      -                      |                                       ✅                                        |

**<sup>1</sup>**: <<glossary:Object>>s and <<glossary:classification>>s are both supported:

Filtering:

- Objects + all attributes
- Classification + all attributes

Model evaluation:

- Objects and Classifications cannot be mixed
- Classification support includes top level radio button
- Object support includes top level object