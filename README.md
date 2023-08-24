<p align="center">
<a href="https://docs.encord.com/docs/active-overview" target="_blank">Documentation</a> |
<a href="https://colab.research.google.com/drive/11iZE1CCFIGlkWdTmhf5XACDojtGeIRGS?usp=sharing" target="_blank">Try it Now</a> |
<a href="https://encord.com/encord_active/" target="_blank">Website</a> |
<a href="https://encord.com/blog/" target="_blank">Blog</a> |
<a href="https://join.slack.com/t/encordactive/shared_invite/zt-1hc2vqur9-Fzj1EEAHoqu91sZ0CX0A7Q" target="_blank">Join our Community</a> |
</p>

<h1 align="center">
  <a href="https://encord.com"><img src="https://raw.githubusercontent.com/encord-team/encord-active/main/docs/static/img/icons/encord_active_logo.png" alt="Encord logo"/></a>
</h1>

<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1; padding: 10px;">
    <a href="https://docs.encord.com/docs/active-overview" target="_blank" style="text-decoration:none">
      <img alt="Documentation" src="https://img.shields.io/badge/docs-Online-blue">
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

## ❓ What is Encord Active?

[Encord Active][encord-active-landing] is an open-source toolkit to test, validate, and evaluate your models and surface, curate, and prioritize the most valuable data for labeling to supercharge model performance.

Use Encord Active to:

- Test, validate, and evaluate your models with advanced error analysis
- Generate model explainability reports
- Surface, curate, and prioritize the most valuable data for labeling
- Search through your data using natural language (beta feature)
- Find and fix dataset errors and biases (duplicates, outliers, label errors)

![homepage-visual]

## Installation

The simplest way to install the CLI is using `pip` in a suitable virtual environment:

```shell
pip install encord-active
```

We recommend using a virtual environment, such as `venv`:

```shell
python3.9 -m venv ea-venv
source ea-venv/bin/activate
pip install encord-active
```

> `encord-active` requires [python3.9][python-39].
> If you have trouble installing `encord-active`, you find more detailed instructions on
> installing it [here](https://docs.encord.com/docs/active-installation). If just want to see it in action try out [this Colab notebook][colab-notebook].

## 👋 Quickstart

Get started immediately by sourcing your environment and running the code below.
This downloads a small dataset and launches the Encord Active App for you to explore:

```shell
encord-active quickstart
```

or you can use <a href="https://hub.docker.com/r/encord/encord-active"><img src="https://www.docker.com/wp-content/uploads/2022/03/horizontal-logo-monochromatic-white.png" height="20"/></a>:

```shell
docker run -it --rm -p 8501:8501 -p 8502:8502 -v ${PWD}:/data encord/encord-active quickstart
```

After opening the UI, we recommend you to head to the [workflow documentation][encord-active-docs-workflow] to see some common workflows.

![projects page](https://raw.githubusercontent.com/encord-team/encord-active/main/docs/static/img/projects-page.jpg)

## 💡 When to use Encord Active?

Encord Active helps you understand and improve your data, labels, and models at all stages of your computer vision journey.

Whether you've just started collecting data, labeled your first batch of samples, or have multiple models in production, Encord Active can help you.

![encord active diagram](https://raw.githubusercontent.com/encord-team/encord-active/main/docs/static/img/process-chart-ea.webp)

## 🔖 Documentation

Our full documentation is available [here][encord-active-docs]. In particular, we recommend checking out:

- [Getting Started](https://docs.encord.com/docs/active-getting-started)
- [Imports](https://docs.encord.com/docs/active-import)
- [Workflows][encord-active-docs-workflow]
- [User Guides](https://docs.encord.com/docs/active-user-guide)
- [CLI Documentation](https://docs.encord.com/docs/active-cli)

## ⬇️ Download a sandbox dataset

Another way to quickly get familiar with Encord Active is to download a dataset from its sandbox.
The download command will ask which pre-built dataset to use and will download it into a new directory in the current working directory.

```shell
encord-active download
cd /path/of/downloaded/project
encord-active start
```

The app should then open in the browser.
If not, navigate to [`localhost:8501`](http://localhost:8501).
Our [docs][encord-active-docs] contain more information about what you can see in the page.

## <img width="24" height="24" src="https://raw.githubusercontent.com/encord-team/encord-active/main/docs/static/img/icons/encord_icon.png"/> Import your dataset

### Quick import Dataset

To import your data (without labels) save your data in a directory and run the command:

```shell
# within venv
encord-active init /path/to/data/directory
```

A project will be created using the data in the directory.

To start the project run:

```shell
cd /path/to/project
encord-active start
```

You can find more details on the `init` command in the [documentation][encord-active-docs-init].

### Import from COCO

To import your data, labels, and predictions from COCO, save your data in a directory and run the command:

```shell
# install COCO extras
(ea-venv)$ python -m pip install encord-active[coco]

# import samples with COCO annotations
encord-active import project --coco -i ./images -a ./annotations.json

# import COCO model predictions
encord-active import predictions --coco results.json
```

### Import from the Encord platform

This section requires [setting up an ssh key][encord-docs-ssh] with Encord, so slightly more technical.

To import an Encord project, use this command:

```shell
encord-active import project
```

The command will allow you to search through your Encord projects and choose which one to import.

## ⭐ Concepts and features

### Quality metrics:

Quality metrics are applied to your data, labels, and predictions to assign them quality metric scores.
Plug in your own or rely on Encord Active's prebuilt quality metrics.
The quality metrics automatically decompose your data, label, and model quality to show you how to improve your model performance from a data-centric perspective.
Encord Active ships with 25+ metrics and more are coming; [contributions][contribute-url] are also very welcome.

### Core features:

- [Data Exploration](https://docs.encord.com/docs/active-exploring-data-and-label-distributions)
- [Data Outlier detection](https://docs.encord.com/docs/active-identify-outliers#data-outliers)
- [Label Outlier detection](https://docs.encord.com/docs/active-identify-outliers#label-outliers)
- [Object Detection/segmentation Model Decomposition](https://docs.encord.com/docs/active-evaluate-detection-models)
- [Classification Model Decomposition](https://docs.encord.com/docs/active-evaluate-classification-models)
- [Similarity Search](https://docs.encord.com/docs/active-exploring-image-similarity)
- [Data & Label Tagging](https://docs.encord.com/docs/active-tagging)
- [Visualize TP/FP/FN](https://docs.encord.com/docs/active-evaluate-detection-models#exploring-the-individual-samples)
- And much more!

Visit our [documentation][encord-active-docs] to learn more.

### Supported data:

| Data Types |     | Labels          |     | Project sizes |               |
|------------|-----|-----------------|-----|---------------|---------------|
| `jpg`      | ✅   | Bounding Boxes  | ✅   | Images        | 25.000        |
| `png`      | ✅   | Polygons        | ✅   | Videos \*     | 25.000 frames |
| `tiff`     | ✅   | Segmentations   | ✅   |               |               |
| `mp4` \*   | ✅   | Classifications | ✅   |               |               |
|            |     | Polylines       | 🟡  |               |               |

\* Requires an Encord Annotate account

## 🧑🏽‍💻Development

### 🛠 Build your own quality metrics

Encord Active is built with customizability in mind. Therefore, you can easily build your own custom metrics 🔧.
See the [Writing Your Own Metric][encord-active-docs-write-metric] page in the docs for details on this topic.

If you need help or guidance feel free to ping us in our  **[Slack workspace][slack-join]**!

## 👪 Community and support

[Join our community on Slack][slack-join] to connect with the team behind Encord Active.
Also, feel free to [suggest improvements or report problems][report-issue] via GitHub issues.

## 🎇 Contributions

If you're using Encord Active in your organization, please try to add your company name to the [ADOPTERS.md][adopters]. It really helps the project to gain momentum and credibility. It's a small contribution back to the project with a big impact.

If you want to share your custom metrics or improve the tool, please see our [contributing docs][contribute-url].

### 🦸 Contributors

<a href="https://github.com/encord-team/encord-active/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=encord-team/encord-active" alt="Contributors graph"/>
</a>

[@Javi Leguina](https://github.com/jleguina)

## Licence

This repository is published under the Apache 2.0 licence.

[adopters]: https://github.com/encord-team/encord-active/blob/main/ADOPTERS.md
[colab-notebook]: https://colab.research.google.com/drive/11iZE1CCFIGlkWdTmhf5XACDojtGeIRGS?usp=sharing
[contribute-url]: https://docs.encord.com/docs/active-contributing
[encord-active-docs-init]: https://docs.encord.com/docs/active-quick-import
[encord-active-docs-workflow]: https://docs.encord.com/docs/active-workflows
[encord-active-docs-write-metric]: https://docs.encord.com/docs/active-write-custom-quality-metrics
[encord-active-docs]: https://docs.encord.com/docs/active-overview
[encord-active-landing]: https://encord.com/encord-active/
[encord-docs-ssh]: https://docs.encord.com/docs/annotate-public-keys#set-up-public-key-authentication
[homepage-visual]: https://raw.githubusercontent.com/encord-team/encord-active/main/homepage_visual.png
[python-39]: https://www.python.org/downloads/release/python-3915/
[report-issue]: https://github.com/encord-team/encord-active/issues/new/choose
[slack-join]: https://join.slack.com/t/encordactive/shared_invite/zt-1hc2vqur9-Fzj1EEAHoqu91sZ0CX0A7Q
