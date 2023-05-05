<p align="center">
<a href="https://docs.encord.com/active/docs" target="_blank">Documentation</a> |
<a href="https://colab.research.google.com/drive/11iZE1CCFIGlkWdTmhf5XACDojtGeIRGS?usp=sharing" target="_blank">Try it Now</a> |
<a href="https://encord.com/encord_active/" target="_blank">Website</a> |
<a href="https://encord.com/blog/" target="_blank">Blog</a> |
<a href="https://join.slack.com/t/encordactive/shared_invite/zt-1hc2vqur9-Fzj1EEAHoqu91sZ0CX0A7Q" target="_blank">Slack Channel</a>
</p>

<h1 align="center">
  <a href="https://encord.com"><img src="https://raw.githubusercontent.com/encord-team/encord-active/main/docs/static/img/icons/encord_active_logo.png" alt="Encord logo"/></a>
</h1>

<div align="center">

[![PRs-Welcome][contribute-image]][contribute-url]
![Licence][license-image]
[![PyPi project][pypi-package-image]][pypi-package]
![PyPi version][pypi-version-image]
[![Open In Colab][colab-image]][colab-notebook]
![downloads-badge][downloads-badge]

[![docs][docs-image]][encord-active-docs]
[!["Join us on Slack"][slack-image]][join-slack]
[![Twitter Follow][twitter-image]][twitter-url]

</div>

## ❓ What is Encord Active?

[Encord Active][encord-active-landing] is an open-source tookit to test, validate, and evaluate your models and surface, curate, and prioritize the most valuable data for labeling to supercharge model performance.

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
> installing it [here](https://docs.encord.com/active/docs/installation). If just want to see it in action try out [this Colab notebook][colab-notebook].

## 👋 Quickstart

Get started immediately by sourcing your environment and running the code below.
This downloads a small dataset and launches the Encord Active App for you to explore:

```shell
encord-active quickstart
```

or you can use <a href="https://hub.docker.com/r/encord/encord-active"><img src="https://www.docker.com/wp-content/uploads/2022/03/horizontal-logo-monochromatic-white.png" height="20"/></a>:

```shell
docker run -it --rm -p 8501:8501 -p 8000:8000 -v ${PWD}:/data encord/encord-active quickstart
```

After opening the UI, we recommend you to head to the [workflow documentation][encord-active-docs-workflow] to see some common workflows.

![projects page](https://raw.githubusercontent.com/encord-team/encord-active/main/docs/static/img/projects-page.jpg)

## 💡 When to use Encord Active?

Encord Active helps you understand and improve your data, labels, and models at all stages of your computer vision journey.

Whether you've just started collecting data, labeled your first batch of samples, or have multiple models in production, Encord Active can help you.

![encord active diagram](https://raw.githubusercontent.com/encord-team/encord-active/main/docs/static/img/process-chart-ea.webp)

## 🔖 Documentation

Our full documentation is available [here](https://docs.encord.com/active/docs). In particular we recommend checking out:

- [Getting Started](https://docs.encord.com/active/docs/getting-started)
- [Imports](https://docs.encord.com/active/docs/import/)
- [Workflows][encord-active-docs-workflow]
- [User Guides](https://docs.encord.com/active/docs/category/user-guide)
- [CLI Documentation](https://docs.encord.com/active/docs/cli)

## ⬇️  Download a sandbox dataset

Another way to start quickly is by downloading an existing dataset from the sandbox.
The download command will ask which pre-built dataset to use and will download it into a new directory in the current working directory.

```shell
encord-active download
cd /path/of/downloaded/project
encord-active visualize
```

The app should then open in the browser.
If not, navigate to [`localhost:8501`](http://localhost:8501).
Our [docs][encord-active-docs] contains more information about what you can see in the page.

## <img width="24" height="24" src="https://raw.githubusercontent.com/encord-team/encord-active/main/docs/static/img/icons/encord_icon.png"/> Import your dataset

### Quick import Dataset

To import your data (without labels) save your data in a directory and run the command:

```shell
# within venv
encord-active init /path/to/data/directory
```

A project will be created using the data in the directory.

To visualize the project run:

```shell
cd /path/to/project
encord-active visualize
```

You can find more details on the `init` command in the [documentation][encord-active-docs-init].

### Import from COCO

To import your data, labels, and predictions from COCO, save your data in a directory and run the command:

```shell
# install COCO extras
(ea-venv)$ python -m pip install encord-active[coco]

# import samples with COCO annotaions
encord-active import project --coco -i ./images -a ./annotations.json

# import COCO model predictions
encord-active import predictions --coco results.json
```

### Import from Encord Platform

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

- [Data Exploration](https://docs.encord.com/active/docs/workflows/understand-data-distribution)
- [Data Outlier detection](https://docs.encord.com/active/docs/workflows/identify-outliers-edge-cases)
- [Label Outlier detection](https://docs.encord.com/active/docs/workflows/identify-outliers-edge-cases#label-outliers)
- [Object Detection/segmentation Model Decomposition](https://docs.encord.com/active/docs/workflows/evaluate-detection-model)
- [Classification Model Decomposition](https://docs.encord.com/active/docs/workflows/evaluate-classification-model)
- [Similarity Search](https://docs.encord.com/active/docs/workflows/image-similarity)
- [Data Tagging](https://docs.encord.com/active/docs/user-guide/tags)
- [Visualize TP/FP/FN](https://docs.encord.com/active/docs/workflows/evaluate-detection-model#exploring-the-individual-samples)
- [COCO Exports](https://docs.encord.com/active/docs/user-guide/filter_export#export-to-coco-file)
- And much more!

Visit our [documentation][encord-active-docs] to learn more.

### Supported data:

| Data   |     | Labels          |     | Project sizes |               |
| ------ | --- | --------------- | --- | ------------- | ------------- |
| `jpg`  | ✅  | Bounding Boxes  | ✅  | Images        | 50.000        |
| `png`  | ✅  | Polygons        | ✅  | Videos        | 50.000 frames |
| `tiff` | ✅  | Segmentation    | ✅  |               |               |
| `mp4`  | ✅  | Classifications | ✅  |               |               |
|        |     | Polylines       | 🟡  |               |               |

## 🧑🏽‍💻Development

### 🛠 Build your own quality metrics

Encord Active is built with customizability in mind. Therefore, you can easily build your own custom metrics 🔧 See the [Writing Your Own Metric][encord-active-docs-write-metric] page in the docs for details on this topic. If you need help or guidance feel free to ping us in the **[Slack channel](https://encordactive.slack.com)**!

## 👪 Community and support

Join our channel on [Slack][join-slack] to connect with the team behind Encord Active.

Also, feel free to [suggest improvements or report problems][report-issue] via GitHub issues.

## 🎇 Contributions

If you're using Encord Active in your organization, please try to add your company name to the [ADOPTERS.md][adopters]. It really helps the project to gain momentum and credibility. It's a small contribution back to the project with a big impact.

If you want to share your custom metrics or improve the tool, please see our [contributing docs][contribute-url].

### 🦸 Contributors

<a href="https://github.com/encord-team/encord-active/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=encord-team/encord-active" />
</a>

[@Javi Leguina](https://github.com/jleguina)

## Licence

This repository is published under the Apache 2.0 licence.

[adopters]: https://github.com/encord-team/encord-active/blob/main/ADOPTERS.md
[colab-image]: https://colab.research.google.com/assets/colab-badge.svg
[colab-notebook]: https://colab.research.google.com/drive/11iZE1CCFIGlkWdTmhf5XACDojtGeIRGS?usp=sharing
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
[contribute-url]: https://docs.encord.com/active/docs/contributing
[docs-image]: https://img.shields.io/badge/docs-online-blue
[downloads-badge]: https://static.pepy.tech/badge/encord-active
[encord-active-docs-init]: https://docs.encord.com/active/docs/import/quick-import-data
[encord-active-docs-workflow]: https://docs.encord.com/active/docs/category/workflows
[encord-active-docs-write-metric]: https://docs.encord.com/active/docs/metrics/write-your-own
[encord-active-docs]: https://docs.encord.com/active/docs
[encord-active-landing]: https://encord.com/encord-active/
[encord-docs-ssh]: https://docs.encord.com/admins/settings/public-keys/#set-up-public-key-authentication
[homepage-visual]: ./homepage_visual.png
[join-slack]: https://join.slack.com/t/encordactive/shared_invite/zt-1hc2vqur9-Fzj1EEAHoqu91sZ0CX0A7Q
[license-image]: https://img.shields.io/github/license/encord-team/encord-active
[new-issue]: https://github.com/encord-team/encord-active/issues/new
[pypi-package-image]: https://img.shields.io/pypi/v/encord-active
[pypi-package]: https://www.piwheels.org/project/encord-active/
[pypi-version-image]: https://img.shields.io/pypi/pyversions/encord-active
[python-39]: https://www.python.org/downloads/release/python-3915/
[report-issue]: https://github.com/encord-team/data-quality-pocs/issues/new
[slack-community]: https://encord-active.slack.com
[slack-image]: https://img.shields.io/badge/Slack-4A154B?logo=slack&logoColor=white
[twitter-image]: https://img.shields.io/twitter/follow/encord_team?label=%40encord_team&style=social
[twitter-url]: https://twitter.com/encord_team
