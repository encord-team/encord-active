<p align="center">
<a href="https://docs.encord.com/active/docs" target="_blank">Documentation</a> |
<a href="https://colab.research.google.com/drive/11iZE1CCFIGlkWdTmhf5XACDojtGeIRGS?usp=sharing" target="_blank">Try it Now</a> |
<a href="https://encord.com/encord_active/" target="_blank">Website</a> |
<a href="https://encord.com/blog/" target="_blank">Blog</a> |
<a href="https://discord.gg/TU6yT7Uvx3" target="_blank">Discord Community</a>
</p>

<h1 align="center">
  <p align="center">Encord Active</p>
  <a href="https://encord.com"><img src="/src/encord_active/app/assets/encord_2_02.png" width="150" alt="Encord logo"/></a>
</h1>

[![PRs-Welcome][contribute-image]][contribute-url]
![Licence][license-image]
[![PyPi project][pypi-package-image]][pypi-package]
![PyPi version][pypi-version-image]
[![docs][docs-image]][encord-active-docs]
[!["Join us on Discord"][discord-image]][join-discord]
[![Twitter Follow][twitter-image]][twitter-url]

In this directory, you find examples of how to use Encord Active.

Generally, you need to have installed [`encord-active`](https://docs.encord.com/active/docs) and [`jupyter-lab`](#jupyter-lab).

## Table of Contensts

- [Jupyter Lab](#jupyter-lab)
  - [Installation](#installation)
  - [Running Jupyter Lab](#running-jupyter-lab)
- [Notebooks](#notebooks)
  - [Getting Started](#getting-started)
  - [Writing Custom Metrics](#writing-custom-metrics)
- [Google Colab](#google-colab)
- [Community and support](#community-and-support)
- [Contributions](#contributions)

## Jupyter Lab

We like using [`jupyter-lab`](https://jupyter.org/install) for running notebooks.

### Installation

With [`pyenv`](https://github.com/pyenv/pyenv), you can get started with this set of instructions:

```shell
$ cd /path/to/where/your/notebooks/should/live
$ # This could be /path/to/encord-active/examples

$ pyenv local 3.10
$ python -m venv venv
$ source venv/bin/activate
(venv) $ python -m pip install --upgrade pip

(venv) $ python -m pip install encord-active[notebook]
(venv) $ jupyter nbextension enable --py --sys-prefix widgetsnbextension
```

> **Note**
> The `ipywidgets` module is used to better display progress bars and not strictly necessary.
> You don't have to install it.
> If you don't, you can ignore the last line above.

### Running Jupyter Lab

To run Jupyter Lab, simply do

```shell
(venv) $ jupyter-lab
```

and a new browser window should open.

> **Note**
> If your current working directory does not contain the examples in this directory, you will have to copy the ones you need into your current working directory.

## Notebooks

### Getting Started

These notebooks show you how to getting started with importing data into Encord Active

- [Downloading a Sandbox Dataset](download-sandbox-dataset.ipynb)
- [Importing an Existing Encord Project](getting-started-with-encord-projects.ipynb)
- [Importing an Existing COCO Project](getting-started-with-coco-project.ipynb)

### Writing Custom Metrics

Here, you will learn more about how to write your own custom metric functions to index your data.

- [Building Your Own Custom Metric](building-a-custom-metric-function.ipynb)

For more information, please refer to our [documentation]([encord-active-docs-write-metric]) on the topic.

### Custom Embeddings

Here, you will learn more about how to add your own custom embeddings to improve your similarity search within Encord Active.

- [Adding Your Own Custom Embeddings](adding-own-custom-embeddings.ipynb)

## Google Colab

We also provide a couple of notebooks, which can be run on the internet without having to install anything on your own system.
However, the data that you interact with in these notebooks will not persist if you revisit them later!

To use the notebooks, you will have to set up an Ngrok account in order to be able to run the Encord Active app. But don't worry, it is easy and well described in the notebooks.

- [Downloading a Sandbox Dataset ![Google Colab][colab-image]](https://colab.research.google.com/drive/11iZE1CCFIGlkWdTmhf5XACDojtGeIRGS?usp=share_link)
- [Importing an Existing Encord Project ![Google Colab][colab-image]](https://colab.research.google.com/drive/1zv4i0SH5tyb1KPVsCZfXDwxV72Ip77zS?usp=share_link)
- [Analyze a TorchVision Dataset ![Google Colab][colab-image]](https://colab.research.google.com/drive/1x44IGQBUgz9jIKIooNGV8ZEnGwf73hxZ?usp=share_link)
- [Analyze a HuggingFace Dataset ![Google Colab][colab-image]](https://colab.research.google.com/drive/1Ohsd1BrO6s9HuliYdHqMsIblaR9KXbpk?usp=sharing)

## Community and Support

Join our community on [Discord][join-discord] to connect with the team behind Encord Active.

Also, feel free to [suggest improvements or report problems][report-issue] via github issues.

## Contributions

If you're using Encord Active in your organization, please try to add your company name to the [ADOPTERS.md][adopters]. It really helps the project to gain momentum and credibility. It's a small contribution back to the project with a big impact.

If you want to share your notebooks, custom metrics, or improve the tool, please see our [contributing docs][contribute-url].

[adopters]: https://github.com/encord-team/encord-active/blob/main/ADOPTERS.md
[colab-image]: https://colab.research.google.com/assets/colab-badge.svg
[colab-notebook]: https://colab.research.google.com/drive/11iZE1CCFIGlkWdTmhf5XACDojtGeIRGS?usp=sharing
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
[contribute-url]: https://docs.encord.com/active/docs/contributing
[docs-image]: https://img.shields.io/badge/docs-online-blue
[encord-active-docs-write-metric]: https://docs.encord.com/active/docs/metrics/write-your-own
[encord-active-docs]: https://docs.encord.com/active/docs
[join-discord]: https://discord.gg/TU6yT7Uvx3
[license-image]: https://img.shields.io/github/license/encord-team/encord-active
[pypi-package-image]: https://img.shields.io/pypi/v/encord-active
[pypi-package]: https://www.piwheels.org/project/encord-active/
[pypi-version-image]: https://img.shields.io/pypi/pyversions/encord-active
[report-issue]: https://github.com/encord-team/data-quality-pocs/issues/new
[discord-image]: https://shields.io/badge/Discord-chat-blue?logo=discord&logoColor=white
[twitter-image]: https://img.shields.io/twitter/follow/encord_team?label=%40encord_team&style=social
[twitter-url]: https://twitter.com/encord_team
