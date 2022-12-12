---
sidebar_position: 1
slug: /
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Link from '@docusaurus/Link';

# Getting Started

**Encord Active** is an open-source Python library that enables users to improve machine learning models in an active learning fashion via
[data quality](category/data-quality), [label quality](category/label-quality), and [model assertions](category/model-assertions).

## Install `encord-active`

Install `encord-active` in your favorite Python environment with the following commands:


```shell
python3.9 -m venv ea-venv
source ea-venv/bin/activate
# within venv
pip install encord-active
```

:::tip

`encord-active` requires [python3.9](https://www.python.org/downloads/release/python-3915/).
If you have trouble installing `encord-active`, you find more detailed instructions on installing it [here](./installation).

:::

## Say Hello to Encord Active

Understand Encord-Active in **5 minutes** by playing!

The following command will start the Encord Active with a demo project where you can play around.
This is the fastest way to explore Encord Active.

```shell
# within venv
encord-active hello
```

This must be run in the same virtual environment where you installed your package.

## Sandbox Data

If you have more time, we have sandbox projects where you can better explore the Encord Active. To get started quickly 
with a sandbox dataset, you can run the following command:

```shell
# within venv
encord-active download
```

The script will ask you to

1. `Where should we store the data?`: specify a directory in which the data should be stored.
2. (potentially) `Directory not existing, want to create it? [y/N]` type <kbd>y</kbd> then <kbd>enter</kbd>.
3. `[?] Choose a project:` choose a project with <kbd>↑</kbd> and <kbd>↓</kbd> and hit <kbd>enter</kbd>

:::tip

If you plan on using multiple datasets, it may be worth first creating an empty directory for all the datasets.

```shell
mkdir /path/to/data/root
# or windows
md /path/to/data/root
```

In step 1. above, specify, e.g., `/path/to/data/root/sandbox1`

:::

When the download process is complete, you visualize the results by following the printed instructions.

:::tip  
You can follow the [COCO dataset tutorial](tutorials/touring-the-coco-dataset.mdx) to learn the features
of te Encord Active.
:::

### Run Encord Active on Google Colab

If you want to quickly explore Encord-Active without installing anything into your local machine, we
have the following Google Colab notebooks for you:
1. [Explore Encord-Active through sandbox projects](https://colab.research.google.com/drive/11iZE1CCFIGlkWdTmhf5XACDojtGeIRGS?usp=sharing)
2. [Explore Encord-Active through your own Encord projects](https://colab.research.google.com/drive/1zv4i0SH5tyb1KPVsCZfXDwxV72Ip77zS?usp=share_link)

## What's Up Next?

If you are an Encord user, you can directly [import](cli/import-encord-project) your own projects to the Encord-Active
easily.

If you are new to the Encord platform, [sign up](https://app.encord.com/register) for an Encord account and 
[upload your projects](sdk/migrating-data) to the Encord platform. Then you can easily import your
projects.

:::tip

We recommend having a look at the [workflows](category/workflows) section to learn about importing your model predictions and improving your model performance.
A couple of example references are: 

1. [Import your model predictions](workflows/import-predictions)
2. Find outliers in your [data](workflows/improve-your-data/identify-outliers-edge-cases) or your [labels](workflows/improve-your-labels/identify-outliers)
3. [Identify metrics](workflows/improve-your-models/metric-importance) that are important for your model performance

You can also have a look at how to [write custom metrics](/metrics/write-your-own) and how to use the [command line interface](https://encord-active-docs.web.app/category/command-line-interface).
:::


