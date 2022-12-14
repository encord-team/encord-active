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

## Check the Installation

To check that Encord Active is installed, run:

```shell
# within venv
encord-active --help
```

This must be run in the same virtual environment where you installed your package.

The `--help` option provides some context to what you can do with `encord-active`.

To learn more about how to use the command line interface, see the [Command Line Interface section](category/command-line-interface).

## Quickstart

To get started quickly, you run the following command.

```shell
# within venv
encord-active quickstart
```

The script will download a small example project to your current working directory and open the application straight away.
The next section will show you how to download larger and, perhaps, more interesting datasets to explore.

## Sandbox Data

To get started quickly with a sandbox dataset, you run the following command:

```shell
# within venv
encord-active download
```

This will allow you to choose a dataset to download.
The dataset will be downloaded to a new directory in your current working directory.

When the download process is complete, you visualise the results by following the printed instructions or following the steps in [Running the App](#running-the-app) section below.

## Importing Data from Encord

If you already have a project on the Encord platform, you can import that project with the following command:

```shell
# within venv
encord-active import project
```

This will import your encord project to a new directory in your current working directory.

:::tip

If you don't have an Encord project ready, you can find your next steps in the SDK section [Migrating Data to Encord](sdk/migrating-data).
Otherwise, you can [download one of our sandbox datasets](/cli/download-sandbox-data).

:::

To be able to do this, you will need the path to your private `ssh-key` associated with Encord and a `project_hash`.
Don't worry! The script will guide you on the way if you don't know it already.

The script will ask you:

1. `Where is your private ssh key stored?`: type the path to your private ssh key
2. `Specify project hash`: paste / type the `project_hash`. If you don't know the id, you can type `?` and hit enter to get all your projects listed with their `project_hash`s before being prompted with the question again. Now you can copy paste the id.

Next, `encord-active` will fetch your data and labels before computing all the [metrics](category/metrics) available in `encord-active`.

Downloading the data and computing the metrics may take a while.
Bare with us, it is worth the wait.

When the process is done, follow the printed instructions to open the app or see more details in the [Open Encord Active](/cli/open-encord-active) page.

## Running the App

To run the Encord Active app, you need to `cd` into the directory that was created by one of the previous commands and run the following command:

```shell
# within venv
cd /path/to/project
encord-active visualise
```

Now, your browser should open a new window with Encord Active.

:::caution

If the script just seems to hang and nothing happens in your browser, try visiting <Link to={"http://localhost:8501"}>http://localhost:8501</Link>.

:::

### What's Up Next?

We recommend having a look at the [workflows](category/workflows) section to learn about importing your model predictions and improving your model performance.
A couple of example references are:

1. [Import your model predictions](workflows/import-predictions)
2. Find outliers in your [data](workflows/improve-your-data/identify-outliers-edge-cases) or your [labels](workflows/improve-your-labels/identify-outliers)
3. [Identify metrics](workflows/improve-your-models/metric-importance) that are important for your model performance

You can also have a look at how to [write custom metrics](/metrics/write-your-own) and how to use the [command line interface](https://encord-active-docs.web.app/category/command-line-interface).
