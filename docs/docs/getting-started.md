---
sidebar_position: 1
slug: /
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Link from '@docusaurus/Link';

# Getting Started

**Encord Active** is an open-source Python library that enables you to improve computer vision models in an active learning fashion by improving your
[data quality](category/data-quality), [label quality](category/label-quality), and [model quality](category/model-quality).

## Install Encord Active

Install `encord-active` in your favorite Python environment with the following commands:

```shell
python3.9 -m venv ea-venv
source ea-venv/bin/activate
# within venv
pip install encord-active
```

:::tip

`encord-active` requires [python3.9](https://www.python.org/downloads/release/python-3915/) or above.
If you have trouble installing `encord-active`, you can find more detailed instructions on installing it [here](./installation).

:::

## Encord Active Quickstart

Understand Encord Active in **5 minutes** by playing!

The script will download a small example project to your current working directory and open the application straight away.
This is the fastest way to explore Encord Active.

```shell
# within venv
encord-active quickstart
```

This must be run in the same virtual environment where you installed your package.

The next section will show you how to download larger and more interesting datasets to explore.

## Sandbox Dataset

If you have more time, we have pre-built a few sandbox datasets with data, labels, and model predictions for you to start exploring Encord Active.

To get started quickly with a sandbox dataset, you can run the following command:

```shell
# within venv
encord-active download
```

This will allow you to choose a dataset to download. When the download process is complete, you visualize the results by following the printed instructions.

:::tip  
You can follow the [COCO sandbox dataset tutorial](tutorials/touring-the-coco-dataset.mdx) to learn the features of Encord Active.
:::

### Run Encord Active on Google Colab

If you want to quickly explore Encord Active without installing anything into your local machine, we
have the following Google Colab notebooks for you:

1. [Explore Encord Active sandbox dataset](https://colab.research.google.com/drive/11iZE1CCFIGlkWdTmhf5XACDojtGeIRGS?usp=sharing)
2. [Explore Encord Active through your own Encord projects](https://colab.research.google.com/drive/1zv4i0SH5tyb1KPVsCZfXDwxV72Ip77zS?usp=share_link)

## Import Your Own Data

To import your own data save your data in a directory and run the command:

```shell
# within venv
encord-active init /path/to/data/directory
```

A project will be created using the data (without labels) in the current working directory (unless used with `--target`).

To visualise the project run:

```shell
cd /path/to/project
encord-active visualise
```

You can find more details on the `init` command in the [CLI section](cli/initialising-project-from-image-directories).

## Import an Encord Project

If you are an Encord user, you can directly [import](cli/import-encord-project) your own projects into Encord Active easily.

```shell
# within venv
encord-active import project
```

This will import your encord project to a new directory in your current working directory.
If you don't have an Encord project ready, you can either

1. [Initialise a project from a local data directory](/cli/initialising-project-from-image-directories)
2. [Migrating data and labels to Encord](sdk/migrating-data) before calling this command
3. [Download one of our sandbox datasets](/cli/download-sandbox-data)

:::info

If you are new to the Encord platform, you can easily [sign up](https://app.encord.com/register) for an Encord account.

:::

To be able to import an Encord project, you will need the path to your private `ssh-key` associated with Encord (see documentation [here](https://docs.encord.com/admins/settings/public-keys/#set-up-public-key-authentication)).

The command will ask you:

1. `Where is your private ssh key stored?`: type the path to your private ssh key
2. `What project would you like to import?`: here, you can (fuzzy) search for the project title that you would like to import. Hit <kbd>enter</kbd> when your desired project is highlighted.

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

## What's Up Next?

We recommend to take a look at one of the [tutorials](category/tutorials) that demonstrate Encord Active's capabilities and the [workflows](category/workflows) section to learn about importing your model predictions and improving your model performance.
A couple of example references are:

1. [Import your model predictions](workflows/import-predictions)
2. Find outliers in your [data](workflows/improve-your-data/identify-outliers-edge-cases) or your [labels](workflows/improve-your-labels/identify-outliers)
3. [Identify metrics](workflows/improve-your-models/metric-importance) that are important for your model performance

You can also have a look at how to [write custom metrics](/metrics/write-your-own) and how to use the [command line interface](https://encord-active-docs.web.app/category/command-line-interface).

### Need Support?

If you got any issues with Encord Active you are more than welcome to [connect with us on Slack](https://join.slack.com/t/encordactive/shared_invite/zt-1hc2vqur9-Fzj1EEAHoqu91sZ0CX0A7Q) or reach out to us at active@encord.com
