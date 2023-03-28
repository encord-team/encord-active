---
sidebar_position: 8
---

# Command Line Interface

Encord Active ships with a command-line interface to help you interact with Encord Active.
With the CLI your can initialize projects, import labels, import existing, run metrics and start the application.

We do our best to make our CLI self explainable so you don't need to jump between your terminal and the documentation.

Simply run `encord-active --help` to get details about all the top-level commands and `encord-active COMMAND --help`

Here is a list of all the top-level commands:

```
quickstart         Start Encord Active straight away 🏃💨
download           Download a sandbox dataset to get started. 📁
init               Initialize a project from your local file system 🌱
import             Import Projects or Predictions ⬇️
visualize          Launch the application with the provided project ✨
metric             Manage project's metrics.
metricize          Execute metrics on your data and predictions 🧠
print              Print useful information 🖨️
config             Configure global settings 🔧
```

## `quickstart`

The command will download a small example project to a subdirectory called `quickstart` in current working directory and open the application straight away.

```
Usage: encord-active quickstart [OPTIONS]

Options:
 --target   -t  DIRECTORY  Directory where the project would be saved.
```

## `download`

In addition to the `quickstart` example, there are a few more examples of open source datasets you can download and use to explore Encord Active capabilities.

```
# List the available projects and choose which one to download
Usage: encord-active download [OPTIONS]

Options:
 --project-name          TEXT       Name of the chosen project.
 --target        -t      DIRECTORY  Directory where the project would be saved.
```

<details>
  <summary>List of downloadable sandbox projects </summary>

#### Berkeley Deep Drive

- **Research Paper:** BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning
- **Authors:** Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, Trevor Darrell
- **Dataset Size:** 1000 images & 12973 annotations
- **Categories:** 8 classes
- **License:** BSD 3-Clause License
- **Release:** 21st September, 2020
- **Read more:** [Webpage](https://bdd-data.berkeley.edu/) & [GitHub](https://github.com/bdd100k/bdd100k)

Sample pictures:
![BDD dataset](./images/BDD.png)

#### COCO Validation 2017 Dataset

- **Research Paper:** [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312)
- **Author:** [Tsung-Yi Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+T) , [Michael Maire](https://arxiv.org/search/cs?searchtype=author&query=Maire%2C+M), [Serge Belongie](https://arxiv.org/search/cs?searchtype=author&query=Belongie%2C+S), [Lubomir Bourdev](https://arxiv.org/search/cs?searchtype=author&query=Bourdev%2C+L) , [Ross Girshick](https://arxiv.org/search/cs?searchtype=author&query=Girshick%2C+R), [James Hays](https://arxiv.org/search/cs?searchtype=author&query=Hays%2C+J), [Pietro Perona](https://arxiv.org/search/cs?searchtype=author&query=Perona%2C+P), [Deva Ramanan](https://arxiv.org/search/cs?searchtype=author&query=Ramanan%2C+D), [C. Lawrence Zitnic](https://arxiv.org/search/cs?searchtype=author&query=Zitnick%2C+C+L), [Piotr Dollár](https://arxiv.org/search/cs?searchtype=author&query=Doll%C3%A1r%2C+P)
- **Dataset Size:**  5000 images, 4784 annotations
- **Categories:** 81 classes
- **License:** CC BY 4.0
- **Release:** 1st, May, 2014
- **Read More:** [GitHub](https://github.com/cocodataset/cocodataset.github.io) & [Webpage](https://cocodataset.org/#home)

Sample pictures:
![COCO dataset](./images/COCO.png)

#### Covid 19 Segmentation Dataset

- **Research Paper:** Unknown
- **Author:** Unknown
- **Dataset Size:** 100 images & 602 annotations
- **Categories:**  13 classes
- **License:** CC BY 4.0
- **Release:** Unknown
- **Read more:** [GitHub](https://github.com/GeneralBlockchain/covid-19-chest-xray-segmentations-dataset)

Sample pictures:
![Covid dataset](./images/Covid.png)

#### Rareplanes

- **Research Paper:** [RarePlanes: Synthetic Data Takes Flight](https://arxiv.org/abs/2006.02963)
- **Author:** [Jacob Shermeyer](https://arxiv.org/search/cs?searchtype=author&query=Shermeyer%2C+J), [Thomas Hossler](https://arxiv.org/search/cs?searchtype=author&query=Hossler%2C+T), [Adam Van Etten](https://arxiv.org/search/cs?searchtype=author&query=Van+Etten%2C+A), [Daniel Hogan](https://arxiv.org/search/cs?searchtype=author&query=Hogan%2C+D), [Ryan Lewis](https://arxiv.org/search/cs?searchtype=author&query=Lewis%2C+R), [Daeil Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+D)
- **Dataset Size:** 2710 images & 6812 annotations
- **Categories:**  7 plane categories
- **License:** CC 4.0 BY SA
- **Release: 4** June, 2020
- **Read More:** [Webpage](https://www.cosmiqworks.org/rareplanes/)

Sample pictures:
![Rareplanes dataset](./images/Rareplanes.png)

#### TACO Dataset

- **Research Paper:** [TACO: Trash Annotations in Context for Litter Detection](https://arxiv.org/abs/2003.06975)
- **Author:** Pedro F Proença, Pedro Simões
- **Dataset Size:** Official: 1500 images, 4784 annotations & Unofficial: 3736 images, 8419 annotations
- **Categories:** 60 litter categories
- **License:** CC BY 4.0
- **Release:** 17th, March, 2020
- **Read More:** [GitHub](https://github.com/pedropro/TACO) & [Webpage](http://tacodataset.org/)

Sample pictures:
![TACO dataset](./images/TACO.png)

#### Limuc Ulcerative Colitis Classification

- **Research Paper:** Improving the Computer-Aided Estimation of Ulcerative Colitis Severity According to Mayo Endoscopic Score by Using Regression-Based Deep Learning
- **Authors:** Gorkem Polat, MSc, Haluk Tarik Kani, MD, Ilkay Ergenc, MD, Yesim Ozen Alahdab, MD, Alptekin Temizel, PhD, Ozlen Atug, MD
- **Dataset Size:** 11276 images
- **Categories:** Medical (Endoscopy/Colonoscopy)
- **License:** Creative Commons Attribution 4.0 International
- **Release:** 14th March 2022
- **Read more:** [Webpage](https://www.notion.so/Jan-9-2023-c23ea70fe8b64b889dc8b8ad4f77ad18) & [GitHub](https://github.com/GorkemP/labeled-images-for-ulcerative-colitis)

</details>

## `visualize`

Launch the application with the provided project ✨

```
Usage: encord-active quickstart [OPTIONS]

Options:
 --target   -t  DIRECTORY  Path of the projects you would like to visualize
```

## `init`

The command initializes new project from locally stored images and labels. It will search for images based on the `data-glob` arguments. By default, all jpeg, jpg, png, and tiff files will be matched.

It will optionally search for labels as well if `label-glob`and `transformer` options are provided.

Both glob results will be passed to your implementation of the `LabelTransformer` interface if you provide a
`transformer` argument.

```
Usage: encord-active init [OPTIONS] ROOT

Arguments:
 * root                 DIRECTORY  The root directory of the dataset you are trying to import

Options:
 --data-glob    -dg     TEXT       Glob pattern to choose files. Repeat the `--data-glob` argument to
                                   match multiple globs.
                                   [default: **/*.jpg, **/*.png, **/*.jpeg, **/*.tiff]
 --label-glob   -lg     TEXT       Glob pattern to choose label files. Repeat the `--label-glob`
                                   argument to match multiple globs. This argument is only used if you
                                   also provide the `transformer` argument [default: None]
 --target       -t      DIRECTORY  Directory where the project would be saved. [default: CWD]
 --name         -n      TEXT       Name to give the new project. If no name is provided, the root
                                   directory will be used with '[EA] ' prepended.
 --symlinks                        Use symlinks instead of copying images to the target directory.
 --dryrun                          Print the files that will be imported WITHOUT importing them.
 --no-metrics                      Skip metrics execution on the initiated project.
 --transformer          PATH       Path to python module with one or more implementations of the
                                   `encord_active.lib.labels.label_transformer.LabelTransformer`
                                   interface
                                   [default: None]
```

The [Initialising from local data](./import/quick-import-data) workflow is a great starting point to using this command.

## `import`

The command is used to import projects and predictions from various sources.

Concrete usage example can be found [here](./import).

```
Usage: encord-active import [OPTIONS] COMMAND [ARGS]...

Import Projects or Predictions ⬇️

Commands:
 predictions  Imports a predictions file. The predictions should be using the `Prediction` model
              and be stored in a pkl file.
 project      Imports a new project from Encord or a local coco project 📦
```

### `project`

Imports a new project from Encord or a local coco project.

```
Usage: encord-active import project [OPTIONS]

Encord Project Arguments:
 --project-hash      TEXT       Encord project hash of the project you wish to import.
                                Leaving it blank will allow you to choose one interactively.
COCO Project Arguments:
 --coco                         Import a project from the coco format
 --images       -i   DIRECTORY  Path to the directory containing the dataset images.
 --annotations  -a   FILE       Path to the file containing the dataset annotations.
 --symlinks                     Use symlinks instead of copying COCO images to the target directory.

Options:
 --target  -t      DIRECTORY  Directory where the project would be saved. [default: CWD]
```

### `predictions`

Imports a predictions file. The predictions should be using the `Prediction` model and be stored in a pkl file.
If `--coco` option is specified the file should be a json following the coco results format.

Concrete usage example can be found [here](./import/import-predictions).

```
Usage: encord-active import predictions [OPTIONS] PREDICTIONS_PATH

Arguments:
 * predictions_path   FILE  Path to a predictions file. [required]

Options:
 --target  -t         DIRECTORY  Path to the target project. [default: CWD]
 --coco                          Import a coco result format file
```

## `metric`

Manage project's metrics.

```
Usage: encord-active metric [OPTIONS] COMMAND [ARGS]...

Commands:
 add                  Add metrics.
 list                 List metrics.
 remove               Remove metrics.
 run                  Run metrics.
 show                 Show information about available metrics.
```

### `add`

Add metrics to the project by specifying a path to a metrics module and titles of metrics within the module. If no metric titles are provided then all metrics found in the python module will be added to the project.

:::info
Make sure your shell's current working directory is that of an Encord Active project, or your command points to one with the `--target` global option.
:::

```
Usage: encord-active metric add [OPTIONS] MODULE_PATH [METRIC_TITLE]...


Arguments:
 *  module_path       FILE               Path to the python module where the metric resides. [required]
    metric_title      [METRIC_TITLE]...  Title of the metric. Can be used multiple times.

Options:
 --target  -t         DIRECTORY  Path to the target project. [default: CWD]
```

If you try to add a metric that is already present, it will be skipped and you will receive a notification.
In case a metric title is not found in the python module, you will get an error.

:::caution
Some shells may treat square braces (`[` and `]`) as special characters. It is suggested to always quote arguments containing these characters to prevent unexpected shell expansion.
:::

### `remove`

Removes metrics from a project.

```
Usage: encord-active metric remove [OPTIONS] METRIC_TITLE...

Arguments:
 * metric_title      METRIC_TITLE...  Title of the metric. Can be used multiple times. [required]

Options:
 --target  -t        DIRECTORY  Path to the target project. [default: CWD]
```

### `list`

List metrics in the project, including editables. Metrics are listed in a case-insensitive sorted order.

```
Usage: encord-active metric list [OPTIONS]

Options:
 --target  -t        DIRECTORY  Path to the target project. [default: CWD]
```

### `run`

Run metrics on the project's data and labels.

```
Usage: encord-active metric run [OPTIONS] [METRIC_TITLE]...

Arguments:
 metric_title      [METRIC_TITLE]...  Title of the metric. Can be used multiple times.

Options:
 --target  -t      DIRECTORY  Path to the target project. [default: CWD]
 --all                        Run all metrics.
 --fuzzy                      Enable fuzzy search in the selection. (press [TAB] or [SPACE] to select more than one) 🪄

```

### `show`

Show information about one or more available metrics in the project.

```
Usage: encord-active metric show [OPTIONS] METRIC_TITLE...

Arguments:
 * metric_title      METRIC_TITLE...  Title of the metric. Can be used multiple times. [required]

Options:
 --target  -t        DIRECTORY  Path to the target project. [default: CWD]

```

## `config`

Encord Active keeps some configurable properties to prevent repetitive input prompts.

The config file is stored at:

- Linux: `~/.config/encord-active/config.toml`
- MacOS: `~/Library/Application Support/encord-active/config.toml`
- Windows: `%APPDATA%/encord-active/config.toml`

```toml
ssh_key_path = "/absolute/path/to/ssh-key" # A key to use when accessing Encord projects
```

```
Usage: encord-active config [OPTIONS] COMMAND [ARGS]...

Commands:
 get           Print the value of an Encord Active configuration property.
 list          List Encord Active configuration properties.
 set           Sets an Encord Active configuration property.
 unset         Unsets the value of an Encord Active configuration property.
```

## `print`

Print useful information.

```

Usage: encord-active print [OPTIONS] COMMAND [ARGS]...

Commands:
 data-mapping           Prints a mapping between `data_hashes` and their corresponding `filename`
 encord-projects        Print the mapping between `project_hash`es of your Encord projects and their titles.
 ontology               Prints an ontology mapping between the class name to the `featureNodeHash` JSON format.
 system-info            Prints the information of the system for the purpose of bug reporting.

Options:
 --json                 Save output to a json file.
```
