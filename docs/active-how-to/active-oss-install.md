---
title: "Install Active OS"
slug: "active-oss-install"
hidden: false
metadata: 
  title: "Install Active OS"
  description: "Installation guide for Encord Active OS. Streamline setup. Get started with Encord installation."
category: "65a71bbfea7a3f005192d1a7"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/local_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

## Prerequisites

> ℹ️ Note
Make sure you have [python 3.9](https://www.python.org/downloads/release/python-3917/) and [Git VCS](https://git-scm.com/download) installed on your system.

To install the correct version of python you can use [pyenv](https://github.com/pyenv/pyenv), [brew (mac only)](https://formulae.brew.sh/formula/python@3.9) or simply [download](https://www.python.org/downloads/release/python-3917/) it.


## From PyPi

Install `encord-active` in your favorite Python environment using the following commands:

```shell Linux/macOS
python3.9 -m venv ea-venv
source ea-venv/bin/activate
pip install encord-active
```
```shell Windows
python -m venv ea-venv
ea-venv\Scripts\activate
pip install encord-active
```

### COCO extras

If you intend to work with files using COCO format you'll have to install Encord Active with an extra dependency:

```shell
pip install encord-active[coco]
```

> ℹ️ Note
You might need to install `xcode-select` if you are on Mac or `C++ build tools` if you are on Windows.


## Check the Installation

To check what version of Encord Active is installed, run:

```shell
$ encord-active --version
```

This command must be run in the same virtual environment where you installed the package.

The `--help` option provides some context to what you can do with `encord-active`. If you'd like to explore the available commands in the Command Line Interface (CLI), you can refer to the [CLI section](https://docs.encord.com/docs/active-cli) for detailed information.

## Docker

We also provide a docker image that works exactly as the CLI.

```shell
docker run -it --rm -p 8000:8000 -v ${PWD}:/data encord/encord-active <command>
```

Running the previous command will mount your current working directory, so everything that happens inside the docker container will persist after it is done.

### SSH key

If you intend to use Encord Active OS with an Encord Annotate project you'll need to mount a volume with your SSH key as well.

```shell
docker run -it --rm -p 8000:8000 -v ${PWD}:/data -v ${HOME}/.ssh:/root/.ssh encord/encord-active
```

When asked for your SSH key, you can point to `~/.ssh/<your-key-file>`.