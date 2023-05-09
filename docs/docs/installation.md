---
sidebar_position: 3
---

# Installation

:::info

Make sure you have `python3.9` and [git](https://git-scm.com/download) VCS installed on your system.

To install the correct version of python you can use [pyenv](https://github.com/pyenv/pyenv), [brew (mac only)](https://formulae.brew.sh/formula/python@3.9) or simply [download](https://www.python.org/downloads) it.

:::

## From PyPi

Install `encord-active` in your favorite Python environment with the following commands:

```shell
$ python3.9 -m venv ea-venv
$ # On Linux/MacOS
$ source ea-venv/bin/activate
$ # On Windows
$ ea-venv\Scripts\activate
(ea-venv)$ python -m pip install encord-active

```

#### Coco extras

If you intend to work on coco related things you'll have to install extra dependencies this way:

```shell
(ea-venv)$ python -m pip install encord-active[coco]
```

:::info
You might need to install `xcode-select` if you are on Mac or `C++ build tools` if you are on Windows.
:::

## Check the Installation

To check that Encord Active is installed, run:

```shell
$ encord-active --help
```

This must be run in the same virtual environment where you installed your package.

The `--help` option provides some context to what you can do with `encord-active`.

To learn more about how to use the command line interface, see the [Command Line Interface section](./cli).

## Docker

We also provide a docker image which works exctly the same as the cli.

```shell
docker run -it --rm -p 8501:8501 -p 8000:8000 -v ${PWD}:/data encord/encord-active <command>
```

This will mount your current working directory, so everything that happens inside the docker container will persist after it is done.

#### SSH key

If you intend to use Encord Active with an Encord Annotate project you'll need to mount a voulume with your SSH key as well.

```shell
docker run -it --rm -p 8501:8501 -p 8000:8000 -v ${PWD}:/data -v ${HOME}/.ssh:/root/.ssh encord/encord-active
```

Then, when asked for your SSH key, you can point to `~/.ssh/<your-key-file>`