<h1 align="center">
  <p align="center">Encord Active</p>
  <a href="https://encord.com"><img src="src/encord_active/app/assets/encord_2_02.png" width="150" alt="Encord logo"/></a>
</h1>

[!["Join us on
Slack"](https://img.shields.io/badge/Slack-4A154B?logo=slack&logoColor=white)][join-slack] [![Twitter
Follow](https://img.shields.io/twitter/follow/encord_team?label=%40encord_team&style=social)][twitter-url]
[![PRs-Welcome][contribute-image]][contribute-url]

## Documentation

Please refer to our [documentation][encord-active-docs].

## Installation

The simplest way to install the CLI is using `pip` in a suitable virtual environment:

```shell
pip install encord-active
```

We recommend using a virtual environment, such as `venv`:

```shell
$ python3.9 -m venv ea-venv
$ source ea-venv/bin/activate
$ pip install encord-active
```

> `encord-active` requires [python3.9](https://www.python.org/downloads/release/python-3915/).
> If you have trouble installing `encord-active`, you find more detailed instructions on
> installing it [here][encord-active-docs].

## Downloading a pre-built project

The quickest way to get started is by downloading an existing dataset.
The download command will setup a directory where the project will be stored and will ask which pre-built dataset to use.

```shell
$ encord-active download
$ encord-active visualise /path/to/project
```

The app should automatically open in the browser. If not, navigate to `localhost:8501`.
Our [Docs][encord-active-docs] contain more information about what you can see in the page.

## Importing an Encord Project

This section requires setting up an ssh key with Encord, so slightly more technical.

> If you haven't set up an ssh key with Encord, you can follow the tutorial in this [link](https://docs.encord.com/admins/settings/public-keys/#set-up-public-key-authentication)

To import an Encord project, use this script:

```shell
$ encord-active import project
```

## Development

### Write your own metrics

Encord Active isn't limited to the metrics we provided, it is actually quite easy to write your own 🔧
See the [Writing Your Own Metric](https://docs.encord.com/admins/settings/public-keys/#set-up-public-key-authentication) page on the WIKI for details on this.

### Pre-commit hooks

If you have installed the dependencies with poetry, then you can install pre-commit hooks by running the following command:

```shell
$ pre-commit install
```

The effect of this will be that `black`, `isort`, `mypy`, and `pylint` needs to run without finding issues with the code before you are allowed to commit.
When you commit and either `black` or `isort` fails, committing again is enough, as the side effect of committing the first time is to reformat files.

Running each of the tools individually on your code can be done as follows:

```shell
$ poetry run black --config=pyproject.toml .
$ poetry run isort --sp=pyproject.toml .
$ poetry run mypy . --ignore-missing-imports
$ poetry run pylint -j 0 --rcfile pyproject.toml [subdir]
```

## Community and Support

Join our community on [Slack][join-slack] and [Twitter][twitter-url]!

[Suggest improvements and report problems][new-issue]

# Contributions

If you're using Encord Active in your organization, please try to add your company name to the [ADOPTERS.md](./ADOPTERS.md). It really helps the project to gain momentum and credibility. It's a small contribution back to the project with a big impact.

Read the [contributing docs][contribute-url].

# Licence

This project is licensed under the terms of the AGPL-3.0 license.

[encord-active-docs]: https://encord-active-docs.web.app/
[contribute-url]: https://encord-active-docs.web.app/contributing
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
[join-slack]: https://join.slack.com/t/encordactive/shared_invite/zt-1hc2vqur9-Fzj1EEAHoqu91sZ0CX0A7Q
[twitter-url]: https://twitter.com/encord_team
[new-issue]: https://github.com/encord-team/encord-active/issues/new
