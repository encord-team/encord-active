---
sidebar_position: 3
---

# From Encord Annotate

**Pull projects from Encord Annotate**

If you already have a project in Encord Annotate, you can start with its respective Encord Active project right away.

:::info
If you are new to the Encord platform, [signing up][encord-sign-up] for an Encord account is quick and easy.
:::

To interactively select a project from the list of available projects in Encord Annotate use the following command:

```shell
encord-active import project
```

To narrow down the search for the project you wish to import, enter some text that matches the project title.
Use the keyboard arrows to navigate and select the desired project, then press <kbd>Enter</kbd> to confirm your choice.

Alternatively, if you prefer to override the selection process, you can use the `--project-hash` option when executing the command.

You will get a directory containing all the data, labels, and [metrics](/category/quality-metrics) of the project. You have the option to choose whether to store the data in the local file system and can opt-in or opt-out accordingly.

When the process is done, follow the printed instructions to open the app or see more details in the [start](../cli#start) CLI command.

If you are importing an Encord Annotate project for the first time, the Command Line Interface (CLI) will prompt you to provide the local path of a private SSH key associated with Encord.
To associate an SSH key with Encord, please refer to the documentation available [here][encord-docs-ssh].
The provided SSH key path will be stored for future use.

:::note
The previous command imports a project into a new folder within the current working directory. However, if a different directory needs to be specified, the `--target` option can be included as follows:

```shell
encord-active import project --target /path/to/store/project
```

This will import the project in a subdirectory of `/path/to/store/project`.
:::


[encord-docs-ssh]: https://docs.encord.com/admins/settings/public-keys/#set-up-public-key-authentication
[encord-sign-up]: https://app.encord.com/register
