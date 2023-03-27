---
sidebar_position: 4
---

# Importing Encord Project

**For projects already in the Encord Annotate platform.**

This workflow will get you through importing your data from Encord Annotate to a local Encord Active project.
You will get a directory containing all the data, labels, and [metrics](/category/quality-metrics) of the project.

To import your project, run this command:

```shell
encord-active import project
```

:::note

This will import a project to new folder in your current working directory. If you prefer to specify a different directory, use:

```shell
encord-active import project --targe /path/to/store/project
```

This will import the project into a subdirectory of `/path/to/store/project`.

:::

:::tip

If you don't have an Encord Annotate project already, you can find other ways to import your local data [here](../importing-data-and-labels).

:::

:::info

If you are new to the Encord platform, you can easily [sign up](https://app.encord.com/register) for an Encord account.

:::

To be able to import an Encord project, you will need the path to your private `ssh-key` associated with Encord (see documentation [here](https://docs.encord.com/admins/settings/public-keys/#set-up-public-key-authentication)).

If this is the first time you import a project, the command line interface (CLI) will ask you for your ssh key.
To associate an ssh key with Encord, you can follow [the documentation here][encord-docs-ssh].
You ssh key path will be stored for later reuse.

Next, the CLI will ask you what project to import based on all the projects you have access to at Encord.
You can type in a search word to find the project you want to import.
Use the keyboard arrows to select your project and hit <kbd>enter</kbd>.

Next, `encord-active` will fetch your data and labels before computing all the [metrics](/category/metrics) available in `encord-active`.

When the process is done, follow the printed instructions to open the app or see more details in the [visualize](../../cli#visualize) CLI command.

[encord-docs-ssh]: https://docs.encord.com/admins/settings/public-keys/#set-up-public-key-authentication
