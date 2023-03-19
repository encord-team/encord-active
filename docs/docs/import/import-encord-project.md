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

If this is the first time you import a project, the command line interface (CLI) will ask you for your ssh key.
To associate an ssh key with Encord, you can follow [the documentation here][encord-docs-ssh].
You ssh key path will be stored for later reuse.

Next, the CLI will ask you what project to import based on all the projects you have access to at Encord.
You can type in a search word to find the project you want to import.
Use the keyboard arrows to select your project and hit <kbd>enter</kbd>.

This will create a new Encord Active project in a new directory in you current working directory.
Afterwards, you can run

```shell
encord-active visualize
```

This will let you choose your newly imported project and open the app.

:::info

For the full documentation of importing Encord projects, please see [here](../../cli/import-encord-project).

:::

[encord-docs-ssh]: https://docs.encord.com/admins/settings/public-keys/#set-up-public-key-authentication
