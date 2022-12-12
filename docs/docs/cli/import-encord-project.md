---
sidebar_position: 2
---

# Importing Encord Project

If you already have a project on the Encord platform, you can import that project with the following command:

```
encord-active import project
```

:::note
This will import a project to new folder in your current working directory. If you prefer to specify a different directory, use:

```shell
encord-active import project --targe /path/to/store/project
```

:::

:::tip

If you don't have an Encord project ready, you can find your next steps in the SDK section [Migrating Data to Encord](/sdk/migrating-data).
Otherwise, you can [download one of our sandbox datasets](/cli/download-sandbox-data).

:::

To be able to do this, you will need the path to your private `ssh-key` associated with Encord and a `project_hash`.
Don't worry! The script will guide you on the way if you don't know it already.
The script will ask you:

1. `Where is your private ssh key stored?`: type the path to your private ssh key
2. `Specify project hash`: paste / type the `project_hash`. If you don't know the id, you can type `?` and hit enter to get all your projects listed with their `project_hash`s before being prompted with the question again. Now you can copy paste the id.

Next, `encord-active` will fetch your data and labels before computing all the [metrics](/category/metrics) available in `encord-active`.

Downloading the data and computing the metrics may take a while.
Bare with us, it is worth the wait.

When the process is done, follow the printed instructions to open the app or see more details in the [Open Encord Active](/cli/open-encord-active) page.
