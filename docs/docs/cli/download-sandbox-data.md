---
sidebar_position: 1
---

# Download Sandbox Dataset

To get started quickly with a sandbox dataset, you run the following command.

```shell
(ea-venv)$ encord-active download
```

The script will ask you to

1. `Where should we store the data?`: specify a directory in which the data should be stored.
2. (potentially) `Directory not existing, want to create it? [y/N]` type <kbd>y</kbd> then <kbd>enter</kbd>.
3. `[?] Choose a project:` choose a project with <kbd>↑</kbd> and <kbd>↓</kbd> and hit <kbd>enter</kbd>

:::tip

If you plan on using multiple datasets, it may be worth first creating an empty directory for all the datasets.

```
$ mkdir /path/to/data/root
# or windows
$ md /path/to/data/root
```

In step 1. above, specify, e.g., `/path/to/data/root/sandbox1`

:::

When the download process is complete, you visualise the results by following the printed instructions.
For more details, see the [Open Encord Active](/cli/open-encord-active) page.



