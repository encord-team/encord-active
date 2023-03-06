---
sidebar_position: 8
---

# Metric Management

Metrics constitute the foundation of Encord Active.
This section documents all the available commands on how to manage and run metrics in your projects.

:::info
Make sure your shell's current working directory is that of an Encord Active project, or your command points to one with the `--target` global option.
:::


## metric add

The `metric add` command adds required metrics to the project and attaches them.

You can add metrics from local python modules (`.py`).  
If you don't specify a metric, Encord Active will add all the metrics in the python module to the project.
```shell
encord-active metric add /path/to/module.py
```

You can also add specific metrics by using their title:
```shell
encord-active metric add /path/to/module.py "Metric Title" "Another Metric Title"
```

If you try to add a metric that is already present, it will be skipped and you will receive a notification.  
In case a metric title is not found in the python module, you will get an error.

:::caution
Some shells may treat square braces (`[` and `]`) as special characters. It is suggested to always quote arguments containing these characters to prevent unexpected shell expansion.
:::


## metric remove

The `metric remove` command removes metrics from the list of attached metrics.

```shell
encord-active metric remove "Metric Title"
```


## metric run

The `metric run` command runs available metrics over the project's data.

If you don't specify a metric, Encord Active will run all the metrics attached to the project.
```shell
encord-active metric run 
```

You can also run specific metrics by using their title:
```shell
encord-active metric run "Metric Title" "Another Metric Title"
```


## metric list

The `metric list` command lists all the metrics attached to the project.

Metrics are listed in a case-insensitive sorted order.
```shell
encord-active metric list
```

## metric show

The `metric show` command shows information about one or more available metrics in the project.

```shell
encord-active metric show "Metric Title"
```

