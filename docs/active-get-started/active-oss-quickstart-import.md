---
title: "Import to Active OS"
slug: "active-oss-quickstart-import"
hidden: false
metadata: 
  title: "Import data and projects to Active OS"
  description: "A quick overview of importing data and projects to Active OS."
category: "65a71bbfea7a3f005192d1a7"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/local_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

## Import Your Own Data

To import your own data save your data in a directory and run the command:

```shell
encord-active init /path/to/data/directory
```

A project will be created using the data (without labels) in the current working directory (unless used with `--target`).

To launch the project in the Encord Active app, run the following command:

```shell
cd /path/to/project
encord-active start
```

You can find more details on the `init` command in the [CLI section](https://docs.encord.com/docs/#init).

## Import an Encord Project

If you are an Encord user, you can directly [import](https://docs.encord.com/docs/active-cli#project) your own projects into Encord Active easily.

```shell
encord-active import project
```

This will import your encord project to a new directory in your current working directory. If you don't have an Encord project ready, you can either

1. [Initialise a project from a local data directory](https://docs.encord.com/docs/active-cli#init)
2. [Import a project from COCO](https://docs.encord.com/docs/active-import-coco-project)
3. [Download one of our sandbox datasets](https://docs.encord.com/docs/active-cli#download)

> ℹ️ Note
> If you are new to the Encord platform, you can easily create an Encord account by [signing up](https://app.encord.com/register).


To import an Encord project, you will need the path to your private SSH key associated with the Encord user. See our documentation [here](https://docs.encord.com/docs/annotate-public-keys).

The command will ask you:

1. `Where is your private ssh key stored?`: type the path to your private ssh key
2. `What project would you like to import?`: here, you can (fuzzy) search for the project title that you would like to import. Hit <kbd>Enter</kbd> when your desired project is highlighted.

Next, `encord-active` will fetch your data and labels before computing all the [metrics](https://docs.encord.com/docs/active-quality-metrics) available in `encord-active`. Downloading the data and computing the metrics may take a while. Bare with us - it'll be worth the wait.

When the process is done, follow the printed instructions to launch the app with the [start][ea-cli-start] CLI command.

## Import Examples

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Clickable Div</title>\n    <style>\n        .clickable-div {\n            display: inline-block;\n            width: 200px;\n            height: 50px;\n            background-color: #ffffff;\n            border: solid;\n            text-align: center;\n            line-height: 50px;\n            color: #000000;\n            text-decoration: none;\n            margin: 10px;\n        }\n\n        .clickable-div:hover {\n            background-color: #ededff;\n        }\n    </style>\n</head>\n<body>\n    <a href=\"https://docs.encord.com/docs/active-quick-import\" class=\"clickable-div\">Quick import data & labels</a> <a href=\"https://docs.encord.com/docs/active-import-model-predictions\" class=\"clickable-div\">Import model predictions</a> <a href=\"https://docs.encord.com/docs/active-import-encord-project\" class=\"clickable-div\">Encord project</a> <a href=\"https://docs.encord.com/docs/active-import-coco-project\" class=\"clickable-div\">COCO project</a>\n</body>\n</html>"
}
[/block]

[ea-cli-start]: https://docs.encord.com/docs/active-cli#start
