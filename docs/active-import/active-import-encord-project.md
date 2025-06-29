---
title: "Import Encord annotation project"
slug: "active-import-encord-project"
hidden: false
metadata: 
  title: "Import Encord annotation project"
  description: "Seamlessly import Encord Annotate projects to Encord Active. Quick transition, all-inclusive data transfer."
  image: 
    0: "https://files.readme.io/1fe9089-image_16.png"
createdAt: "2023-07-11T16:27:41.855Z"
updatedAt: "2023-08-09T13:03:43.944Z"
category: "65a71bbfea7a3f005192d1a7"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/local_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

**Pull projects from Encord Annotate**

If you already have a project in Encord Annotate, you can start with its respective Encord Active project right away.

> ℹ️ Note
> If you are new to the Encord platform, [signing up](https://app.encord.com/register) for an Encord account is quick and easy.


To interactively select a project from the list of available projects in Encord Annotate use the following command:

```shell
encord-active import project
```

To narrow down the search for the project you wish to import, enter some text that matches the project title. Use the keyboard arrows to navigate and select the desired project, then press <kbd>Enter</kbd> to confirm your choice.

Alternatively, if you prefer to override the selection process, you can use the `--project-hash` option when executing the command.

You will get a directory containing all the data, labels, and [metrics](https://docs.encord.com/docs/active-quality-metrics) of the project. You have the option to choose whether to store the data in the local file system and can opt-in or opt-out accordingly.

When the process is done, follow the printed instructions to launch the app with the [start][ea-cli-start] CLI command.

If you are importing an Encord Annotate project for the first time, the Command Line Interface (CLI) will prompt you to provide the local path of a private SSH key associated with Encord. To associate an SSH key with Encord, please refer to the documentation available [here](https://docs.encord.com/docs/annotate-public-keys#set-up-public-key-authentication). The provided SSH key path will be stored for future use.

> ℹ️ Note
> The previous command imports a project into a new folder within the current working directory. However, if a different directory needs to be specified, the `--target` option can be included as follows:
>
> ```shell
> encord-active import project --target /path/to/store/project
> ```
>
> This will import the project in a subdirectory of `/path/to/store/project`.


[ea-cli-start]: https://docs.encord.com/docs/active-cli#start