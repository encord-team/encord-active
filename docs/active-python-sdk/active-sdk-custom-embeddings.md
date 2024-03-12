---
title: "Custom embeddings"
slug: "active-sdk-custom-embeddings"
hidden: false
metadata: 
createdAt: "2023-07-21T09:09:02.242Z"
updatedAt: "2023-08-09T16:15:32.198Z"
category: "65a71bbfea7a3f005192d1a7"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/local_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

The easiest way to add custom embeddings to your project is to use [this notebook][custom-embeddings-notebook] as a starting point.

[custom-embeddings-notebook]: https://colab.research.google.com/github/encord-team/encord-notebooks/blob/main/colab-notebooks/Encord_Active_Add_Custom_Embeddings.ipynb