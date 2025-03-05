---
title: "Import"
slug: "active-oss-import"
hidden: false
metadata: 
  title: "Import"
  description: "Import data into Encord Active: Easily bring in images, videos (MP4), and soon DICOM (DCM) formats. Streamline data integration."
category: "65a71bbfea7a3f005192d1a7"
---

To use Encord Active, you'll need data. This page shows you ways in which you can import your data into Encord Active.

Encord Active supports the following formats for images (jpg, png), videos (MP4) (DICOM (DCM) support is coming soon). Select the format that best fits your current data storage location.


[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/local_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

The import workflows currently support the following data and label integration:

<table>
  <thead>
    <tr>
      <td align="center">
        <b>Import Type</b>
      </td>
      <td colspan="2" align="center">
        <b>Data Type</b>
      </td>
      <td colspan="3" align="center">
        <b>Label Type</b>
      </td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"></td>
      <td align="center">
        <b>Images</b>
      </td>
      <td align="center">
        <b>Videos</b>
      </td>
      <td align="center">
        <b>Classification</b>
      </td>
      <td align="center">
        <b>Bounding Boxes</b>
      </td>
      <td align="center">
        <b>Polygons</b>
      </td>
	  <td align="center">
        <b>Polyline</b>
      </td>
      <td align="center">
        <b>Bitmask</b>
      </td>
      <td align="center">
        <b>Key-point</b>
      </td>
    </tr>
    <tr>
      <td>
        Quick import data & labels
      </td>
      <td align="center">✅</td>
      <td align="center">-</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>	  
    </tr>
    <tr>
      <td>
        Import model predictions
      </td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>	  
    </tr>
    <tr>
      <td>
        Encord project
      </td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>	  
    </tr>
    <tr>
      <td>
        COCO project
      </td>
      <td align="center">✅</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>
      <td align="center">✅</td>	  
    </tr>
  </tbody>
</table>

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Clickable Div</title>\n    <style>\n        .clickable-div {\n            display: inline-block;\n            width: 200px;\n            height: 50px;\n            background-color: #ffffff;\n            border: solid;\n            text-align: center;\n            line-height: 50px;\n            color: #000000;\n            text-decoration: none;\n            margin: 10px;\n        }\n\n        .clickable-div:hover {\n            background-color: #ededff;\n        }\n    </style>\n</head>\n<body>\n    <a href=\"https://docs.encord.com/docs/active-quick-import\" class=\"clickable-div\">Quick import data & labels</a> <a href=\"https://docs.encord.com/docs/active-import-model-predictions\" class=\"clickable-div\">Import model predictions</a> <a href=\"https://docs.encord.com/docs/active-import-encord-project\" class=\"clickable-div\">Encord project</a> <a href=\"https://docs.encord.com/docs/active-import-coco-project\" class=\"clickable-div\">COCO project</a>\n</body>\n</html>"
}
[/block]