---
title: "Export Collections to CSV"
slug: "active-export-collection-to-csv"
hidden: false
metadata: 
  title: "Export Collections to CSV"
  description: "Learn how to export Collections in Encord Active Cloud CSV."
category: "6480a3981ed49107a7c6be36"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/hosted_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

Any Collection can be exported to CSV. The CSV file contains the following:

- DataTitle: Specifies the assigned name of the image, image group, image sequence, or video when the data uploaded to Encord.

- DataHash: Specifies the unique data ID for each image, image group, image sequence, or video in the Collection.

- ImageTitle: Specifies the file name for the image or video in the Collection.

- ImageHash: Specifies the unique ID for each image or video contained in the Collection.

- VideoFrame: Specifies the frame number in the Collection. This value is `0` if the Collection does not contain video.

- EditorURL: Specifies Annotate Label Editor unique ID for each image or video frame in the Collection.

**Example:**

```
DataTitle	DataHash	ImageTitle	ImageHash	VideoFrame	EditorURL
image-sequence-8c45bcea	2171a8de-d8ab-4b5d-aec0-17f49239f98b	blueberries-17.JPG	05e23742-daf5-44cc-8602-a8925d381d89	9	https://dev.encord.com/label_editor/2171a8de-d8ab-4b5d-aec0-17f49239f98b&dc1e94d0-879c-4374-b523-895a40ef87a9/9
image-sequence-8c45bcea	2171a8de-d8ab-4b5d-aec0-17f49239f98b	blueberries-01.JPG	1932938e-ec83-443c-8ecd-a9f229bbfa64	0	https://dev.encord.com/label_editor/2171a8de-d8ab-4b5d-aec0-17f49239f98b&dc1e94d0-879c-4374-b523-895a40ef87a9/0
image-group-6274340e	24104c92-00fe-416c-9620-59df4b0295dd	cherries_12.jpg	32ad8717-81b3-40f8-98d3-d279a3955b80	3	https://dev.encord.com/label_editor/24104c92-00fe-416c-9620-59df4b0295dd&dc1e94d0-879c-4374-b523-895a40ef87a9/3
image-group-6274340e	24104c92-00fe-416c-9620-59df4b0295dd	cherries_14.jpg	394bca42-8f9a-4fa5-b501-f0c4731d9fbe	1	https://dev.encord.com/label_editor/24104c92-00fe-416c-9620-59df4b0295dd&dc1e94d0-879c-4374-b523-895a40ef87a9/1
Blueberries.mp4	6eabace8-2d84-4a77-b266-f86f4ef74fee	Blueberries.mp4	6eabace8-2d84-4a77-b266-f86f4ef74fee	1	https://dev.encord.com/label_editor/6eabace8-2d84-4a77-b266-f86f4ef74fee&dc1e94d0-879c-4374-b523-895a40ef87a9/1
Blueberries.mp4	6eabace8-2d84-4a77-b266-f86f4ef74fee	Blueberries.mp4	6eabace8-2d84-4a77-b266-f86f4ef74fee	138	https://dev.encord.com/label_editor/6eabace8-2d84-4a77-b266-f86f4ef74fee&dc1e94d0-879c-4374-b523-895a40ef87a9/138
Blueberries.mp4	6eabace8-2d84-4a77-b266-f86f4ef74fee	Blueberries.mp4	6eabace8-2d84-4a77-b266-f86f4ef74fee	193	https://dev.encord.com/label_editor/6eabace8-2d84-4a77-b266-f86f4ef74fee&dc1e94d0-879c-4374-b523-895a40ef87a9/193
image-sequence-8c45bcea	2171a8de-d8ab-4b5d-aec0-17f49239f98b	blueberries-25.JPG	cea53be3-f0f2-41de-8b54-79c471efb2fc	1	https://dev.encord.com/label_editor/2171a8de-d8ab-4b5d-aec0-17f49239f98b&dc1e94d0-879c-4374-b523-895a40ef87a9/1
image-group-6274340e	24104c92-00fe-416c-9620-59df4b0295dd	cherries_09.jpg	f225843f-c8de-4368-80c8-e3a13b24ea23	6	https://dev.encord.com/label_editor/24104c92-00fe-416c-9620-59df4b0295dd&dc1e94d0-879c-4374-b523-895a40ef87a9/6
```

<details>

<summary><b>To export a Collection to CSV:</b></summary>

1. Log in to the Encord platform.
   The landing page for the Encord platform appears.
   
2. Click **Active** in the main menu.
   The landing page for Active appears.

3. Click the Project.
   The landing page for the Project appears with the _Explorer_ tab selected.

4. Click **Collections**.
   The _Collections_ page appears.

5. Select the checkbox for the Collection for export.

6. Click the more icon.
   A small menu appears.

7. Click **Generate CSV** from the menu.
   The _Download Generated CSV_ dialog appears when Active finishes generating a CSV file for the Collection.

8. Click **Download CSV**.
   The CSV file downloads to your local computer.

</details>