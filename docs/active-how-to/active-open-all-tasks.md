---
title: "Open Tasks from Active to Annotate"
slug: "active-open-all-tasks"
hidden: false
metadata: 
  title: "Open All Tasks from Active to Annotate"
  description: "Learn how to open Annotate tasks using Collections in Encord Active Cloud."
category: "6480a3981ed49107a7c6be36"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/hosted_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

After creating a Collection, you can open annotation tasks of images/frames in the Collection, when the Collection is sent to Annotate. That means regardless of the image's/frame's current status in the Workflow (Annotate, Review, Complete) the task is opened/reopend in Annotate.

### Important Information

- There is no insight on the image's/frame's current status in the Workflow (Annotate, Review, Complete) in Annotate.

- Reopening a closed task maintains the priority specified in the Collection.

> ❗️ CRITICAL INFORMATION
> 
> This action cannot be undone. This is a bulk operation on all images/frames in the Collection, and once the action is submitted to Annotate, all images/frames in the Collection are opened/reopened in their Annotate Project. That means regardless of the image's/frame's current status in the Workflow (Annotate, Review, Complete) the task is opened/reopend in Annotate. 

## Open tasks from Active to Annotate

In addition to opening tasks, all priorities, comments on individual frames/images, and the comment added for the Collection, are sent to the Annotate Project.

<details>

<summary><b>To send a Collection to Annotate:</b></summary>

1. Log in to the Encord platform.
   The landing page for the Encord platform appears.
   
2. Click **Active** in the main menu.
   The landing page for Active appears.

3. Click the Project.
   The landing page for the Project appears with the _Explorer_ tab selected.

4. Click **Collections**.
   The _Collections_ page appears.

5. Select the checkbox for the Collection to send to Annotate.

6. Click **Send to Annotate**.
   The _Send to Annotate_ dialog appears.
   <img src="https://storage.googleapis.com/docs-media.encord.com/static/img/active/user-guide/active-send-to-annotate.png" width="400" />

7. Specify the following:

   - **Adjust priority of tasks:** Specifies the priority of all the images/frames in the Collection, when the images/frames are sent to Annotate.
   
   - **Include model predictions as pre-labels:** Includes model predictions as pre-labels in all frames/images in the Collection

   - **Reopen tasks:** Reopens all annotation tasks on images/frames in the Collection.

     > ❗️ CRITICAL INFORMATION
     > 
     > This action cannot be undone. 
   
   - **Delete all existing labels:** Deletes all exsiting labels on all frames/images in the Collection.

     > ❗️ CRITICAL INFORMATION
     > 
     > This action cannot be undone.
   
   - **Leave a comment:** Applies the comment on all frames/images in the Collection.

    > 🚧 WARNING
    > 
	> Comments applied on a Collection cannot be deleted in bulk. We recommend using comments, created on a Collection, in image sequence and video Datasets.

8. Click **Submit**.
   The _Subset created successfully_ dialog appears once creation completes.

9. Click the link in the dialog to go to the Project in Annotate.

10. Users in Annotate can then view opened/reopend tasks, priorities on tasks, and [can access comments](https://docs.encord.com/docs/annotate-label-editor#comments) made in the Collection. 

> ℹ️ Note
> After annotating the data, sync the Project data between Annotate and Active. To sync the Project data go to **Active > [select the project] > click More > Sync Project Data**.

</details>