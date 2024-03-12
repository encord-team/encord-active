---
title: "Create a Collection"
slug: "active-create-collections"
hidden: false
metadata: 
  title: "Create a Collection"
  description: "Learn how to create Collections in Encord Active Cloud to enhance data organization, search, and collaboration."
category: "6480a3981ed49107a7c6be36"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/hosted_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

[block:html]
{
  "html": "<div style=\"position: relative; padding-bottom: 47.5%; height: 0;\"><iframe src=\"https://www.loom.com/embed/e1bb92fdc39d4b4588931f4d885ae7b3?sid=8c5dfcd0-bd90-4c61-99a0-6ca05df78eac\" frameborder=\"0\" webkitallowfullscreen mozallowfullscreen allowfullscreen style=\"position: absolute; top: 0; left: 0; width: 100%; height: 100%;\"></iframe></div>"
}
[/block]

Collections are created by tagging/labeling images and then building groups (Collections) based on the tagged images. Tagging is a versatile feature used in almost all Encord Active workflows, whether you are relabeling, augmenting, exporting, or deleting data.

In Encord Active, creating Collections provides several advantages:

- Organization: Allows you to organize your data effectively within the platform. By assigning Collection tags to your data points, you can group and categorize data based on common characteristics, making it easier to manage and navigate large subsets of the dataset.

- Enhanced search and filtering: Collections in Encord Active enables powerful search and filtering capabilities. You can search for specific data points or filter data based on tags, narrowing down your focus to the relevant information you need.

- Customizable metadata: Collection tags serve as customizable metadata that can provide additional context and information about your data. You can define and assign tags that align with your specific project requirements, providing meaningful insights and annotations for efficient data analysis.

- Collaboration and knowledge sharing: Tagging promotes collaboration and knowledge sharing among team members in Encord Active. With consistent tagging conventions, team members can easily understand and access tagged data, facilitating efficient collaboration and ensuring everyone is on the same page.

These are just a few of the advantages of tagging in Encord Active, and there may be more benefits specific to your project and workflow.

To give you a better idea about how Active and Annotate work together, here are a couple of use cases.

<details>

<summary><b>To create a Collection from an Annotate Project:</b></summary>

1. Log in to the Encord platform.
   The landing page for the Encord platform appears.

2. Create a Project ([Annotation Project](https://docs.encord.com/docs/annotate-annotation-projects) or [Training Project](https://docs.encord.com/docs/annotate-training-projects)) in Encord Annotate.

3. Click **Active** from the main menu.
   The landing page for Active appears.
   
4. Click the **Import Annotate Project** button.
   The _Select an Annotation Project_ dialog appears.

5. Click the **Import** button for the Annotate project you want to import.
   The _Confirm Project Import_ dialog appears. The dialog provides an estimate on how long the import may take.

6. Specify the **Sample rate** (in FPS) for the import of videos.

    > ℹ️ Note
	> Select **None** imports the entire video, without modification to the FPS of the video.

7. Click **Confirm**.

8. Close the _Select an Annotation Project_ dialog.
   The landing page for Active appears with progress of the import for the project.

9. Click the Project.
   The landing page for the Project appears with the _Explorer_ tab selected.

10. Search, sort, and filter your data/labels/predictions until you have the subset of the data you need.

11. Select one or more of the images in the Explorer workspace.
    A ribbon appears at the top of the Explorer workspace.

12. Click **Select all** to select all the images in the subset.

13. Click **Add to a Collection**.

14. Click **New Collection**.

15. Specify a meaningful title and description for the Collection.

    > ℹ️ Note
    > The title specified here is applied as a tag/label to every selected image.

16. Click **Collections** to verify the Collection appears in the Collections list.

</details>

<details>

<summary><b>To create a Collection from data uploaded to Active:</b></summary>

1. Contact Encord to get started with Encord Active.

2. Log in to the Encord platform.
   The landing page for the Encord platform appears.

3. Click **Active** in the main menu.
   The landing page for Active appears.

4. Click the Project.
   The landing page for the Project appears with the _Explorer_ tab selected.

5. Search, sort, and filter your data/labels/predictions until you have the subset of the data you need.

6. Select one or more of the images.
   A ribbon appears at the top of the Explorer workspace.

7. Click **Select all** to select all the images in the subset.

8. Click **Add to a Collection**.

9. Click **New Collection**.

10. Specify a meaningful title and description for the Collection.

    > ℹ️ Note
    > The title specified here is applied as a tag/label to every selected image.

11. Click **Collections** to verify the Collection appears in the Collections list.

</details>

## Next Steps

### Data Cleansing/Curation and Label Correction/Validation

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Clickable Div</title>\n    <style>\n        .clickable-div {\n            display: inline-block;\n            width: 200px;\n            height: 50px;\n            background-color: #ffffff;\n            border: solid;\n            text-align: center;\n            line-height: 50px;\n            color: #000000;\n            text-decoration: none;\n            margin: 10px;\n        }\n\n        .clickable-div:hover {\n            background-color: #ededff;\n        }\n    </style>\n</head>\n<body>\n    <a href=\"https://docs.encord.com/docs/active-import-from-annotate\" class=\"clickable-div\">1. Import from Annotate</a> <a href=\"https://docs.encord.com/docs/active-send-collection-to-annotate\" class=\"clickable-div\">3. Send to Annotate</a> <a href=\"https://docs.encord.com/docs/active-sync-with-annotate\" class=\"clickable-div\">4. Sync with Annotate</a> <a href=\"https://docs.encord.com/docs/active-update-collections\" class=\"clickable-div\">5. Update Collection</a>\n</body>\n</html>"
}
[/block]

### Model and Prediction Validation

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Clickable Div</title>\n    <style>\n        .clickable-div {\n            display: inline-block;\n            width: 200px;\n            height: 50px;\n            background-color: #ffffff;\n            border: solid;\n            text-align: center;\n            line-height: 50px;\n            color: #000000;\n            text-decoration: none;\n            margin: 10px;\n        }\n\n        .clickable-div:hover {\n            background-color: #ededff;\n        }\n    </style>\n</head>\n<body>\n    <a href=\"https://docs.encord.com/docs/active-import-from-annotate\" class=\"clickable-div\">1. Import from Annotate</a> <a href=\"https://docs.encord.com/docs/active-import-model-predictions-cloud\" class=\"clickable-div\">2. Import Predictions</a> <a href=\"https://docs.encord.com/docs/active-model-predictions-eval\" class=\"clickable-div\">3. Review Prediction Metrics</a> <a href=\"https://docs.encord.com/docs/active-send-collection-to-annotate\" class=\"clickable-div\">5. Send to Annotate</a> <a href=\"https://docs.encord.com/docs/active-sync-with-annotate\" class=\"clickable-div\">6. Sync with Annotate</a> <a href=\"https://docs.encord.com/docs/active-update-collections\" class=\"clickable-div\">7. Update Collection</a>\n</body>\n</html>"
}
[/block]