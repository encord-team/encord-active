---
title: "Import Project from Annotate"
slug: "active-import-from-annotate"
hidden: false
metadata: 
  title: "Import Project from Annotate"
  description: "Import Annotate Projects to Active to improve your workflows."
category: "6480a3981ed49107a7c6be36"
---

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Aligned Image with Page Break</title>\n    <style>\n        .aligned-image {\n            display: block;\n            margin: auto; /* This centers the image */\n        }\n\n        .page-break {\n            page-break-after: always; /* This adds a page break after the image */\n        }\n    </style>\n</head>\n<body>\n    <img src=\"https://storage.googleapis.com/docs-media.encord.com/static/img/active/hosted_02.png\" width=\"200 alt=\"Your Image\" class=\"aligned-image\">\n    <div class=\"page-break\"></div>\n</body>\n</html>"
}
[/block]

[block:html]
{
  "html": "<div style=\"position: relative; padding-bottom: 47.5%; height: 0;\"><iframe src=\"https://www.loom.com/embed/8f00b8478c49440e9ff4c9c7488b371a?sid=ff97c399-4324-43eb-a825-321f43835cf3\" frameborder=\"0\" webkitallowfullscreen mozallowfullscreen allowfullscreen style=\"position: absolute; top: 0; left: 0; width: 100%; height: 100%;\"></iframe></div>"
}
[/block]

Creating a Project in Annotate is a good place to start using Active. From Annotate you can configure the Dataset, Ontology, and workflow for your Project. Once that is done, move to Active to provide a streamlined dataset for your annotators.

> ❗️ CRITICAL INFORMATION
>  
>  Your organization's Encord Admin (the person whose authentication key is used when setting up your Active deployment on Encord) must be added as an **Admin** to your Annotate Projects, for Annotate Projects to be imported into Active. If the Encord Admin is not added as an **Admin** to your Annotate Project, the Annotate Project does not appear in the Project list when you click the **Import Annotate Project** button in Active.

> ❗️ CRITICAL INFORMATION
> 
> We strongly recommend only importing Annotate Projects that use Workflows. The current feature set in Active is optimized to work with Annotate Projects that use Workflows. While Annotate Projects that use Manual QA can be imported into Active, there are a number of features that Manual QA projects do not support.

<ActiveFileSizeBestPractice />

## Data Import Behavior

Active imports your data in stages. This allows you to start working with your data as quickly as possible without delay. Active imports data in the following stages:

| Stage   | Import                      | Description                                                                                                                                 |
| :------ | :-------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ |
| 1 | Data import                 | All images/video frames and all Metadata. This takes just a few minutes.                                                                    |
| 2 | Label import                | All labels/annotations (instance and frame level) on the images/video frames. Label import time depends on the number of labels for import. |
| 3 | Metric calculation          | Calculations to unlock filtering and sorting by all quality metrics, and Overview shortcuts.                                                                                   |
| 4 | Embedding calculation       | Calculations to unlock Embeddings and similiarity search.                                                                                                 |
| 5 | Advanced metric calculation | Calculations to unlock embedding reductions (view and filter) and the more complex metrics.                                                                                         |


### Stage 1: Data Import

Import images / video frames, and metadata from your Annotate Project into Active in just a few minutes!

Once Stage 1 completes you can:

- View all of the images/video frames in your Project on the _Explorer_ page

- Filter your images/video frames by metadata

- Create Collections based on your data

- Send to Annotate:

  - Send priorities and add and send comments

  - Create Datasets and Projects

You must WAIT for a later stage to:

- View labels/annotations

- Filter by anything except by metadata

- Use [Overview shortcuts](https://docs.encord.com/docs/active-issue-shortcuts-prediction-types)

- Sort your data using any criteria except Random

- Search your data using natural language and image search

- Use Embeddings

- Access the _Analytics_ page

- Access the _Model Evaluation_ page


### Stage 2: Label Import

During Stage 2, all image and video frame labels/annotations (instance and frame-level), are imported into your Project. More labels take longer to import.

Once Stage 2 completes you can:

- Perform all tasks from the previous stages

- View all labels/annotations on your data in the _Explorer_ page

- Filter all data by Class

- Access to the Label view on the _Explorer_ page

- Send to Annotate:
  
  - Reopen tasks
  
  - Perform bulk Classification

You must WAIT for a later stage to:

- Filter by anything except by metadata and class

- Use [Overview shortcuts](https://docs.encord.com/docs/active-issue-shortcuts-prediction-types)

- Sort your data using any criteria except Random

- Search your data using natural language and image search

- Use Embeddings

- Access the _Analytics_ page

- Access the _Model Evaluation_ page

### Stage 3: Metric Calculation

During Stage 3, Active performs metric calculations for flitering and Overview shortcuts.

Once Stage 3 completes you can:

- Perform all tasks from the previous stages

- Filter your data using any criteria

- Use [Overview shortcuts](https://docs.encord.com/docs/active-issue-shortcuts-prediction-types)

You must WAIT for a later stage to:

- Sort your data using any criteria except Random

- Search your data using natural language and image search

- Use Embeddings

- Access the _Analytics_ Page

- Access the _Model Evaluation_ page


### Stage 4: Embedding calculation

During Stage 4, Active performs calculations for embeddings and search capabilities.


Once Stage 4 completes you can:

- Perform all tasks from the previous stages

- Similarity search for images and frames

- Natural language and image searches

- Use Embeddings

- Sort your data using any criteria


### Stage 5: Advanced Metric Calculation

Stage 5 completes the import and unlocks all other remaining features for Active. For example, embedding reductions, metrics that depend on embeddings, and filtering on embedding reductions.


## Import an Annotate Project

<details>

<summary><b>To import an Annotate Project:</b></summary>

1. Log in to the Encord platform.
   The landing page for the Encord platform appears.

2. Create a Project ([Annotation Project](https://docs.encord.com/docs/annotate-annotation-projects) or [Training Project](https://docs.encord.com/docs/annotate-training-projects)) in Encord Annotate.

3. Click **Active** from the main menu.
   The landing page for Active appears.
   
4. Click the **Import Annotate Project** button.
   The _Select an Annotation Project_ dialog appears.
   ![Start import](https://storage.googleapis.com/docs-media.encord.com/static/img/active/user-guide/import-project_01.png)

5. Click the **Import** button for the Annotate project you want to import.
   The _Confirm Project Import_ dialog appears. The dialog provides an estimate on how long the import may take.
   ![Import info](https://storage.googleapis.com/docs-media.encord.com/static/img/active/user-guide/import-project_02.png)

6. Specify the **Sample rate** (in FPS) for the import of videos.

    > ℹ️ Note
	> Select **None** imports the entire video, without modification to the FPS of the video.

7. Click **Confirm**.
   ![Confirm import](https://storage.googleapis.com/docs-media.encord.com/static/img/active/user-guide/import-project_03.png)   

8. Close the _Select an Annotation Project_ dialog.
   The landing page for Active appears with progress of the import for the project. Stage 1 (Data import) of the Project import completes in a few minutes.

9. Click the Project to monitor the status of the import.
   The landing page for the Project appears with the _Explorer_ tab selected and displaying the stage of the import.
   ![Import stages](https://storage.googleapis.com/docs-media.encord.com/static/img/active/user-guide/import-project_04.png)
   
   ![Import stages progress](https://storage.googleapis.com/docs-media.encord.com/static/img/active/user-guide/import-project_05.png)

Once all five stages of the import complete, you are ready to [filter, sort, and search data/labels](https://docs.encord.com/docs/active-filtering), [create collections](https://docs.encord.com/docs/active-collections) and optimize your model performance.

</details>


## Next Steps

### Data Cleansing/Curation and Label Correction/Validation

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Clickable Div</title>\n    <style>\n        .clickable-div {\n            display: inline-block;\n            width: 200px;\n            height: 50px;\n            background-color: #ffffff;\n            border: solid;\n            text-align: center;\n            line-height: 50px;\n            color: #000000;\n            text-decoration: none;\n            margin: 10px;\n        }\n\n        .clickable-div:hover {\n            background-color: #ededff;\n        }\n    </style>\n</head>\n<body>\n    <a href=\"https://docs.encord.com/docs/active-create-collections\" class=\"clickable-div\">2. Create Collection</a> <a href=\"https://docs.encord.com/docs/active-send-collection-to-annotate\" class=\"clickable-div\">3. Send to Annotate</a> <a href=\"https://docs.encord.com/docs/active-sync-with-annotate\" class=\"clickable-div\">4. Sync with Annotate</a> <a href=\"https://docs.encord.com/docs/active-update-collections\" class=\"clickable-div\">5. Update Collection</a>\n</body>\n</html>"
}
[/block]

### Model and Prediction Validation

[block:html]
{
  "html": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>Clickable Div</title>\n    <style>\n        .clickable-div {\n            display: inline-block;\n            width: 200px;\n            height: 50px;\n            background-color: #ffffff;\n            border: solid;\n            text-align: center;\n            line-height: 50px;\n            color: #000000;\n            text-decoration: none;\n            margin: 10px;\n        }\n\n        .clickable-div:hover {\n            background-color: #ededff;\n        }\n    </style>\n</head>\n<body>\n    <a href=\"https://docs.encord.com/docs/active-import-model-predictions-cloud\" class=\"clickable-div\">2. Import Predictions</a> <a href=\"https://docs.encord.com/docs/active-model-predictions-eval\" class=\"clickable-div\">3. Review Prediction Metrics</a> <a href=\"https://docs.encord.com/docs/active-create-collections\" class=\"clickable-div\">4. Create Collection</a> <a href=\"https://docs.encord.com/docs/active-send-collection-to-annotate\" class=\"clickable-div\">5. Send to Annotate</a> <a href=\"https://docs.encord.com/docs/active-sync-with-annotate\" class=\"clickable-div\">6. Sync with Annotate</a> <a href=\"https://docs.encord.com/docs/active-update-collections\" class=\"clickable-div\">7. Update Collection</a>\n</body>\n</html>"
}
[/block]