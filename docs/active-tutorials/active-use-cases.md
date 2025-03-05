---
title: "Active Use Cases"
slug: "active-use-cases"
hidden: false
metadata: 
  title: "Active Use Cases"
  description: "Use cases when using Active."
category: "6480a3981ed49107a7c6be36"
---

Active provides value to you in a number of use cases. This page lists a few. If you have other use cases you would like to explore with Active, contact us.

## Data Cleansing/Curation

[block:html]
{
  "html": "<div style=\"position: relative; padding-bottom: 47.5%; height: 0;\"><iframe src=\"https://www.loom.com/embed/b09f407c864840999514ff7590a1b766?sid=556f8420-a0de-4108-9648-d5f7dca8413d\" frameborder=\"0\" webkitallowfullscreen mozallowfullscreen allowfullscreen style=\"position: absolute; top: 0; left: 0; width: 100%; height: 100%;\"></iframe></div>"
}
[/block]

Alex, a DataOps manager at **self-dr-AI-ving**, faces challenges in managing and curating data for self-driving cars. Alex's team struggles with scattered data, overwhelming amounts of data, unclear workflows, and an inefficient data curation processes. Alex is currently a big user of Encord Annotate, but would like to provide better datasets for annotation.

1. **Initial setup**: Alex gathers a large number of images and gets them imported into Active. Alex then logs into Encord and navigates to Active (freemium). 

2. **First collection**: Alex opens the Project and after searching, sorting, and filtering the data, selects the images and clicks **Add to a Collection** and then clicks **New Collection**. Alex names the Collection **RoadSigns** as the Collection is designed for annotating road signs for the team.

3. **Data curation**: Alex then further bulk-finds traffic sign images using the embeddings and similarity search. Alex then clicks **Add to a Collection** and then clicks **Existing Collection** and adds them images to the **RoadSigns** Collection in a matter of clicks.

4. **Labeling workflow**: Thinking about different use-cases (for example, "Labeling" and "Data quality") Alex assigns good quality road signs for labeling, and bad quality road signs for "Data quality" and future "Deletion". In the future Alex might use "Active learning" to prioritize the data for labeling.
	
5. **Sent to Annotate:** Alex goes to the _Collections_ page, selects the _Roadsigns_ Collection and clicks **Create Dataset**. Active creates the dataset and [a new project in Annotate](https://docs.encord.com/docs/annotate-annotation-projects#creating-annotation-projects). Alex then configures the workflow, annotators, and reviewers for the Project in Annotate.

6. **Review and insights**: At the end of the week, Alex reviews the _RoadSigns_ Project in Annotate. The dataset has been annotated. Alex goes to Active, clicks the **More** button on the Project then clicks **Sync Project Data**. Alex then clicks the Project, clicks _Analytics_ and then _Model Predictions_ where Alex gains insights into:

    - Number of labels per image
    - Quality of annotations
    - Distribution of annotations across metric
	
The process is seamless and fast, and Alex can focus on more strategic tasks while her team enjoys a much-improved, streamlined data curation workflow.

## Label Correction/Validation

Chris, a Machine Learning Engineer at a micro-mobility startup, has been working with Encord Annotate. His team is dealing with a large set of scooter images that need accurate labeling for obstacle detection. After an initial round of annotations, Chris notices that some labels are incorrect or ambiguous. This has a significant impact on the performance of the ML model.

1. **Access Collections:** Chris logs into Encord Active. Chris opens the Scooters project that was imported from Annotate. Chris goes to the _Collections_  page for the project. Chris browses the existing Collections to see if he should create a new Collection or add to an existing Collection.

2. **Data exploration:** Chris searches, sorts, and filters the previously annotated scooter images and identifies those that need re-labelling. 

3. **Create re-labeling Collection:** Chris selects the images and clicks **Add to a Collection**. Chris then clicks **New Collection** and names the Collection **Re-label - Scooters**.

4. **Initiate re-labelling:** With the Collection ready, Chris returns to the _Collections_ page, selects the **Re-label - Scooters** Collections and clicks **Create Dataset**. The Collection is sent to Annotate.

5. **Assigning in Annotate:** In Annotate, Chris assigns re-labelling tasks to specific annotators. Annotators then complete their relabelling tasks.

6. **Quality check in Active:** After the re-labeling tasks are completed, Chris clicks **Sync Project Data**. The updated labels then sync back to the Project. Chris reviews the changes, confirms the label quality, and plans for model re-training.

The "Collections" feature has simplified the task of identifying and re-labeling inaccurate or ambiguous data, streamlining the entire data annotation and quality control process for Chris and his team.

## Model/Prediction Evaluation

_Coming soon..._