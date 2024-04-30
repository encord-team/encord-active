---
title: "Compare Model Performance"
slug: "active-compare-model-performance"
hidden: false
metadata: 
  title: "Compare Model Prediction Performance"
  description: "Learn how to compare the predictive performance of your model."
category: "6480a3981ed49107a7c6be36"
---

You have trained your model and now you are ready to see how it performs.  It is time to perform a cycle of the Active model optimization workflow.

![Encord Active workflow](https://storage.googleapis.com/docs-media.encord.com/static/img/active/active-workflow-model-optimization.png)

Now you want to compare your model's performance before using Encord (or maybe after running a number of data curation and label validtion cycles). Active supports doing direct model prediction performnce comparison from within your Active Project.

<details>

<summary><b>To compare your model's performance:</b></summary>

This process assumes you have already imported your model's predictions in to Active at least twice.

1. Log in to Encord.
   The Encord Homepage appears.

2. Create a **Workflow** Project in Annotate.

3. Add your Active Admin as an `Admin` on your **Project and all Datasets used in the Project**.

4. Click **Active**.
   The Active landing page appears.

5. Import your Annotate Project.

6. Click an Active Project.
   The Project opens on the _Explorer_.

7. Click **Model Evaluation**.
   The _Model Evaluation_ page appears with _Summary_ displaying.

8. Select an entry from the dropdown under **Prediction Set** under _Overview_.

9. Select an entry from the dropdown under **Compare against** under _Overview_.

10. Click through the various entries on the left side of the Model Evaluation page to view the comparison.

11. Add more data and start the data curation, label validation, and model optimization cycles until the model reaches a performance level that you require.

</details>

<details>

<summary><b>To compare your model's performance from scratch:</b></summary>

This process assumes you are just getting started with Encord. You have not trained your model yet. You are using Encord to prepare your data for annotation, annotating your data, labeling your data, validating your labels, fixing any label issues, then training your model.

1. Log in to Encord.
   The Encord Homepage appears.

2. Create a **Workflow** Project in Annotate.

3. Add your Active Admin as an `Admin` on your **Project and all Datasets used in the Project**.

4. Click **Active**.
   The Active landing page appears.

5. Import your Annotate Project.

6. Click an Active Project.
   The Project opens on the _Explorer_.

7. Click **Model Evaluation**.
   The _Model Evaluation_ page appears.

8. [Import a Prediction Set](https://docs.encord.com/docs/active-import-model-predictions-cloud).

9.  Perform data curation on your Project in Active.

10. Send the Project to Annotate.

11. Label and review your data in Annotate.

12. Sync the Active Project with the updated Annotate Project.

13. Perform label validation on your updated and sync'd Project.

14. Send the Project to Annotate.

15. Label and review your data in Annoate.

16. Retrain your model using the curated and validated data/labels.

17. Click the Active Project.
    The Project opens on the _Explorer_.

18. Click **Model Evaluation**.
   The _Model Evaluation_ page appears.

19. [Import the updated Prediction Set](https://docs.encord.com/docs/active-import-model-predictions-cloud).

20. Select an entry from the dropdown under **Prediction Set** under _Overview_.

21. Select an entry from the dropdown under **Compare against** under _Overview_.

22. Click through the various entries on the left side of the Model Evaluation page to view the comparison.

23. Add more data and start the data curation, label validation, and model optimization cycles until the model reaches a performance level that you require.

</details>
