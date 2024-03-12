---
title: "Evaluating classification models"
slug: "active-evaluate-classification-models"
hidden: true
metadata: 
  title: "Evaluating classification models"
  description: "Enhance model assessment: Encord Active for classification metrics. Accuracy, Precision, Recall, F1 scores & more. Optimize with insights."
  image: 
    0: "https://files.readme.io/ba490b4-image_16.png"
createdAt: "2023-07-11T16:27:42.137Z"
updatedAt: "2023-08-09T12:38:12.905Z"
category: "6480a3981ed49107a7c6be36"
---

Encord Active provides the capability to examine <<glossary:classification>> performance metrics including Accuracy, Precision, Recall, and F1 scores, along with a confusion matrix. Furthermore, these performance metrics can be assessed based on various class combinations.

To follow this workflow, importing model predictions into an Encord Active project is a prerequisite. You can refer to the instructions on [importing model predictions](https://docs.encord.com/docs/active-import-model-predictions) in the documentation.

## Steps

1. Navigate to the _Model Quality_ > _Metrics_ tab on the left sidebar.
2. Under the **Classifications** tab, you will see the main performance metrics (accuracy, mean precision, mean recall, and mean F1 scores), metric importance graphs, confusion matrix, and class-based precision and recall plot.
3. You can filter by classes in the upper bar to see plots for your classes of interest.
4. Via the confusion matrix, you can detect which classes are confused with each other (uni-directional or bi-directional).
5. On the **Precision-Recall** plot, you can observe which classes the model has difficulty in and which classes it does well.
6. According to insights you get here, you can, e.g., prioritize from which classes you need to collect more data.

## Example

The following is the model performance result for a Pokemon dataset (classes: _Charmander, Mewtwo, Pikachu, Squirtle_).

![metric](https://storage.cloud.google.com/docs-media.encord.com/static/img/images/workflows/evaluate-classification-model/img_1.png)

## Finding Important Metrics

The three important metrics for this dataset are _Image-level Annotation Quality, Red Values_, and _Uniqueness_.
When we look at their correlation, we see that as the Image-Level Annotation Quality increases, model performance increases, too. On the other hand, Red Values and Uniqueness have a negative correlation with the model performance.

When we look at the confusion matrix, we find that most of the predictions are correct; Meanwhile, we can easily observe that a significant part of the _Charmander_ images was predicted as _Pikachu_, resulting in low recall for the _Charmander_ and low precision for the _pikachu_ classes. So there might be value in investigating these wrongly labeled Charmander samples.

## Performance by Metric

Now, choose _Performance By Metric_ on the left sidebar. On this page, you can observe the **True-Positive Rate** as a function of the chosen metric. You can detect which regions model performs poorly or well, so that you can prioritize your next data collection and labeling work accordingly. Classes can be filtered via global top bar for class-specific visualization. From the image below, it can be seen that performance decreases when the image's redness property increases. So, we can find similar images, like the ones where the model is failing, and annotate more of them to boost the performance in this region.

![performance_by_metric](https://storage.cloud.google.com/docs-media.encord.com/static/img/images/workflows/evaluate-classification-model/img_2.png)

## Exploring the Individual Samples

Using the explorer page, you can visualize the ranked images for specific outcomes (True Positives, False Positives).

![explorer](https://storage.cloud.google.com/docs-media.encord.com/static/img/images/workflows/evaluate-classification-model/img_3.png)

This page is very similar to the other explorer pages under _Data Quality_ and _Label Quality_ tabs; however, since you have the prediction results now, the images can be filtered according to their outcome type. When the **True Positive** outcome is selected, only the images that are predicted correctly will be shown; likewise, when the **False Positive** outcome is selected, only the wrongly predicted images will be shown.

By inspecting False-Positive images, you can detect:

- Where your model is failing.
- Possible duplicate errors.
