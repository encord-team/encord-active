---
sidebar_position: 2
---

# Finding Important Metrics

**Visualise the relationship between your model performance and metrics**

With this workflow, you will be able to identify the most important [metrics](category/metrics) for your model performance and prioritise further data exploration and actions.

`Prerequisites:` Dataset, Labels, Predictions

### Steps:

1. Navigate to the _Model Quality_ > _Metrics_ tab.
2. Select label classes to include in the top left drop down menu.
3. Determine IoU threshold using the slider in the top bar. By default, IoU threshold is set to 0.50.
4. Next, Encord Active automatically computes mAP, mAR, [Metric Importance](../../pages/model-quality/metrics), and
   [Metric Correlation](../../pages/model-quality/metrics).

   **Metric importance**: Measures the _strength_ of the dependency between a metric and model
   performance. A high value means that the model performance would be strongly affected by
   a change in the metric. For example, a high importance in 'Brightness' implies that a change
   in that quantity would strongly affect model performance. Values range from 0 (no dependency)
   to 1 (perfect dependency, one can completely predict model performance simply by looking
   at this metric).

   **Metric [correlation](https://en.wikipedia.org/wiki/Correlation)**: Measures the _linearity
   and direction_ of the dependency between a metric and model performance.
   Crucially, this metric tells us whether a positive change in a metric
   will lead to a positive change (positive correlation) or a negative change (negative correlation)
   in model performance. Values range from -1 to 1.

5. Metrics denoted with (P) are _Prediction-level metrics_ and metrics with (F) are _Frame-level metrics_.
6. Once an important metric is identified, navigate to _Performance By Metric_ in the _Model Quality_ tab.
7. Select the important metric you want to understand using the drop-down menu on the top bar.
8. By default, the performance chart is shown in aggregate for all classes, optionally you can choose to decompose performance by class or select individual classes to be shown in the top left drop down menu.
9. The plot shows the _True Positive Rate_ (TPR) and the _False Negative Rate_ (FNR) by metric to help you identify which metric characteristics you model have a hard time predicting.

### Example

![metric_importance](../../images/index_importance.png)
Metric importance plots indicate that _Object Area - Relative (P)_ is an important metric that has an important relationship
with the model performance.

In this case, go to **Performance By Metric** page and choose "_Object Area - Relative (P)_" in the **Select metric for your predictions** drop down menu.
Here, you can understand why _Object Area - Relative (P)_ has a relationship with the model performance, and you can act based upon insights you got from here.
Let's examine _Object Area - Relative (P)_ metric:

![metric_importance](../../images/object_area_relative_performance.png)

As indicated in the details, this metric refers to the object area as a percentage of total image area.
Blue dashed horizontal line (around 0.17 TPR and 0.77 FNR) is the average true positive rate and false negative rate of the selected classes, respectively.
So, what we get from the above graph is that objects, whose area is less than the 0.24%, have a very low performance.
In other words, the model predictions that are small are very often incorrect.
Similarly, labelled objects, for which the area is small, has a high false negative rate.

Based on this insight, you may improve your model with several actions, such as:

- Filtering model predictions to not include the smallest objects
- Increasing your model's input resolution
- Increasing the confidence threshold for small objects
