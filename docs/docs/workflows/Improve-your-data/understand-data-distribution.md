---
sidebar_position: 1
---

# Understand your Data Distribution

**Get insights into the distribution of your visual data with Encord Active**

 Encord Active enables you to visually explore your data distribution by pre-defined metrics, custom metrics, and label classes. Understanding your data distribution by different metrics helps you uncover areas where you might be missing data that could improve your models performance on different outliers or edge cases. 
 

 `Prerequisites:` Dataset

:::tip

If you have uploaded your model predictions you can combine this workflow with [Find Important Metrics](/workflows/improve-your-models/metric-importance) to better prioritise what metrics to look at. 

:::

### Setup
If you haven't installed Encord Active, run:

```shell
python3.9 -m venv ea-venv
source ea-venv/bin/activate
# within venv
pip install encord-active
```

In this workflow we will be using the COCO validation dataset:

```shell
# run download command
encord-active download
Loading prebuilt projects ...
[?] Choose a project: [open-source][validation]-coco-2017-dataset (1145.2 mb)
  >[open-source][validation]-coco-2017-dataset (1145.2 mb)
   [open-source][test]-limuc-ulcerative-colitis-classification (316.0 mb)
   [open-source]-covid-19-segmentations (55.6 mb)
   [open-source][validation]-bdd-dataset (229.8 mb)
```

After downloading the dataset we visualise it:
```shell
# open the UI
cd path/to/[open-source][validation]-coco-2017-dataset
encord-active visualise
```

## Steps
Navigate to the _Data Quality_ > _Explorer_ tab and select a quality metric in the top left menu to order your data by.

Select a metric to order your data by in the dropdown menu in the top of the page (e.g., Brightness or Aspect Ratio).

![data-quality-similar-images.png](../../images/data-quality-similar-images.png)

In the dashboard you can see the distribution of your data according to the chosen metric.

Use the slider to navigate the dataset ordered by the chosen metric.

![data-quality-similar-images-quality.png](../../images/data-quality-similar-images-quality.png)



