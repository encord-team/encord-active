---
sidebar_position: 4
---

# Model Quality Metrics

Model quality metrics help you evaluate your data and labels based on a trained model and imported model predictions.


## Acquisition functions {#acquisition-functions}

Acquisition functions are a special type of model quality metric, primarily used in active learning to score data samples according to how informative they are for the model and enable smart labeling of unannotated data.

| Title                                                                                                | Metric Type | Data Type |
|------------------------------------------------------------------------------------------------------|-------------|----------|
| [Entropy](#entropy) - <small>Rank images by their entropy.</small>                                   | `image`     |          |
| [LeastConfidence](#least-confidence) - <small>Rank images by their least confidence score.</small>   | `image`     |          |
| [Margin](#margin) - <small>Rank images by their margin score.</small>                                | `image`     |          |
| [Variance](#variance) - <small>Rank images by their variance.</small>                                | `image`     |          |
| [AverageFrameScore](#average-frame-score) - <small>Rank images by their average object score</small> | `image`     | `object`  |

### Entropy {#entropy}

Rank images by their entropy.

In information theory, the entropy of a random variable is the average level of “information”, “surprise”, or “uncertainty” inherent to the variable's possible outcomes.
The higher the entropy, the more “uncertain” the variable outcome.

The mathematical formula of entropy is:

$H(p) = -\sum_{i=1}^{n} p_i \log_{2}{p_i}$

It can be employed to define a heuristic that measures a model’s uncertainty about the classes in an image using the average of the entropies of the model-predicted class probabilities in the image.
Like before, the higher the image's score, the more “confused” the model is.
As a result, data samples with higher entropy score should be offered for annotation.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/acquisition_metrics/acquisition_functions.py)


### Least Confidence {#least-confidence}

Rank images by their least confidence score.

_Least confidence_ (LC) score of a model's prediction is the difference between 1 (100% confidence) and its most confidently predicted class label.
The higher the LC score, the more “uncertain” the prediction.

The mathematical formula of the LC score of a model's prediction $x$ is:

$H(p) = 1 - \underset{y}{\max}(P(y|x))$

It can be employed to define a heuristic that measures a model’s uncertainty about the classes in an image using the average of the **LC** score of the model-predicted class probabilities in the image.
Like before, the higher the image's score, the more “confused” the model is.
As a result, data samples with higher LC score should be offered for annotation.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/acquisition_metrics/acquisition_functions.py)


### Margin {#margin}

Rank images by their margin score.

Margin score of a model's prediction is the difference between the two classes with the highest probabilities.
The lower the margin score, the more “uncertain” the prediction.

It can be employed to define a heuristic that measures a model’s uncertainty about the classes in an image using the average of the margin score of the model-predicted class probabilities in the image.
Like before, the lower the image's score, the more “confused” the model is.
As a result, data samples with lower margin score should be offered for annotation.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/acquisition_metrics/acquisition_functions.py)


### Variance {#variance}

Rank images by their variance.

Variance is a measure of dispersion that takes into account the spread of all data points in a data set.
The variance is the mean squared difference between each data point and the centre of the distribution measured by the mean.
The lower the variance, the more “clustered” the data points.

The mathematical formula of variance of a data set is:

$Var(X) = \frac{1}{n} \sum_{i=1}^{n}(x_i - \mu)^2, \text{where } \mu = \frac{1}{n} \sum_{i=1}^{n}x_i$

It can be employed to define a heuristic that measures a model’s uncertainty about the classes in an image using the average of the variance of the model-predicted class probabilities in the image.
Like before, the lower the image's score, the more “confused” the model is.
As a result, data samples with lower variance score should be offered for annotation.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/acquisition_metrics/acquisition_functions.py)


### Average Frame Score {#average-frame-score}

This method ranks images based on the mean score of their predicted objects, applicable specifically for object-level predictions such as bounding-box or segmentation.

The metric calculates the maximum confidence level for each class within every object-level prediction, averages these across all object predictions within an image, and assigns this average as the image's score. 
In the absence of any predictions for an image, a score of zero is assigned.

A lower score indicates that the model's predictions for an image lack certainty. This measurement is particularly effective in scenarios where the presence of at least one object is anticipated in every image.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/acquisition_metrics/acquisition_functions.py)