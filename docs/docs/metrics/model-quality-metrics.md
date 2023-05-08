---
sidebar_position: 4
---

# Model Quality Metrics

Model quality metrics help you evaluate your data and labels based on a trained model and imported model predictions.


## Acquisition functions

Acquisition functions are a special type of model quality metric, primarily used in active learning to score data samples according to how informative they are for the model and enable smart labeling of unannotated data.

| Title                                                                                              | Metric Type | Data Type |
|----------------------------------------------------------------------------------------------------|-------------|-----------|
| [Entropy](#entropy) - <small>Rank images by their entropy.</small>                                 | `image`     |           |
| [LeastConfidence](#least-confidence) - <small>Rank images by their least confidence score.</small> | `image`     |           |
| [Margin](#margin) - <small>Rank images by their margin score.</small>                              | `image`     |           |
| [Variance](#variance) - <small>Rank images by their variance.</small>                              | `image`     |           |


### Entropy

Rank images by their entropy.

In information theory, the entropy of a random variable is the average level of “information”, “surprise”, or “uncertainty” inherent to the variable's possible outcomes.
The higher the entropy, the more “uncertain” the variable outcome.

The mathematical formula of entropy is:

$H(p) = -\sum_{i=1}^{n} p_i \log_{2}{p_i}$

It can be employed to define a heuristic that measures a model’s uncertainty about the classes in an image using the average of the entropies of the model-predicted class probabilities in the image.
Like before, the higher the image's score, the more “confused” the model is.
As a result, data samples with higher entropy score should be offered for annotation.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/acquisition_functions.py)


### Least Confidence

Rank images by their least confidence score.

_Least confidence_ (LC) score of a model's prediction is the difference between 1 (100% confidence) and its most confidently predicted class label.
The higher the LC score, the more “uncertain” the prediction.

The mathematical formula of the LC score of a model's prediction $x$ is:

$H(p) = 1 - \underset{y}{\max}(P(y|x))$

It can be employed to define a heuristic that measures a model’s uncertainty about the classes in an image using the average of the **LC** score of the model-predicted class probabilities in the image.
Like before, the higher the image's score, the more “confused” the model is.
As a result, data samples with higher LC score should be offered for annotation.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/acquisition_functions.py)


### Margin

Rank images by their margin score.

Margin score of a model's prediction is the difference between the two classes with the highest probabilities.
The lower the margin score, the more “uncertain” the prediction.

It can be employed to define a heuristic that measures a model’s uncertainty about the classes in an image using the average of the margin score of the model-predicted class probabilities in the image.
Like before, the lower the image's score, the more “confused” the model is.
As a result, data samples with lower margin score should be offered for annotation.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/acquisition_functions.py)


### Variance

Rank images by their variance.

Variance is a measure of dispersion that takes into account the spread of all data points in a data set.
The variance is the mean squared difference between each data point and the centre of the distribution measured by the mean.
The lower the variance, the more “clustered” the data points.

The mathematical formula of variance of a data set is:

$Var(X) = \frac{1}{n} \sum_{i=1}^{n}(x_i - \mu)^2, \text{where } \mu = \frac{1}{n} \sum_{i=1}^{n}x_i$

It can be employed to define a heuristic that measures a model’s uncertainty about the classes in an image using the average of the variance of the model-predicted class probabilities in the image.
Like before, the lower the image's score, the more “confused” the model is.
As a result, data samples with lower variance score should be offered for annotation.

Implementation on [GitHub](https://github.com/encord-team/encord-active/blob/main/src/encord_active/lib/metrics/acquisition_functions.py)
