---
title: "Acquisition functions"
slug: "active-acquisition-functions"
hidden: true
metadata: 
  title: "Acquisition functions"
  description: "Enhance model training with Encord Active's diverse acquisition functions: uncertainty-based & diversity-based strategies. Optimize active learning."
  image: 
    0: "https://files.readme.io/b6ed5cd-image_16.png"
createdAt: "2023-08-01T15:17:23.578Z"
updatedAt: "2023-08-09T11:44:29.536Z"
category: "6480a3981ed49107a7c6be36"
---

We want you to select the data samples that will be the most informative to your model, so a natural approach would be to score each sample based on its predicted usefulness for training. Since labeling samples is usually done in batches, you could take the top _k_ scoring samples for annotation. This type of function, that takes an unlabeled data sample and outputs its score, is called _acquisition function_.

## Uncertainty-based acquisition functions

In **Encord Active**, we employ the _uncertainty sampling_ strategy where we score data samples based on the uncertainty of the model predictions. The assumption is that samples the model is unconfident about are likely to be more informative than samples for which the model is very confident about the label.

We include the following uncertainty-based acquisition functions:

- [Least confidence](https://docs.encord.com/docs/active-model-quality-metrics#least-confidence)

<img src="https://storage.googleapis.com/docs-media.encord.com/static/img/ea-equations/least-confidence-equation.png" width="480" />

- [Margin](https://docs.encord.com/docs/active-model-quality-metrics#margin)

<img src="https://storage.googleapis.com/docs-media.encord.com/static/img/ea-equations/margin-equation-2.png" width="300" /> , Where <img src="https://storage.googleapis.com/docs-media.encord.com/static/img/ea-equations/margin-equation-y1.png" width="18" /> and <img src="https://storage.googleapis.com/docs-media.encord.com/static/img/ea-equations/margin-equation-y2.png" width="18" /> are the first and second-highest predicted labels.

- [Variance](https://docs.encord.com/docs/active-model-quality-metrics#variance)

<img src="https://storage.googleapis.com/docs-media.encord.com/static/img/ea-equations/variance-equation.png" width="650" />

- [Entropy](https://docs.encord.com/docs/active-model-quality-metrics#entropy)

<img src="https://storage.googleapis.com/docs-media.encord.com/static/img/ea-equations/entropy-equation.png" width="450" />

- [Mean object confidence](https://docs.encord.com/docs/active-model-quality-metrics#mean-object-confidence)

<img src="https://storage.googleapis.com/docs-media.encord.com/static/img/ea-equations/mean-object-confidence-equation.png" width="250" />

<!--

Least confidence: U(x) = 1 - P_\theta(\hat{y}|x), where \hat{y} = \underset{y \in \mathcal{Y}}{\arg\max} P_\theta(y|x)

Margin:  U(x) = P_\theta(\hat{y_1}|x) - P_\theta(\hat{y_2}|x), where \hat{y_1}$ and $\hat{y_2} are the first and second highest-predicted labels

U(x) = P_\theta(\hat{y_1}|x) - P_\theta(\hat{y_2}|x)\;where\; \hat{y_1}\: and\: \hat{y_2} \: are \: the \: first \: and \: second \: highest \: predicted \: labels


Variance: $U(x) = Var(P_\theta(y|x)) = \frac{1}{|Y|} \underset{y \in \mathcal{Y}}{\sum} (P_\theta(y|x) - \mu)^2$, where $\mu = \frac{1}{|Y|} \underset{y \in \mathcal{Y}}{\sum} P_\theta(y|x)$


Entropy: $U(x) = \mathcal{H}(P_\theta(y|x)) = -\underset{y \in \mathcal{Y}}{\sum} P_\theta(y|x) \log P_\theta(y|x)$


Mean object confidence: $U(x) = \frac{1}{N} \sum_{i=1}^{N} {\max} P_\theta^i(y|x)$

\-->

> 👍 Tip
> Follow the links provided for each acquisition function for detailed explanations of each, including alternative formulas, guidance on interpreting the output scores, and its implementation in GitHub.


> 🚧 Caution
> On the following scenarios, uncertainty-based acquisition functions must be used with extra care:
> - Softmax outputs from deep networks are often not calibrated and tend to be quite overconfident.
> - For convolutional neural networks, small, seemingly meaningless perturbations in the input space can completely change predictions.


## Diversity-based acquisition functions

Diversity sampling is an active learning strategy that aims to ensure that the labeled training set includes a broad range of examples from across the input space. The underlying assumption is that a diverse set of training examples will allow the model to learn a more generalized representation, improving its performance on unseen data.

In contrast to uncertainty-based methods, which prioritize examples that the model finds difficult to classify, diversity-based methods prioritize examples based on their novelty or dissimilarity to examples that are already in the training set. This can be particularly useful when the input space is large and the distribution of examples is uneven.

We include the following diversity-based acquisition function:

- [Image Diversity](https://docs.encord.com/docs/active-data-quality-metrics#image-diversity)

This metric clusters the images according to number of classes in the <<glossary:Ontology>> file. Then, it chooses samples from each cluster one-by-one to form an equal number of samples from  each cluster. Samples are chosen according to their proximity to cluster centroids (closer samples will be chosen first).

> 👍 Tip
> Diversity-based acquisition functions are generally easier to use compared to the uncertainty-based functions because they may not require any ML model. See the [Running Diversity Based Acquisition Function on Unlabeled Data](https://docs.encord.com/docs/active-diversity-sampling-on-unlabeled-data) tutorial to learn how to use them in you project easily.


## Which acquisition function should I use?

_“Ok, I have this list of acquisition functions now, but which one is the best? How do I choose?”_

This isn’t a question for which there is an easy answer - it heavily depends on your problem, your data, your model, your labeling budget, your goals, etc. This choice can be crucial to your results and comparing multiple acquisition functions during the active learning process is not always feasible.

Simple uncertainty measures like least confident score, margin score and entropy make good first considerations.

> 👍 Tip
> If you’d like to talk to an expert on the topic, the Encord ML team can be found in the #general channel in our Encord Active [Slack community](https://join.slack.com/t/encordactive/shared_invite/zt-1hc2vqur9-Fzj1EEAHoqu91sZ0CX0A7Q).

## How can I utilize the chosen acquisition function?

Explore the [Easy Active Learning on MNIST](https://docs.encord.com/docs/active-easy-active-learning-mnist) tutorial for a quick overview using a well-known example dataset.

### Tutorials

[Easy Active Learning on MNIST](https://docs.encord.com/docs/active-easy-active-learning-mnist): A quick overview of the acquisition functions using a well-known example dataset.  
[Diversity sampling without an ML model](https://docs.encord.com/docs/active-diversity-sampling-on-unlabeled-data): Using diversity sampling to rank images without training any model.  
[Selecting hard samples for object detection](https://github.com/encord-team/encord-active/blob/main/examples/active%20learning/object-detection/select-hard-samples-to-annotate.ipynb): A jupyter notebook to run acquisition functions using an object detector.
