# Acquisition Functions

We want you to select the data samples that will be the most informative to your model, so a natural approach would be to score each sample based on its predicted usefulness for training.
Since labeling samples is usually done in batches, you could take the top _k_ scoring samples for annotation.
This type of function, that takes an unlabeled data sample and outputs its score, is called _acquisition function_.

## Uncertainty-based acquisition functions

In **Encord Active**, we employ the _uncertainty sampling_ strategy where we score data samples based on the uncertainty of the model predictions.
The assumption is that samples the model is unconfident about are likely to be more informative than samples for which the model is very confident about the label.

We include the following uncertainty-based acquisition functions:

- [Least Confidence][ea-acquisition-function-least-confidence]

  $U(x) = 1 - P_\theta(\hat{y}|x)$, where $\hat{y} = \underset{y \in \mathcal{Y}}{\arg\max} P_\theta(y|x)$

- [Margin][ea-acquisition-function-margin]

  $U(x) = P_\theta(\hat{y_1}|x) - P_\theta(\hat{y_2}|x)$, where $\hat{y_1}$ and $\hat{y_2}$ are the first and second highest-predicted labels

- [Variance][ea-acquisition-function-variance]

  $U(x) = Var(P_\theta(y|x)) = \frac{1}{|Y|} \underset{y \in \mathcal{Y}}{\sum} (P_\theta(y|x) - \mu)^2$, where $\mu = \frac{1}{|Y|} \underset{y \in \mathcal{Y}}{\sum} P_\theta(y|x)$

- [Entropy][ea-acquisition-function-entropy]

  $U(x) = \mathcal{H}(P_\theta(y|x)) = -\underset{y \in \mathcal{Y}}{\sum} P_\theta(y|x) \log P_\theta(y|x)$

:::tip

By following the links provided for each acquisition function mentioned above, you will find detailed explanations of what each acquisition function represents, alternative formulas, guidance on interpreting the output scores, and also its implementation in GitHub.

:::

:::caution
On the following scenarios, uncertainty-based acquisition functions must be used with extra care:
- Softmax outputs from deep networks are often not calibrated and tend to be quite overconfident.
- For convolutional neural networks, small, seemingly meaningless perturbations in the input space can completely change predictions.
:::

## Diversity-based acquisition functions {#diversity-based-acquisition-function}

Diversity sampling is an active learning strategy that aims to ensure that the labeled training set includes a broad range of examples from across the input space. The underlying assumption is that a diverse set of training examples will allow the model to learn a more generalized representation, improving its performance on unseen data.

In contrast to uncertainty-based methods, which prioritize examples that the model finds difficult to classify, diversity-based methods prioritize examples based on their novelty or dissimilarity to examples that are already in the training set. This can be particularly useful when the input space is large and the distribution of examples is uneven.

We include the following diversity-based acquisition function:

- [Image Diversity][ea-acquisition-function-image-diversity]: This metric clusters the images according to number of classes in the 
ontology file. Then, it chooses samples from each cluster one-by-one to form an equal number of samples from 
each cluster. Samples are chosen according to their proximity to cluster centroids (closer samples will be 
chosen first).

:::tip
Diversity-based acquisition functions are generally easier to use compared to the uncertainty-based functions because 
they do not require any ML model. See [diversity-sampling-on-unlabeled-data example](../tutorials/diversity-sampling-on-unlabeled-data.mdx) to 
learn how to use them in you project easily.
:::

## Which acquisition function should I use?

_“Ok, I have this list of acquisition functions now, but which one is the best? How do I choose?”_

This isn’t an easy question to answer and heavily depends on your problem, your data, your model, your labeling budget, your goals, etc.
This choice can be crucial to your results and comparing multiple acquisition functions during the active learning process is not always feasible.

This isn’t a question for which we can just give you a good answer.
Simple uncertainty measures like least confident score, margin score and entropy make good first considerations.

:::tip
If you’d like to talk to an expert on the topic, the Encord ML team can be found in the #general channel in our Encord Active [Discord community](https://discord.gg/TU6yT7Uvx3).
:::

## How can I utilize the chosen acquisition function?

Explore the [Easy Active Learning on MNIST][easy-active-learning-in-mnist] tutorial for a quick overview using a well-known example dataset.

[ea-acquisition-function-least-confidence]: ../metrics/model-quality-metrics/#least-confidence
[ea-acquisition-function-margin]: ../metrics/model-quality-metrics/#margin
[ea-acquisition-function-variance]: ../metrics/model-quality-metrics/#variance
[ea-acquisition-function-entropy]: ../metrics/model-quality-metrics/#entropy
[easy-active-learning-in-mnist]: ../tutorials/easy-active-learning-on-mnist
