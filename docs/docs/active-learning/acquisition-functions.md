# Acquisition Functions

We want you to select the data samples that will be the most informative to your model, so a natural approach would be to score each sample based on its predicted usefulness for training.
Since labeling samples is usually done in batches, you could take the top _k_ scoring samples for annotation.
This type of function, that takes an unlabeled data sample and outputs its score, is called _acquisition function_.

## Uncertainty-based acquisition functions

In **Encord Active**, we employ the _uncertainty sampling_ strategy where we score data samples based on the uncertainty of the model predictions.
The assumption is that samples the model is unconfident about are likely to be more informative than samples for which the model is very confident about the label.

We include the following uncertainty-based acquisition functions:
* Least Confidence $U(x) = 1 - P_\theta(\hat{y}|x)$, where $\hat{y} = \underset{y \in \mathcal{Y}}{\arg\max} P_\theta(y|x)$
* Margin $U(x) = P_\theta(\hat{y_1}|x) - P_\theta(\hat{y_2}|x)$, where $\hat{y_1}$ and $\hat{y_2}$ are the first and second highest-predicted labels
* Variance $U(x) = Var(P_\theta(y|x)) = \frac{1}{|Y|} \underset{y \in \mathcal{Y}}{\sum} (P_\theta(y|x) - \mu)^2$, where $\mu = \frac{1}{|Y|} \underset{y \in \mathcal{Y}}{\sum} P_\theta(y|x)$

* Entropy $U(x) = \mathcal{H}(P_\theta(y|x)) = -\underset{y \in \mathcal{Y}}{\sum} P_\theta(y|x) \log P_\theta(y|x)$

:::caution 
On the following scenarios, uncertainty-based acquisition functions must be used with extra care:
* Softmax outputs from deep networks are often not calibrated and tend to be quite overconfident.
* For convolutional neural networks, small, seemingly meaningless perturbations in the input space can completely change predictions.
:::


## Which acquisition function should I use?

_“Ok, I have this list of acquisition functions now, but which one is the best? How do I choose?”_ 

This isn’t an easy question to answer and heavily depends on your problem, your data, your model, your labeling budget, your goals, etc.
This choice can be crucial to your results and comparing multiple acquisition functions during the active learning process is not always feasible. 

This isn’t a question for which we can just give you a good answer.
Simple uncertainty measures like least confident score, margin score and entropy make good first considerations.

:::tip
If you’d like to talk to an expert on the topic, the Encord ML team can be found in the #general channel in our Encord Active [Slack workspace](https://join.slack.com/t/encordactive/shared_invite/zt-1hc2vqur9-Fzj1EEAHoqu91sZ0CX0A7Q).
:::