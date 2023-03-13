# Getting Started

To get started with using Encord Active for active learning, you should choose:
1. an Encord Active project,
2. a machine learning model and
3. an acquisition function.

Also, you need to take into account some basics on **dataset initialization** and **model selection** while you make your choices.
If you already have these principles covered, you can directly advance to #todo.


## Dataset initialization

In the active learning paradigm your model selects examples to be labeled, however, to make these selections you need a model from which you can get useful representations or uncertainty metrics - a model that already “knows” something about the data.

This is typically accomplished by training an initial model on a random subset of the training data. You would want to use just enough data to get a model that can make the acquisition function useful to kickstart the active learning process.

Also, **transfer learning** with pre-trained models can further reduce the required size of the seed dataset and accelerate the whole process.

:::tip
We recommend that initially you separate (not literally) your project data into training, test and validation sets as it’s important to note that the test and validation datasets still need to be selected randomly and annotated in order to have unbiased performance estimates.
:::


## Model selection

Selecting a model for active learning is not a straightforward task.

Often this is done primarily with domain knowledge rather than validating models with data.
For example, searching over architectures and hyperparameters using the initial seed training set.
However, models that perform better in this limited data setting are not likely to be the best performing once you’ve labeled 10x as many examples.
You should avoid using those models to select your data.

Instead, you should select data that optimizes the performance of your final model.
So you want to use the type of model that you expect to perform best on your task in general.


## Acquisition function selection




## Plug the model into an acquisition metric


## What's next?


talk about stopping criterion