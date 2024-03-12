---
title: "Active learning"
slug: "active-learning"
hidden: true
metadata: 
  title: "Active Learning"
  description: "Optimize model training with Encord Active's active learning tools. Efficient labeling, improved accuracy. Learn more about active learning."
  image: 
    0: "https://files.readme.io/ae0744a-image_16.png"
createdAt: "2023-07-11T17:05:59.696Z"
updatedAt: "2023-08-09T14:53:21.123Z"
category: "6480a3981ed49107a7c6be36"
---
The annotation process can sometimes be extensively time-consuming and expensive. While images and videos can often be scraped, or even taken automatically, labeling for tasks like segmentation and motion detection is laborious. Some domains, such as medical imaging, require domain knowledge from experts with limited accessibility.

When the unlabeled data is abundant, wouldn't it be nice if you could pick out the 5% of samples most useful to your model, rather than labeling large swathes of redundant data points? This is the idea behind active learning.

**Encord Active** provides you with the tools to take advantage of the active learning method - and is integrated with **Encord Annotate** to deliver the best annotation experience.

If you are already familiar with the active learning foundation, continue your read with an exploration of [Encord Active's acquisition functions](https://docs.encord.com/docs/active-model-quality-metrics#acquisition-functions) and our end-to-end tutorial about [easy active learning on MNIST](https://docs.encord.com/docs/active-easy-active-learning-mnist).

## What is active learning?

Active learning is an iterative process where a [machine learning model](https://encord.com/blog/introduction-to-building-your-first-machine-learning) is used to select the best examples to be labeled next. After annotation, the model is retrained on the new, larger dataset, then selects more data to be labeled until reaching a stopping criterion. This process is illustrated in the figure below.

<center>

![active-learning-cycle.svg](https://storage.cloud.google.com/docs-media.encord.com/static/img/images/active-learning/active-learning-cycle.svg)

</center>


Check out our [practical guide to active learning for computer vision](https://encord.com/blog/a-practical-guide-to-active-learning-for-computer-vision/) to learn more about active learning, its tradeoffs, alternatives and a comprehensive explanation on active learning pipelines.