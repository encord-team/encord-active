---
sidebar_position: 2
---

# Performance by Metric

On this page, your model scores are displayed as a function of the metric that you selected in the sidebar. 
Samples are discretized into n equally sized buckets and the middle point of each bucket is displayed as the x-value in the plots. 
Bars indicate the number of samples in each bucket, while lines indicate the true positive rate and false negative rate of each bucket, respectively.

Metrics marked with (P) are metrics computed on your predictions.
Metrics marked with (F) are frame level metrics, which depends on the frame that each prediction is associated with. 

For metrics that are computed on predictions (P) in the "True Positive Rate" plot, the corresponding "label metrics" (O/F) _computed on your labels_ are used for the "False Negative Rate" plot.
