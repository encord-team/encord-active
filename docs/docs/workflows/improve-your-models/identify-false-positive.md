---
sidebar_position: 3
---
# Identifying False Positives

Using the false positive tab in Encord you can quickly identify in which areas your model fails. With this
functionality you can, for example:
- Detect missing ground-truth labels
- Diagnose annotation errors  
- Learn which classes your model confuses

and more, depending on your use cases.

`Prerequisites:` Dataset, Labels, Predictions 

### Steps:
1. Navigate to the _Model Assertions_ > _False Positives_ tab on the left sidebar.
2. Visualise predictions and try to get insights on where model fails.
3. Under each image, an explanation is given for why the prediction is false positive. The three reasons are:
   - No overlapping with the ground-truth object. This means that there is no label with the same class as the predicted
class which overlaps with the prediction.
   - IoU is too low. This means that the prediction does overlap with a label of the same class. However, the IoU 
between the prediction and the label is lower than the IoU threshold which is selected in the top bar.
   - Prediction with higher confidence is already matched with the ground-truth object.
Since the mAP score chooses the prediction with the highest model confidence that has an IOU larger than the set threshold, other predictions that matched the label with a sufficiently high IOU will be considered false positives. 
4. Note, that the boxed magenta object is the prediction, while the remaining objects are labels for the same 
image/frame.

