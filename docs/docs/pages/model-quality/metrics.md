---
sidebar_position: 1
---

# Metrics 

**Metric importance**: Measures the *strength* of the dependency between the metric and model 
                    performance. A high value means that the model performance would be strongly affected by 
                    a change in the metric. For example, high importance in 'Brightness' implies that a change
                    in that quantity would strongly affect model performance. Values range from 0 (no dependency) 
                    to 1 (perfect dependency, one can completely predict model performance simply by looking 
                    at this metric).

**Metric [correlation](https://en.wikipedia.org/wiki/Correlation)**: Measures the *linearity 
                    and direction* of the dependency between a metric and model performance. 
                    Crucially, this metric tells us whether a positive change in a metric 
                    will lead to a positive change (positive correlation) or a negative change (negative correlation)
                    in model performance. Values range from -1 to 1.
