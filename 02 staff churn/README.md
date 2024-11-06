# Staff Churn Prediction
Machine Learning (ML) Zoomcamp Mid-term Project


## The problem domain

Ask yourself, were it possible:

1. Would you like forewarning of a staff member leaving?
1. Which traits / indicators are particularly conducive to forewarning of staff churn?
1. Could staff survey responses be analysed to calculate a probability of a staff member leaving, so their manager can proactively respond?
1. Could all staff survey responses be regularly analysed to allow HR to plan around forecast churn?

Using a [public dataset](https://www.kaggle.com/datasets/mahmoudemadabdallah/hr-analytics-employee-attrition-and-performance) of employee survey responses and HR data -- for a particular US company and their survey questions -- this project attempts to find whether this is indeed possible, and the use cases that could thus be demonstrated thereby.

The value of such an innovation would be measured in both the attraction, retention, management intervention, and bottom line of such an organisation. However, doing so is not without trade-off.  This is rather the proof-of-concept, and it remains for others to address the operational, compliance and support impacts of such an approach.

After all, people are not data.  But data, here, could perhaps predict the behaviour of one's staff. And that alone, is the art-of-the-possible intended here.

## Objective

1.  Consider traits and insights that can be gleaned from the dataset to guide / understand the impacts on employees.
1.  Targeting the `Attrition` feature, build a machine learning model to predict whether a given employee is likely to churn
1.  Rank features that promote or reduce the likelihood of churn
1.  "Operationalise" the machine learning model by wrapping it in an API, itself wrapped into a Docker container