# Bootstrapping, evaluation, and usage of current models
---

## Online observations
---

### Bootstrap

#### [How to Calculate Bootstrap Confidence Intervals For Machine Learning Results in Python](https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/)
From https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/

"It is important to both present the expected skill of a machine learning model a well as confidence intervals for that model skill"

"Present confidence interval: "confidence intervals provide a range of model skills and a likelihood that the model skill will fall between the ranges when making predictions on new
data. For example, a 95% likelihood of classification accuracy between 70% and 75%."

The Bootstrap: General way to estimate statistics that can be used to calculate empical confidence intervals and show skill score (=accuracy). 

Bootstrap involves two steps: **Calculate a Population of Statistics** and **Calculate Confidence Intervals**

"The first step is to use the bootstrap procedure to resample the original data a number of times and calculate the statistic of interest.

The dataset is sampled with replacement. This means that each time an item is selected from the original dataset, **it is not removed, allowing that item to possibly be selected again for the sample**.

The statistic is calculated on the sample and is stored so that we build up a population of the statistic of interest.

The number of bootstrap repeats defines the variance of the estimate, and more is better, often hundreds or thousands."

For each iteration you set a percentage of the available data --> OOB-samples (=samples not included in sample)

Test-size varies because the same pictures can be chosen several times in the training data. 

"Once the scores are collected, a histogram is created to give an idea of the distribution of scores" (Gaussian = normalfordelingskurve, non-Gaussian = alt andet).

---

#### [What Is Bootstrapping in Statistics](https://www.thoughtco.com/what-is-bootstrapping-in-statistics-3126172)
From https://www.thoughtco.com/what-is-bootstrapping-in-statistics-3126172

"Bootstrapping is a statistical technique that falls under the broader heading of resampling"

"One goal of inferential statistics is to determine the value of a parameter of a population."

Some data point may be dupicated and some may not be included in the bootstrap sample. 

There is some mathematical reason for this to work, even though it seems like you are doing the impossible. 

---

#### [Bootstrap aggregating bagging](https://www.youtube.com/watch?v=2Mg8QD0F1dQ) (video)
From https://www.youtube.com/watch?v=2Mg8QD0F1dQ

You compute the mean of all the different "bags" / "samples" and then you have the accuracy. 
_You compute the mean of the mean_ so to speak...

---

#### [Confidence Intervals from Bootstrap re-sampling](https://www.youtube.com/watch?v=iN-77YVqLDw)
From https://www.youtube.com/watch?v=iN-77YVqLDw

God video med Chris Wild. 

### Evaluation of Neural Networks

---

#### [Metrics To Evaluate Machine Learning Algorithms in Python](https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/)
From https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/

---

#### [How to Evaluate the Skill of Deep Learning Models](https://machinelearningmastery.com/evaluate-skill-deep-learning-models/)
From https://machinelearningmastery.com/evaluate-skill-deep-learning-models/

"Deep learning models are stochastic."

"We do that by splitting the data into two parts, fitting a model or specific model configuration on the first part of the data and using the fit model to make predictions on the rest, then evaluating the skill of those predictions. *This is called a train-test split and we use the skill as an estimate for how well we think the model will perform in practice when it makes predictions on new data*."

"A train-test split is a good approach to use if you have a lot of data or a very slow model to train, but the resulting skill score for the model will be noisy because of the randomness in the data (variance of the model)."

"This additional randomness gives the model more flexibility when learning, but can make the model less stable (e.g. different results when the same model is trained on the same data)."

#### 