# 📚 Introduction to Machine Learning

Imagine that you want to sell a car. You don't know how much it costs. You don't want to put a price that is too high because you will not sell it. You don't want to put a price that is too low because you will lose money.

In this case, you search in the internet and find a website that sells cars. You find a car that is similar to yours but with different characteristics.

| Year | Make | Mileage | ... | Price |
| ---- | ---- | ------- | --- | ----- |
| 1995 | Gaz  | 200.000 | ... | $1.1k |
| 1980 | Vaz  | 100.000 | ... | $0.6k |
| 2016 | Bwm  | 5.000   | ... | $23k  |

Using this informations, a machine learning model can be created to predict the price of your car. The information like year, make, and mileage are called **features** (what we know). The price is called the **target** (what we want).

# 📚 Machine Learning vs Rule-Based System

In a rule-based system, we would have to write a set of rules to predict the price of the car. Let's take another example: predicting if an email is spam or not.

Supose that we have some emails and we read them. We find that the word "money" appears in some spam emails. So, we write a rule:

```python
def is_spam(email):
    if "money" in email:
        return True
    else:
        return False
```

Now, we notice that the word "promotion" appears in some spam emails. So, we write another rule:

```python
def is_spam(email):
    if "money" in email:
        return True
    elif "promotion" in email:
        return True
    else:
        return False
```

This process may be repeated many times. The problem is that we can't write all the rules, that's not scalable. Machine learning can help us to solve this problem.

We can use our rules as features for a machine learning model. Example:

- Money is in the email
- Promotion is in the email
- Title length is greater than 15
- Body length is greater than 100
- ...
- Target: spam or not spam

Now, we convert our emails into a table with features and target:

- Email 1: [1, 0, 1, 1, ..., 1]
- Email 2: [0, 1, 0, 1, ..., 0]
- Email 3: [1, 1, 1, 0, ..., 1]
- ...

The prediction of the model can be a probability to be spam or not spam. So we can write a rule to classify the email as spam if the probability is greater than 0.5.

# 📚 Machine Learning Types

## 📖 Supervised Machine Learning

Supervised learning is a type of machine learning where the model is trained on a labeled dataset. The dataset consists of input data and the corresponding output data. The model learns to map the input data to the output data.

- **Regression**: The output is a continuous value.
Ex.: Predicting the price of a car.
- **Classification**: The output is a category.
Ex.: Predicting if an email is spam or not spam.

## 📖 Unsupervised Machine Learning

Unsupervised learning is a type of machine learning where the model is trained on an unlabeled dataset. The model learns to find patterns in the data.

- **Clustering**: The model groups similar data points together.
Ex.: Grouping customers by their preferences.
- **Association**: The model finds rules that describe large portions of the data.
Ex.: Finding products that are frequently bought together.

## 📖 Semi-Supervised Machine Learning

Semi-supervised learning is a type of machine learning where the model is trained on a dataset that contains a small amount of labeled data and a large amount of unlabeled data.

- **Classification**: The model is trained on a small amount of labeled data and a large amount of unlabeled data.
Ex.: Predicting if an email is spam or not spam using a small amount of labeled emails and a large amount of unlabeled emails.
- **Clustering**: The model is trained on a small amount of labeled data and a large amount of unlabeled data.
Ex.: Grouping customers by their preferences using a small amount of labeled data and a large amount of unlabeled data.

## 📖 Reinforcement Learning

Reinforcement learning is a type of machine learning where the model learns to make decisions by interacting with an environment. The model learns to maximize a reward by taking actions in the environment.

- **Classification**: The model learns to make decisions by interacting with an environment.
Ex.: Training a robot to play a game.
- **Control**: The model learns to make decisions by interacting with an environment.

# 📚 Main Challenges of Machine Learning

## 📖 Variables, Pipelines and Controlling Chaos

The different types of variables can be a challenge in machine learning. We have different types of variables:
- Binary
- Categorical
- Continuous
- Discrete
- Time
- ...

To solve the problem of have different types of variables, we can use pipelines. A pipeline is a sequence of data processing components. Each component receives data, processes it, and passes it to the next component. Pipelines are very common in machine learning.

Controling chaos is another challenge in machine learning. We have to deal with a lot of data, a lot of features, a lot of models, a lot of hyperparameters, etc. We have to control all of this to make the model work.

A good pratice is follow some principles:
- Assume you are going to iterate a lot
- Give yourself time within the project deadlines
- Be sysematic. Normaly, change one thing at a time
- Nothing is lost. You learn something from every experiment
- Perfection is the enemy of good. Be clear on your objective and stop once you reach it
- Nothing is fixed (data, code, hyperparameters, etc).

## 📖 Train, Dev and Test sets

Making good choices in how you set up your training, development, and test sets can have a big impact on the performance of your model.

In previous Machine Learning era, it was common to split the data into two sets: training and test. The training set was used to train the model, and the test set was used to evaluate the model.

Nowadays, it is common to split the data into three sets: training, development, and test:

- **Training set**: Used to train the model.
- **Development set**: Used to evaluate the model and make decisions about the model.
- **Test set**: Used to evaluate the model generalization.

In previous Machine Learning era, it was common separate the data into [60, 20, 20], for example. Nowadays, in Big Data era, it is common to separate the data into [98, 1, 1], for example.

Rule of the thumb: the development and test sets should come from the same distribution.

## 📖 Bias vs Variance

Bias and variance are two sources of error in machine learning models. Bias is the error introduced by approximating a real-world problem, which may be extremely complicated, by a much simpler model. Variance is the error introduced by the model's sensitivity to small fluctuations in the training set.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20230307114300/image_2023-03-07_114312041.png" width="300">

In a case that we have high bias and lol variance, we call it **underfitting**. In a case that we have low bias and high variance, we call it **overfitting**.

|    | Scenario 1 | Scenario 2 | Scenario 3 | Scenario 4 |
|--- | ---------- | ---------- | ---------- | ---------- |
|Train set Error| 1%  | 15% | 15% | 0.5% |
|Dev set Error  | 16% | 16% | 30% | 1%   |
| Bias          | Low | High| High| Low  |
| Variance      | High| Low | High| Low  |

Train set error is utilized to measure bias. The diferenct between train set error and dev set error is utilized to measure variance.

- High Bias? Try bigger network, train longer, try another architecture, etc.
- High Variance? More data, regularization, try another architecture, etc.

# 📚 Decision Trees

## 📖 Introduction

Decision Trees are a type of model used in machine learning for classification and regression tasks. They are easy to understand and interpret, making them a popular choice for many applications.

A Decision Tree is a tree-like structure where each internal node represents a feature (or attribute). The decision of wich node is next is based on the value of the feature. This follows some criteria, like entropy, gini index, chi-square, etc.

## 📖 Mathematical foundations

Entrpy is an indicator of how messy the data is. The formula for entropy is:

$ H(X) = - \sum_{i=1}^{c} P(x_i) \log_2 P(x_i) $

Where:

- $ H(X) $ is the entropy of the dataset
- $ P(x_i) $ is the probability of class $ x_i $

For a binary classification problem, the formula is:

$ H(X) = - P(x_1) \log_2 P(x_1) - P(x_2) \log_2 P(x_2) $

Why do we use entropy in Decision Trees? For each decision, we want to reduce the entropy of the dataset.

How do we know that we need to create a node? We calculate the conditional entropy. The formula for conditional entropy is:

$ H(T|X) = \sum_{c \in x} \frac{|X_c|}{|X|} H(X_c) $

Where:

- $ H(T|X) $ is the conditional entropy of the dataset
- $ |X_c| $ is the number of samples in class $ c $
- $ |X| $ is the number of samples in the dataset
- $ H(X_c) $ is the entropy of the class $ c $

If we have Information Gain (reduction of entropy), we split the dataset. The formula for Information Gain is:

$ IG(T, X) = H(T) - H(T|X) $

Where:

- $ IG(T, X) $ is the Information Gain of the dataset

That is one approach to create a Decision Tree. Another approach is to use the Gini Index, for example. The Gini Index is a measure of how often a randomly chosen element would be incorrectly classified. The formula for the Gini Index is:

$ Gini(X) = 1 - \sum_{i=1}^{c} P(x_i)^2 $

Where:

- $ Gini(X) $ is the Gini Index of the dataset
- $ P(x_i) $ is the probability of class $ x_i $

These approaches have the same goal: increase the Information Gain.

# 📚 Evaluation Metrics

## 📖 How to choose an evaluation metric?

We have different evaluation metrics for different types of problems. For classification, for example, we have some metrics:

- Threshold metrics: Ratio when a prediction does not match.
    - Accuracy
    - Error
    - Sensitivity
    - Specificity
    - G-Mean
    - Precision
    - Recall
    - Fbeta-measure

- Ranking metrics: Based on score of a class membership and variaton of the threshold to measure the effectiviness of the classifier.
    - ROC curve
    - ROC AUC
    - Precision-Recall curve

- Probability metrics: Quantify uncertainty in a classifier's prediction.
    - Logarithmic loss
    - Brier score

<img src="https://machinelearningmastery.com/wp-content/uploads/2019/12/How-to-Choose-a-Metric-for-Imbalanced-Classification-latest.png" width="500">

## 📖 Threshold metrics

<img src="https://images.datacamp.com/image/upload/v1701364260/image_5baaeac4c0.png" width="500">

- **Accuracy**: $ \frac{TP + TN}{TP + TN + FP + FN} $
    Of all the classes, how much we predicted correctly.
- **Error**: $ 1 - Accuracy $
    Of all the classes, how much we predicted incorrectly.
- **Sensitivity**: $ \frac{TP}{TP + FN} $
    Of all the positive true classes, how much we predicted correctly.
- **Specificity**: $ \frac{TN}{TN + FP} $
    Of all the negative true classes, how much we predicted correctly.
- **G-Mean**: $ \sqrt{Sensitivity \times Specificity} $
    The geometric mean of Sensitivity and Specificity.
- **Precision**:
    - PPV(Positive Predictive Value): $ \frac{TP}{TP + FP} $
        Of all the positive predicted classes, how much we predicted correctly.
    - NPV(Negative Predictive Value): $ \frac{TN}{TN + FN} $
        Of all the negative predicted classes, how much we predicted correctly.
- **Recall**: $ \frac{TP}{TP + FN} $
    Of all the positive classes, how much we predicted correctly.
- **Fbeta-measure**: $ \frac{(1 + \beta^2) \times Precision \times Recall}{\beta^2 \times Precision + Recall} $ where $ \beta $ is a parameter that determines the weight of Precision and Recall.
    The harmonic mean of Precision and Recall.

## 📖 Ranking metrics

Ranking metrics are used when we need to evaluate how effective are the classifier's to separate the classes.

- **ROC (Receiver Operating Characteristic) curve**: A graphical representation of the trade-off between true positive rate and false positive rate.

<img src="https://miro.medium.com/v2/resize:fit:1200/1*Bgc9QOjhnL70g2SQxyj6hQ.png" width="300">

**True Positive Rate (TPR) (Sensitivity)**: $ \frac{TP}{TP + FN} $

**False Positive Rate (FPR) (Specificity)**: $ \frac{FP}{FP + TN} $

**AUC (Area Under the Curve)**: The area under the ROC curve. The higher the AUC, the better the model.

**PR (Precision-Recall) curve**: A graphical representation of the trade-off between Precision and Recall.

<img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*6QPLsDvjo4H6OZrxEBI8Fg.png" width="350">

# 📚 Train and Validation

The first step is to split the data into training and validation sets. The **training set** is used to train the model, and the **validation set** is used to evaluate the model.

The training set is sent to data **preparation and model training** (normalization, outliers removal, ...). Now, we use the same training set and the validation set to evaluate the model in **train evaluation**.

Next, we check evaluation. If the model is not good, we have to **tunning** the hyperparameters. If the model is good, we create the **inference artifact**.

So, the train set is divided in training and validation sets. We can divide just using holdout ou we can use cross-validation.

