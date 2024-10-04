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


# 📚 CRISP-DM

CRISP-DM stands for Cross-Industry Standard Process for Data Mining. It is a process model that describes the common approaches used by data mining experts.

<img src="https://upload.wikimedia.org/wikipedia/commons/b/b9/CRISP-DM_Process_Diagram.png" width="300">

Let's take a example of a project to predict the price of a car:

- **Business Understanding**: Understand the problem and the goal of the project. The goal is to predict the price of a car using its features.
- **Data Understanding**: Collect and understand the data. The data is a table with cars features and prices.
- **Data Preparation**: Clean and prepare the data. Remove missing values, normalize the data, etc.
- **Modeling**: Create and evaluate the model. Train the model using the data.
- **Evaluation**: Evaluate the model. Check if the model is good.
- **Deployment**: Deploy the model. Use the model to predict the price of new cars.

# 📖 Model Selection Process

