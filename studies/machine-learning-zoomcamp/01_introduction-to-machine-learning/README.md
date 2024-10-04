# 📖 Introduction to Machine Learning

Imagine that you want to sell a car. You don't know how much it costs. You don't want to put a price that is too high because you will not sell it. You don't want to put a price that is too low because you will lose money.

In this case, you search in the internet and find a website that sells cars. You find a car that is similar to yours but with different characteristics.

| Year | Make | Mileage | ... | Price |
| ---- | ---- | ------- | --- | ----- |
| 1995 | Gaz  | 200.000 | ... | $1.1k |
| 1980 | Vaz  | 100.000 | ... | $0.6k |
| 2016 | Bwm  | 5.000   | ... | $23k  |

Using this informations, a machine learning model can be created to predict the price of your car. The information like year, make, and mileage are called **features** (what we know). The price is called the **target** (what we want).

# 📖 Machine Learning vs Rule-Based System

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

# 📖 Supervised Machine Learning

In supervised machine learning, we "teach" the model using a dataset with features and target. The model learns the relationship between the features and the target. 

In our case, we show a car and its price to the model. The model learns the relationship between the car's features and its price. So the model learns the patterns and can predict the price of a new car.

The matrix of features is called **X** and the target is called **y**. When we train a model, we want to find a function that maps the features to the target:

$g(X) \approx y$

The function $g$ is the model. We can user this logic to:

- Regression: predict a continuous value (price of a car)
- Binary classification: predict one of two classes (spam or not spam)
- Multiclass classification: predict one of many classes (cat, dog, horse, ...)
- Ranking: predict the order of items (search results like Google)

# 📖 CRISP-DM

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

