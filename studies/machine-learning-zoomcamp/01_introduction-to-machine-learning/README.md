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

