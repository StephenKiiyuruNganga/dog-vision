# 🐶End to end multiclass dog breed classification

In this epic journey, we:
1. Build an end to end multiclass image classifier using Tensorflow 2.0 and Tensorflow Hub
2. Deploy the model inside a streamlit app so that the world can enjoy it. Checkout the app [here]().

## 1. Problem

Identifying the breed of a dog given an image of a dog

## 2. Data

The data is from this [Kaggle competion](https://www.kaggle.com/c/dog-breed-identification).

## 3. Evaluation

Kaggle requires a csv listing the prediction probabilities for each dog breed of each test image. Read more on this [here](https://www.kaggle.com/competitions/dog-breed-identification/overview/evaluation).

## 4. Features

Information about the data:

- We're dealing with images (unstructured data) so it's best if we use deep/transfer learning
- There are 120 breeds of dogs (i.e. 120 different classes)
- There are 10,000+ images in the training and test sets
- The training set has labels
