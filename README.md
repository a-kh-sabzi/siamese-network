# siamese-network

## Introduction

This project implements a siamese network that takes two characters as input and outputs a similarity score between 0 and 1. The output 0 means that the two input characters are completely different and output 1 means that they are exactly the same.

## Data set

This neural network was trained and tested on [Omniglot data set](https://github.com/brendenlake/omniglot) for one-shot learning.

![examples from the data set](/assets/images/omniglot_grid.jpg)

This data set includes 1623 hand drawn characters from 50 different alphabets. There are 20 examples for every character. Thus, we have 1623 different classes and 20 samples from each class.

The first 30 alphabets were used for training and the rest were used for evaluating the network. This means, we have 964 classes for training.

## Siamese Network

Sometimes the number of classes are dynamic or the number of samples per class is small. In these cases, we can't use normal networks. Because, we either have to retrain them as the number of classes change or the number of samples won't be enough to train an accurate network. A solution to these problems is to use siamese networks. 

Normal networks output a class label for classifying problems. But siamese networks output a similarity score that compares the two inputs of the network. These networks are made from two twin sub-networks that share their weights. Each of these sub-networks takes an input and extracts their features. Then these features are used to compare the inputs. 

As mentioned, Siamese Networks output a similarity score rather than a class label. This means that these networks can't be used for ened-to-end classifying and you still need further processing to classify a sample. 

## References

[Lake, B. M., Salakhutdinov, R., and Tenenbaum, J. B. (2015). Human-level concept learning through probabilistic program induction.](http://www.sciencemag.org/content/350/6266/1332.short) _Science_, 350(6266), 1332-1338.

[One Shot Learning with Siamese Networks using Keras](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d)
