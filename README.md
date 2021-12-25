# siamese-network

## Introduction

This project implements a siamese network that takes two characters as input and outputs a similarity score between 0 and 1. The output 0 means that the two input characters are completely different and output 1 means that they are exactly the same.

## Data set

This neural network was trained and tested on Omniglot data set for one-shot learning that can be found at:
<https://github.com/brendenlake/omniglot>

![examples from the data set](/assets/images/omniglot_grid.jpg)

This data set includes 1623 hand drawn characters from 50 different alphabets. There are 20 examples for every character. Thus, we have 1623 different classes and 20 samples from each class.
The first 30 alphabets were used for training and the rest were used for evaluating the network. This means, we have 964 classes for training.

## References

[Lake, B. M., Salakhutdinov, R., and Tenenbaum, J. B. (2015). Human-level concept learning through probabilistic program induction.](http://www.sciencemag.org/content/350/6266/1332.short) _Science_, 350(6266), 1332-1338.
