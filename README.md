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

## Network Architecture

The siamese network used in this project is depicted below:

![model architecture](/assets/images/model_named.png)

The sub-model in the above image is shown below:

![sub model architecture](/assets/images/sub_model_named.png)

## Training

This network was trained on 19280 samples from 964 different classes. Optimiaztion was done using Adam optimizor with a learning rate of 0.001. Training batch size was 32. Each sample was paired with another sample from the same class and a sample from another random class. Thus, each epoch of training was done over 38560 pairs. The model was implemented in keras and it was trained for 10 epochs.

## Evaluation

This network was evaluated using the n-way one shot testing method. Each under test sample was compared with one sample from the same class and n-1 samples from other classes. If the similarity score for the correct class was the highest then the prediction was correct. This test was repeated k times for k different samples. The percentage of the correct predictions is the accuracy of the network. Input of a 20-way one shot testing for a sample is shown below.

![one shot testing's input](/assets/images/one_shot_testing.png)

Accuracy of 20-way one shot testing for 500 random samples from training images is 89.6%.

Accuracy of 20-way one shot testing for 500 random samples from evaluation images is 77.2%.

The network was trained for 10 epochs. These accuracies can increase with further training.  

## References

[Lake, B. M., Salakhutdinov, R., and Tenenbaum, J. B. (2015). Human-level concept learning through probabilistic program induction.](http://www.sciencemag.org/content/350/6266/1332.short) _Science_, 350(6266), 1332-1338.

[One Shot Learning with Siamese Networks using Keras](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d)
