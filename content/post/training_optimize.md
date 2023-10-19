---
title: "Neural Network Model Training Optimization"
date: 2023-10-14T23:20:18+01:00
draft: false
author: "Pinaki Pani"
description: "Neural Network Training Optimzation."
feature_image: "https://plus.unsplash.com/premium_photo-1681586125940-635652c804b7?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2012&q=80"
tags: ["Deep Learning", "Python"]
---

# Training Neural Networks

---

Its important to learn to model a neural network and how to train the deep neural network to fit our data. There are times when we train our data to a neural network model to fit perfectly but still nothing works as planned. There could be a multitude of reasons for this failure, however some of the common things that we can look over so as to follow a generic guideline and to refrain from running into common failure issues. Issues like poor architecture, noisy data or even slow models are pretty common issues that we might run into while working with neural networks.

So we need to find some ways to optimize the training of our neural network models. Let's go over some of the crucial techniques that we should keep in mind while desigining the nerual network model architecture and as well as while training it.

### 1. Testing

---

Needless to say but almost every model needs to have been tested with training and test data. The available dataset is supposed to be divided into training and testing data with a certain percentage of split. The model is then trained over the training data and then tested over the test data to make sure that it is able to live up to a certain point of accuracy.

### 2. Overfitting and Underfitting:

---

Even in normal day to day life we have problems and there are times where we either over simplify it or undersimplify it. The concept is similiar in Machine learning. When the model is trained over our training data so precisely that the model is now only able to predict over the training data perfectly. Any modulation with a slightly different data just gives wrong outcome. This is called overfitting. The model will be able to only make correct predictions with very higih acuracy over the training data set, whereas fails grandly over the test dataset.

In underfitting the model hasn't been trained enough so as to make obvious correct predictions. Even over the training data the model makes a lot of mistakes. So to avoid such mistakes we need to train the model over our available data even more so as to make generalised predicitons.

To make corrections for our model we could maybe allow our model to overfit and then take measures to make corrections to it, notching it down a bit, so as to fit the data perfectly.

### 3. Early Stopping

---

Now let's say our model is very complicated, to be honest way too overly complicated than we need, and we have to live with it.
We could initially train a model for a few epochs so that it barely fits the training data. We then keep going and train the model for more number of epochs and keep checking how well the model fits our training data.

While we do this we also need to keep checking with our test data as well to make sure that the model doesn't start to overfit. We can figure this out when we see the training error keeps decreasing while the test error keeps on increasing. We can check the curves of the training and testing data results and check at what point the errors starts spreading keeps increasing as shown below in the graph.

### 4. Regularization

---

As we work more with neural networks we can conclude that at times when coeficients (i.e, **_theta value / weights_**) are large there are more and more chances of overfitting. Therefore it is important to penalize the weights that are way too large so as to decrease the effect of the features associated to these weights. This penalization is otherwise also known as Regularization. I am not going to go deeper into the derivations and how it works as there are tons of detailed explantion available for free, especially the one explained by _Andrew Ng in his coursera course_ regarding regularization. This course is a great starting point to understand the concept behind everything that happens.

Initially what we do is we take the old error function and add a term, which is big when the weights are big. Here the additional term could be use the sum of the absolute value of the coefficients or the other way is to take the sum of the square of the coefficients. We multiply these sum with a penalty parameter that determines how much to penalize the coefficients. We often refer to this parameter as lambda.

Whenever the absolute values are considered then this type of regularization is known as **L1 regularization**.
While if we consider the squares of the coefficients then this is known as **L2 regularization**.

- **L1 regalarization** is very useful when we need to get rid of some of the features from our dataset, otherwise known as _feature selection_ process. L1 generally gives a saprse vector.
- **L2 regularization** is rather useful when we are certain of all the features we are going to use in our model training and just need to adjust the coefficients asssociated with them, hence determining how effective a feature is to determine the final output. L2 usually doesnt favor sparse vector as it tries to maintain all the weights to be homogenously small.

### 5. Dropout

---

Sometimes one part of our neural network model has a very large weights while the others dont associate with larger coefficients. So what dropout suggests is to randomly take some nodes off of the model and then try to figure out the output. While this is one run or epoch the other epochs will have some other randomly selected sets of nodes to turn off and go ahead with our output prediction.
Such a parameter exists in our alogrithm that determines the _probability that each node will be dropped_. So for each epoch each nodes get dropped with a probability of the perentage mentioned in the said parameter. This method of dropping a particular node is termed as dropout and is commonly used in neural networks.

### 6. Local Minima

---

One of the interesting issues that we might come across often while trining neural network models is that our errors minimize up to a certain level and then they start increasing. This usually happens when gradient descent is used and the descent leads us to a certain level of minima but that not necessarily might be population minimam and is generally termed as local minima. So this might not give us the best error regularization, while it does allow the model to be regularised upto a certain level.

So now to solve issues as such we can take a few steps that could help us find the best possible population minima value.

> **Random Restart** - What random restart suggests is that we start doing gradient descent for our error from random places which could help us reach the global minima or atleast a pretty good local minima value.

### 7. Vanishing Gradient

---

One of the other issues that often occurs while training the neural network models is that the gradients are so small that it almost seems like it vanished. The logic behind vanishing gradient is that when we find the derivative of the error with respect to the coefficients of the nodes then by the rule of _**backpropagation**_ we can denote that it is the product of all the derivatives by the _**chain rule**_ calculated at the node in the corresponding path to the output.

These derivatives are derived from sigmoid function so they are already very small, and the products of these small numbers is even tinier. This tiny gradient makes the descent very difficult and slow. So its possible that we might never be able to reach a ceratin level of minimum value or else the time taken to reach a minima is way too high.

One of the ways that we can avoid vanishing gradients is using a different type of activation function. Activation functions such as **Hyperbolic Tangent function** or a **Rectified Linear Unit** activation function are often used.

> The range of values a sigmoid fucntion can return is from 0 to 1 while the **tanh()** function can return a range of values from -1 to 1, which in turn denotes that the deivatives for such fucntion will also be slightly larger than the derivatives of sigmoid fucntion.

> **relu()** on the other hand denotes that it will return a positive value as it is while if the result is negative it returns 0. The derivaties always gives 1 for positive numbers. It barely breaks linearity but provides better and complex non linear solutions without sacrificing much accuracy.

There are further more issues that one might come across while training their model. It is important to check for above issues and look out for more. A successful model training needs thorough knowledge of the dataset one is handling and working upon.
