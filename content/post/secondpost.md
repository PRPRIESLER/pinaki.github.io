---
title: "Generative Adversarial Networks"
date: 2021-06-27T19:25:28+05:30
draft: false
author: "Pinaki Pani"
description: "Sample article showcasing basic Markdown syntax and formatting for HTML elements."
feature_image: "https://images.unsplash.com/photo-1606778303077-3780ea8d5420?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1050&q=80"
tags: ["Deep Learning","GANs","Python"]
---

## Catch up with RNNs and key differences
-------

RNNs generate one word at a time similarly they also generate one pixel at a time for images. Whereas GANs help to generate a whole image in parallel. It uses a generator-discriminator Network. The generator model takes random noise and runs it through a differentiable function to transform/reshape it to a more realistic image. The random input noise that we chose determines what type of images to produce. 
The generator needs to be trained, but its training is different from usual supervised models. We show the model a lot of images and ask it to produce an image that comes out of the same probability distribution as of the images shown to it. Mostly it can be done by tweaking the parameters to increase the probability that the generator will generate something similar to our training dataset, but it is very difficult to arrive at that probability. So, a discriminator is what usually used to do the task of guiding our generator to achieve the required probability.

Initially this discriminator is tasked to generate probabilities of the images passed through it. When we begin, we start to get Images from our training dataset as well as images from the generator which would not look similar to the real training data.

`The discriminator assigns a probability close to 1 for our training/real images and a probability close to 0 for the generated images. This basically determines which images are real and which ones are not.`

The more the generator maps random noise to images the probability distribution over the generated images represented by the model becomes denser. The discriminator outputs high value wherever the density of real/training data is greater than the generated data. Generator changes the samples it produces to move uphill along the function learned by the discriminator. So basically, the generator moves its samples to the areas where the model distribution is not yet dense enough. At this point it becomes very difficult and the discriminator generates a probability of 0.5 as the images can now either be from the real dataset or maybe generated.

*Now this is the point where I need to add this note, as the explanation is the summary of the whole paper or if to be even simplified for the GAN paper that was written would be:*
>That the generator learns to generate images or let's say it generates data that has a similar distribution as of the distribution of the real data. This is how it tries to trick the discriminator into thinking that the generated image is actually from the real dataset when tested. Thus now we know what distribution the genertor data converges to.

## Payoffs and Equilibria (In context of Machine Learning!)
--------

Both Generator and Discriminator has their own cost function to be optimized. Now the cost for discriminator is just the negative of the cost for the generator. The minimum cost from the generator and the maximum cost for the discriminator causes the state of equilibria.
If we look at the Generator and the Discriminator architecture, we can see that there is always a hidden layer in both the cases as the need to have the universal approximator property, which basically mean that the model is able to approximate one continuous function space between two Euclidean spaces, where the approximation is done usually with respect to compact convergence theory.

_The output of the generator is calculated by a feedforward layer with tanh calculation to get the output between -1 and 1, and mostly the output of the discriminator is calculated by a sigmoid layer as we need it to be a probability._

Loss for both the gen and disc are defined and the losses are optimized simultaneously. This architecture is of DCGAN designed by Indigo and Facebook.
To scale GANs up to large images we need to use CNN.
Usually while working with CNN the size of the image is decreased while the depth is increased with every CNN and maxpooling layer. But we need to increase the size of the image as well as the depth for working with large images on GANs.
Now while training the Discriminator and the Generator we need to keep in mind that we need to smoothen the true labels for discriminator which is basically making the labels of real images from one to 0.9 as that will allow the network to work around more gradient nuances than just having 0 gradient. 
* ## Discriminator Training:
----
Get the loss on real/training images. After getting the loss we can generate the fake images, which will allow the discriminator to grab the fake loss on the generated images. Summing up both the losses we can perform backpropagation and then that allows us to update the discriminator parameter weights.

* ## Generator Training:
----
The generator generates fake images. Initially according to the base paper of GANs the basic idea is to train the generator in such a way that it could maximize the Discriminator loss, or in other way, minimize the loss: (1 – Disc(fake_image))

<img src="/images/second_formula.png" alt="drawing" width="200"/>

![](/images/second_formula.png)
which basically comes down to this minmax formula as shown above. 

Now there are many other techniques that has come forward for better and efficient generator and discriminator training. We can find more about it in the paper here :  [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)