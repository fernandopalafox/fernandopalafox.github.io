---
title: "VAE's and representation learning"
tags: [wh, ml]
publishDate: 2024-11-06
draft: true
---

I've become fascinated by [representation learning](https://en.wikipedia.org/wiki/Feature_learning). 
The basic idea is to feed tons of data (like images) into a neural network that learns how to reconstruct it from a simple representation of the images. 
This is typically done with an "information bottleneck" which forces the neural network to learn a lower-dimensional representation of the images. 
For example, let's say you're trying to reconstruct the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database), where each image is 28x28 pixels. 
Each image can be represented by a 784-dimensional vector, but, in theory, you should be able to reconstruct each image by learning what 10-dimensional vector the image corresponds to. 
This vector has 10 dimensions because the dataset has 10 possible digits to choose from, and each element of the vector corresponds to a digit.
So, an image with a "6" should map to a 10D vector where most elements are zero, and one element (the "6 dimension") has a value larger than zero. 
In this post I won't go into the details about how this process works (there's tons of tutorials online, like [this one](https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776)), instead I'm going to talk about why I'm interested in this field and what I'm doing to learn about it. 

The reason representations can be so exciting is because you can use it to learn a model of the world: feed it enough images and form a latent space that is expressive enough and you can use it to "imagine" images of the world, situations that haven't yet happened. 
For a robot, this can be immensely useful since it can reason about how the world will change given an action. 
Very cool. 
For some cool examples of this in action, take a look at [Danijar Hafner](https://danijar.com/) or [David Ha](https://worldmodels.github.io/)

Anyways, my first attempt at getting to know this space was to implement [this paper](https://arxiv.org/abs/1605.06432) they essentially learn the dynamics of a latent space and can "imagine" how an image of a dynamical system will change given an input. 
My initial goal was to reproduce their results in learning a representation of an image of a pendulum. 
Once the learning is done, the model should be able to produce "imagined" images of a moving pendulum given an input. 
However, after a couple weeks of work, I realized I bit off way more than I can chew.
So instead, I worked on a simpler example: no dynamics, just learn how to represent a pendulum. 

So, here it is. 
I wrote up a pretty standard [variational autoencoder (VAE)](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73) in JAX and ran it. 
The implementation is is pretty simple and I've shared the code here ADD LINK. 
Here's a couple animations, one showing the learning process, and how reconstructed pendulums change as we move along the latent space. 
