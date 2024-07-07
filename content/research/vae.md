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

Anyways, my first attempt at getting to know this space was to implement a Deep Variational Bayesian Filter as presented in [this paper](https://arxiv.org/abs/1605.06432). 
This architecture learns the dynamics of a latent space and can "imagine" how an image of a dynamical system will change given an input.
My initial goal was to reproduce their results in learning a representation of an image of a pendulum. 
Once the learning is done, the model should be able to produce "imagined" images of a moving pendulum given an input. 
However, after a couple weeks of work, I realized I bit off way more than I can chew.
So instead, I worked on a simpler example: no dynamics, just learn how to represent a pendulum. 
For this, I wrote up a pretty standard [variational autoencoder (VAE)](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73) in JAX and trained it on a dataset of pendulum images that I generated with a simple Gymnasium environment.
Here's the code ADD LINK. 
Here's a couple animations, one showing the learning process, and how reconstructed pendulums change as we move along the latent space. 

Lessons learned:
- It's probably a good idea to start off small in these projects, particularly when you're doing something new. Starting off with the the DVBF paper was probably a mistake, since there were too many moving parts for me to debug. 
- Read Karpathy's [Recipe for Training a Neural Networks](https://karpathy.github.io/2019/04/25/recipe/). Most useful points for me were to start by overfitting on a simple batch, and to never trust default learning schedules. 
- When implementing variational autoencoders, it's important to pay close to attention to the balance between the reconstruction term and the KL divergence term. Too much weight on the reconstruction term results in a posterior that does not match a Gaussian distribution, and your model becomes a regular autoencoder. This means that you've essentially learned a deterministic mapping between data and latent space, and you won't be able to generate new samples from the latent space. Too much weight on KL divergence and your posterior will match a Gaussian, but your reconstruction performance will likely suffer. 
- In the case of the pendulum, I struggled to strike this balance as evidenced by the zero variance. This is likely because the inputs are just too simple: it's a simple pendulum and there 