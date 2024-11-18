---
title: Belief Space Planning vs. ALPaCA
publishDate: 2024-11-18
draft: False
tags: [ct, ml]
---

I'm on a quest to understand how to define and use the uncertainty of a dynamical system without making any assumptions about the system's structure.
In the context of controls, in ["Belief Space Planning Assuming Maximum Likelihood Observations"](https://groups.csail.mit.edu/robotics-center/public_papers/Platt10.pdf) by Platt et al present B-LQR, an algorithm for computing controls while reasoning about expected changes in state uncertainty.

On the other hand, in ["Meta-Learning Priors for Efficient Online Bayesian Regression"](https://arxiv.org/abs/1807.08912) Harrison et al. present ALPaCA, a formalism for bayesian regression (which can be applied to a dynamical system) using a neural network with a variational last layer which provide a measure of output uncertainty.

I want to understand if there's a way of combining these two, so here I provide a brief comparison of both. 
This post is part of a series on Active Learning for Dynamical Systems. 
Checkout the previous post [here](dual.md).

-- 

## Bayesian Regression with ALPaCA

ALPaCA is basically a function approximator with extra bells and whistles to shape its inductive bias using samples of the functions it expects to find at test time.
In ALPaCA, we have a distribution (which defines uncertainty) over the parameters of the last layer, which are updated based on a Bayesian update that uses the observed data.
In the context of dynamical systems, the function we approximate is one whose inputs are actions and state, and the outputs are an expected next state. 
There is no control going on here.
Uncertainty updates are done using observed data and updating our distribution over the last layer. 
This distribution is then used to form an estimate of the uncertainty of the function output. 

## Belief Space Planning with B-LQR

This paper on the other hand, B-LQR has uncertainty over state, and they use an extended Kalman filter to estimate a belief.  
Unlike ALPaCA, their algorithm also generates a control trajectory using an extension of a linear quadratic regulator.
A clever part of B-LQR is that they make long-horizon control plans (which require some measure of uncertainty), by assuming the next measurements will be the current belief mean (i.e., max likelihood estimate) propagated one step forward using the dynamics model

## What's Next? 
Two ideas: now that we have a formalism for uncertainty in a dynamics model, how can we combine it with the planning in B-LQR? 
I'm going to try to write down a version of the expected next state that accounts for the last layer update in ALPaCA. 
If this works, it will be an uncertainty aware planner that uses a neural network for the system dynamics. 
Cool. 
