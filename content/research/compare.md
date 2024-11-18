---
title: WIP Belief Space Planning vs. ALPaCA
publishDate: 2024-11-18
draft: False
tags: [ct, ml]
---

I'm on a quest to understand how to define and use the uncertainty of a dynamical system without making any assumptions about the system's structure.
In the context of controls, in ["Belief Space Planning Assuming Maximum Likelihood Observations"](https://groups.csail.mit.edu/robotics-center/public_papers/Platt10.pdf) by Platt et al introduce an algorithm for computing controls while reasoning about expected changes in state uncertainty.

On the other hand, in ["Meta-Learning Priors for Efficient Online Bayesian Regression"](https://arxiv.org/abs/1807.08912) Harrison et al. present ALPaCA, a formalism for bayesian regression (which can be applied to a dynamical system) using a neural network with a variational last layer which provide a measure of output uncertainty.

I want to understand if there's a way of combining these two, so here I provide a brief comparison of both. 
This post is part of a series on Active Learning for Dynamical Systems. 

-- 

## ALPaCA

ALPaCA is basically a function approximator with extra bells and whistles that allow it to shape its inductive bias with variations of the function it expects to find at test time. 
Part of the output of ALPaCA is a covariance, which gives us a measure of uncertainty. 
In the context of dynamical systems, the function we approximate is one whose inputs are actions and state, and then it outputs an expected next state. 
There is no control going on here.
Uncertainty updates are done using observed data and updating our distribution over the last layer. 
This distribution is then used to form an estimate of the uncertainty of the function output. 

On the other hand, B-LQR is essentially just a Kalman filter that's estimating state based on noisy measurements. 
Their algorithm generates a control trajectory to minimize a cost.
A clever part of their algorithm is that they make long-horizon control plans (which require some measure of uncertainty), by assuming the next measurements will be the current belief mean (i.e., max likelihood estimate) propagated one step forward using the dynamics model
Note, here uncertainty is over STATE, not dynamics. 

Another way of seeing the difference is by looking at what distributions we're using.
- ALPaCA: $p(y∣x,Dt)=\int p(y|x,\theta)p(\theta|D_t)d\thetap(y∣x,Dt​)=∫p(y∣x,θ)p(θ∣Dt​)dθ$. Uncertainty over a function value given inputs.
- B-LQR: $b_t = (m_t, \Sigma_t)$, mean and covariance for a Gaussian. Uncertainty over state given a measurement.

What if we have some online process for learning the belief dynamics? 
In Tedrake paper, we assume next mean will be current mean propagated one step forward.
This is a simple approximation, what if we do current mean propagated one step forward + some correction term (i.e., neural network).
This should improve long-horizon planning over a simple mean approach. 
And once it learned, it's differentiable, perhaps making dual control a bit easier? 
