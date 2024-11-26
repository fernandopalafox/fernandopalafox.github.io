---
title: WIP Belief Space Planning and Deep Dynamics Uncertainty
publishDate: 2024-11-26
draft: False
tags: [ct, ml]
---

In belief space planning we typically have uncertainty about the state.
But what if we want to think about dynamics uncertainty too?
As I've explored in previous posts (LINK THOSE HERE), I'm interested in ways of making minimal assumptions about the dynamics while still being able to quantify its uncertainty.
In the case of ALPaCA, we talk about uncertainty over the last layer of neural network trained to imitate the state. 
In this post I'll explore how to combine belief space planning with dynamics uncertainty. 

Let's work on the case where we have no state uncertainty but we do have state uncertainty. 
In this case, our belief with be solely over the value of the last linear layer in the dynamics neural network. 
Can we do belief space planning for this case? 

# Pattern matching the EKF in B-LQR

In B-LQR, we use an EKF to estimate the state $\mathbf{x}_t$.
Let's start by writing down the equations used in the EKF in B-LQR:
The linearized state and observations dynamics are
$$
\begin{align}
    \mathbf{x}_{t+1} &\approx \mathbf{A}_t(\mathbf{x}_t - \mathbf{m}_t) + f(\mathbf{m}_t, \mathbf{u}_t)\\
    \mathbf{z}_t &\approx \mathbf{C}_t(\mathbf{x}_t - \mathbf{m}_t) + g(\mathbf{m}_t) + \mathbf{w}_t.
\end{align}
$$
How does this translate to the case where we are instead estimating what the "true" value of the dynamics parameters is?
Specifically, we maintain an estimate of the parameters defining the distribution of the last layer, $\theta$ (IS THIS RIGHT?). 
First pass at a the dynamics dynamics 
$$
    \mathbf{\theta}_{t+1} = A_t(\mathbf{\theta}_t - \hat{\mathbf{\theta}}_t) + f(\hat{\mathbf{\theta}}_t, \mathbf{u}_t)
$$
In this case, the transition function $f$ would be the recursive Bayesian update shown in the ALPaCA paper. 

Something about this rubs me the wrong way. 
In ALPaCA, we're adjusting our estimate of the mean and covariance of a distribution over the last layer. 
The random variable is what we're trying to estimate. 
On the other hand, in B-LQR, we don't try to estimate the random variable (which is measurement noise), instead we refine our belief over state. 
Aha! 
In ALPaCA, our measurement model is basically to do a forward pass on the neural network + linear layer and obtain an estimate of the next state.
Then, using the actual, observed next state, we do Bayesian linear regression to get an estimate for what K should be,
ALPaCA measurement equation is:
$$
    \mathbf{x}_{t+1} = \mathbf{K}^\top \mathbf{\phi}(\mathbf{x}_t, \mathbf{u}_t) + \mathbf{\epsilon}
$$
Where $\mathbf{x}_{t+1}$ is the observation and $\mathbf{K}$ is a function of the parameters $\mathbf{\theta}$.

Since this measurement model doesn't have the exact linear structure we need to run an EKF on $\mathbf{K}$, let's rearrange it.
Define a new measurement model as follows: 
$$
    \mathbf{x}_{t+1} =(\mathbf{\phi}^\top \otimes \mathbf{I})\mathrm{vec}(\mathbf{K}) + \mathbf{\epsilon} 
$$
where it can be shown that $\mathbf{\phi}^\top \otimes \mathbf{I} = \mathbf{K}^\top \mathbf{\phi}(\mathbf{x}_t)$.
So basically, we've turned the original ALPaCA formulation into a linear measurement model we can run an EKF on.

What's next? 
Think about how to integrate with rest of B-LQR 