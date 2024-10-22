---
title: Gaussian Processes and Uncertainty in World Models
tags: [ml]
publishDate: 2024-10-20
draft: false
---

WIP

I've been looking into ways of incorporating [epistemic uncertainty](https://en.wikipedia.org/wiki/Uncertainty_quantification#Aleatoric_and_epistemic) into learned [world models](https://danijar.com/project/dreamerv3/).
David suggested I looking into using Gaussian processes (GPs) like James Harrison in his work on [Variational Bayesian Last Layers](https://arxiv.org/abs/2404.11599) or [ALPaCA](https://arxiv.org/abs/1807.08912). 
James' work looks super interesting, but once I started digging into it I realized I didn't really understand how GPs REALLY worked (even though I implemented one [here](research/gp.md) lol), so in this post I'll derive them.

---

## Definition

Consider a supervised learning problem, where we have a set of $N$ inputs $\mathbf{x} \in \mathbf{X}$ and outputs $\mathbf{y} \in \mathbf{Y}$ and we wish to predict $\mathbf{y^*} \in \mathbf{Y^*}$ at test inputs $\mathbf{x^*} \in \mathbf{X^*}$.
In many prediction tasks, we find parameters $\mathbf{\theta}$ for a model $f(\mathbf{x}|\theta)$ such that $f(\mathbf{x}|\theta) \approx \mathbf{y}, \forall \mathbf{y} \in \mathbf{Y}$. 
That is, we find parameters that when plugged into the model produces outputs $\mathbf{Y}$ given inputs $\mathbf{X}$.
Then, if we assume our model can generalize, we can use the same same model to predict what the outputs would look like for unseen inputs $\mathbf{x^*}$ (test inputs).
This is what's done in [maximum-likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation).

Gaussian processes are different because they produce estimates over functions instead of parameters.
That is, instead of estimating the most likely $\theta$ given the data by finding $p(\theta|\mathbf{X}, \mathbf{Y})$, we estimate the most likely *function*[^1] given the data by finding $p(f|\mathbf{X}, \mathbf{Y})$.

To do so, we start by defining a prior distribution over functions that's conditioned on the training data $\mathbf{X}$. 
We denote this distribution as $p(\mathbf{f}|\mathbf{X})$ where 
$$
    \mathbf{f} = 
    \begin{bmatrix}
        f(\mathbf{x}_1) \\
        \vdots \\
        f(\mathbf{x}_N)
    \end{bmatrix}.
$$
Then, we assume this prior is a Gaussian with mean $\mathbf{\mu}$ and covariance $\mathbf{K}$ where
$$
    \mathbf{\mu} = 
    \begin{bmatrix}
        m(\mathbf{x}_1) \\
        \vdots \\
        m(\mathbf{x}_N)
    \end{bmatrix}
$$
and $\mathbf{K}_{ij} = \kappa(\mathbf{x}_i, \mathbf{x}_j)$. 
$\kappa$ is known as the kernel function and it measures the similarity between two points $\mathbf{x}_i$ and $\mathbf{x}_j$, and $m$ is a mean function (usually set to zero).
Therefore, $\mathbf{f} \sim \mathcal{N}(\mathbf{f}| \mathbf{\mu}, \mathbf{K})$.

> [!info] Outputs are jointly Gaussian
> 
> Assuming the prior over functions is a Gaussian implies the outputs $\mathbf{f}$ are distributed according to a joint Gaussian distribution with mean $\mathbf{\mu}$ and covariance $\Sigma$. 
> That is, $p(f(\mathbf{x}), \dots, f(\mathbf{x}_N)) = \mathcal{N}(\mathbf{f}| \mathbf{\mu}, \mathbf{\Sigma})$. 
> 
> This is a strong assumption, but it is useful because it allows us to easily compute the posterior distribution over functions given training data (as we will see later) and therefore make predictions.

We can sample the prior to get an idea what the functions look like before seeing the data. 
To do this, we define a set of $N$ input points (say a grid between -5 and 5), compute $\mathbf{K}$ and $\mathbf{\mu}$, and then generate samples from a Gaussian distribution with mean $\mathbf{\mu}$ and covariance $\mathbf{K}$.
Each sample will be a vector of $N$ outputs corresponding to a realization of $\mathbf{f}$.
The shape of each function is implicitly defined by our chosen kernel function $\kappa$. 
Below is a plot showing samples from a prior with a [squared exponential kernel](https://www.cs.toronto.edu/~duvenaud/cookbook/) where $\ell = \sigma = 1$.

<figure style="text-align: center;">
  <img src="media/gp_sample_prior.png" alt="" style="width:65%">
  <figcaption style="max-width: 95%; margin: auto;"><em></em></figcaption>
</figure>

## Predicting using noise-free observations

Given training data consisting of inputs $\mathbf{X}$ and noiseless outputs $\mathbf{f}$, where $\mathbf{f}_i = f(\mathbf{x}_i), \forall i \in \{1,..,N\}$ we would like to predict outputs $\mathbf{y^*}$ at test inputs $\mathbf{x^*} \in \mathbf{X^*}$.
To do so using a GP, we must find a distribution over functions conditioned on the test inputs $\mathbf{X^*}$, training inputs $\mathbf{X}$, and training outputs $\mathbf{f}$.
That is, we seek a distribution $p(\mathbf{f^*}|\mathbf{X^*}, \mathbf{X}, \mathbf{f})$ where $\mathbf{f^*}$ are the outputs at the test inputs.

Recall we assumed that function outputs are jointly Gaussian. 
Therefore, we can write the joint distribution of the training and test outputs as
$$
\begin{align}
    \begin{bmatrix}
        \mathbf{f} \\
        \mathbf{f^*}
    \end{bmatrix} &\sim 
    \mathcal{N}
    \left(
        \begin{bmatrix}
            \mathbf{\mu} \\
            \mathbf{\mu^*}
        \end{bmatrix},
        \begin{bmatrix}
            \mathbf{K} & \mathbf{K}^* \\
            \mathbf{K}^{*T} & \mathbf{K}^{**}
        \end{bmatrix}
    \right),
\end{align}
$$
where $\mathbf{K} = \kappa(\mathbf{X}, \mathbf{X})$, $\mathbf{K}^* = \kappa(\mathbf{X}, \mathbf{X^*})$, and $\mathbf{K}^{**} = \kappa(\mathbf{X^*}, \mathbf{X^*})$.
Then, we can write the conditional distribution analytically using [known results](https://statproofbook.github.io/P/mvn-cond.html) as follows:
$$
\begin{align}
    p(\mathbf{f^*}|\mathbf{X^*}, \mathbf{X}, \mathbf{f}) &= \mathcal{N}(\mathbf{f^*}| \mathbf{\mu^c}, \mathbf{\Sigma^c}) \\
    \mathbf{\mu^c} &= \mathbf{\mu}^* + \mathbf{K}^{*T} \mathbf{K}^{-1} (\mathbf{f} - \mathbf{\mu}) \\
    \mathbf{\Sigma^c} &= \mathbf{K}^{**} - \mathbf{K}^{*T} \mathbf{K}^{-1} \mathbf{K}^*,
\end{align}
$$

Below is a plot of sampled functions from a posterior distribution given a set using a squared exponential kernel with $\ell = \sigma = 1$.
<figure style="text-align: center;">
  <img src="media/gp_posterior_samples.png" alt="" style="width:65%">
  <figcaption style="max-width: 85%; margin: auto;"><em>Posterior samples given 5 noiseless data pairs (black crosses). Shaded region denotes the 95% confidence interval (2 stds from mean).  </em></figcaption>
</figure>

In practice, observations are noisy. 
I will not cover that case here, but the derivation conditional distribution is very similar and can be found in Ch. 15 of [Murphy's Probabilistic ML](https://probml.github.io/pml-book/).


## Questions
- What exactly is the uncertainty in GPs? Why does it make sense?  
- How exactly can we define a distribution over functions? And why can we do this with a finite number of points?
- Explanation of ALPaCA


## Resources

- Murphy's [Probabilistic ML](https://probml.github.io/pml-book/)
- Rasmussen's [Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/)

[^1]: For the development of GPs, we will refer to functions as a finite-dimensional vector of function values at a set of input points. Technically, functions are infinitely-dimensional objects that require an infinite number of (input,output) pairs to be fully described (unless we have an explicit functional form for a function, e.g., $f(x) = x^2$). However, when working with GPs we are able to describe functions with a finite number of points because we assume that function values at different points are **correlated**. This correlation is defined according to a selected kernel function. However, the underlying function is still infinite-dimensional. Disclaimer: I'm still trying to wrap my head around these details. But I think this is the gist of it.