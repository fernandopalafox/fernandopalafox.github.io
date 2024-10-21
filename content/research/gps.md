---
title: WIP Gaussian Processes 
tags: [ml]
publishDate: 2024-10-20
draft: false
---

WIP

I've been looking into ways of incorporating [epistemic uncertainty](https://en.wikipedia.org/wiki/Uncertainty_quantification#Aleatoric_and_epistemic) into learned [world models](https://danijar.com/project/dreamerv3/).
David suggested I looking into using Gaussian processes (GPs) like James Harrison in his work on [Variational Bayesian Last Layers](https://arxiv.org/abs/2404.11599) or [ALPaCA](https://arxiv.org/abs/1807.08912). 
James' work looks super interesting, but once I started digging into it I realized I didn't really understand how GPs REALLY worked (even though I though I implemented one [here] lol), so in this post I'll derive them.

---

# Derivation

Consider a supervised learning problem, where we have a set of $N$ inputs $\mathbf{x} \in \mathbf{X}$ and outputs $\mathbf{y} \in \mathbf{Y}$ and we wish to predict $\mathbf{y^*} \in \mathbf{Y^*}$ at test inputs $\mathbf{x^*} \in \mathbf{X^*}$.
In many prediction tasks, we find parameters $\mathbf{\theta}$ for a model $f(\mathbf{x};\theta)$ such that $f(\mathbf{x};\theta) \approx \mathbf{y}, \forall \mathbf{y} \in \mathbf{Y}$. 
That is, we find parameters that when plugged into the model produces outputs $\mathbf{Y}$ given inputs $\mathbf{X}$.
Then, if we assume our model can generalize, we can use the same same model to predict what the outputs would look like for unseen inputs $\mathbf{x^*}$ (test inputs).
This is what's done in [maximum-likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation).

Gaussian processes are different because they produce estimates over functions instead of parameters.
That is, instead of estimating the most likely $\theta$ given the data(by finding $p(\theta|\mathbf{X}), \mathbf{Y}$), we estimate the most likely *function* given the data by finding $p(f|\mathbf{X}, \mathbf{Y})$.

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
Then, we assume this prior is a Gaussian with mean $\mu(\mathbf{X})$ and covariance $\mathbf{K}(\mathbf{X})$ where
$$
    \mu(\mathbf{x}) = 
    \begin{bmatrix}
        m(\mathbf{x}_1) \\
        \vdots \\
        m(\mathbf{x}_N)
    \end{bmatrix}
$$
and $\mathbf{K}(\mathbf{X})_{ij} = \kappa(\mathbf{x}_i, \mathbf{x}_j)$. 
$\kappa$ is known as the kernel function and it measures the similarity between two points $\mathbf{x}_i$ and $\mathbf{x}_j$, and $m$ is a mean function (usually set to zero).

We can sample the prior to get an idea what the functions look like before seeing the data. 
To do this, we define a set of $N$ input points (say a grid between -5 and 5), compute $K$ and $\mu$, and then generate samples from a Gaussian distribution with mean $\mu$ and covariance $K$.
Each sample will be a vector of $N$ outputs corresponding to a possible function $f(\mathbf{x})$.
The shape of each function is implicitly defined by our chosen kernel function $\kappa$. 
Below is a plot showing samples from a prior with a [squared exponential kernel](https://www.cs.toronto.edu/~duvenaud/cookbook/) where $\ell = \sigma = 1$.

<figure style="text-align: center;">
  <img src="media/gp_sample_prior.png" alt="" style="width:65%">
  <figcaption style="max-width: 95%; margin: auto;"><em></em></figcaption>
</figure>



Then, we can use Bayes' rule to find the posterior distribution over functions $p(\mathbf{f}|\mathbf{X}, \mathbf{Y})$.


## Questions
- What exactly is the uncertainty in GPs? Why does it make sense?  
- Explanation of ALPaCA





FINISH THIS LATER
Before diving into the details, a helpful intuition is that a GP predicts outputs for test points by measuring the similarity between the test and training points.
Then, it uses this similarity to test the 


To do so, we start by assuming the outputs $y=f(\mathbf{x}), \forall \mathbf{x} \in \mathbf{X}$ are drawn from a joint Gaussian distribution with mean $\mu(\mathbf{x})$ and covariance $\Sigma(\mathbf{x})$
That is, 
$$
    p(f(\mathbf{x}), \dots, f(\mathbf{x}_N)) = \mathcal{N}(\mathbf{f}; \mu(\mathbf{x}), \mathbf{\Sigma(\mathbf{x})})
$$
where $\mathcal{N}$ denotes a Gaussian distribution. 

Then, if we are able to 

## Resources

- Murphy's [Probabilistic ML](https://probml.github.io/pml-book/)
- Rasmussen's [Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/)