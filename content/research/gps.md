---
title: Gaussian Processes 
tags: [ml]
publishDate: 2024-10-20
draft: false
---

I've been looking into ways of incorporating [epistemic uncertainty](https://en.wikipedia.org/wiki/Uncertainty_quantification#Aleatoric_and_epistemic) into learned [world models](https://danijar.com/project/dreamerv3/).
David suggested I looking into using Gaussian processes (GPs) like James Harrison in his work on [Variational Bayesian Last Layers](https://arxiv.org/abs/2404.11599) or [ALPaCA](https://arxiv.org/abs/1807.08912). 
James' work looks super interesting, but once I started digging into it I realized I didn't really understand how GPs REALLY worked (even though I though I implemented one [here] lol), so in this post I'll derive them.

---

Consider a supervised learning problem, where we have a set of inputs $\mathbf{x} \in \mathbf{X}$ and outputs $\mathbf{y} \in \mathbf{Y}$ and we wish to predict $\mathbf{y^*} \in \mathbf{Y^*}$ at test inputs $\mathbf{x^*} \in \mathbf{X^*}$.
In many prediction tasks, we find parameters $\mathbf{\theta}$ for a model $f(\mathbf{x};\theta)$ such that $f(\mathbf{x};\theta) \approx \mathbf{y}, \forall \mathbf{y} \in \mathbf{Y}$. 
That is, we find parameters that when plugged into the model produces outputs $\mathbf{Y}$ given inputs $\mathbf{X}$.
Then, if we assume our model can generalize, we can use the same same model to predict what the outputs would look like for unseen inputs $\mathbf{x^*}$ (test inputs).
This is what's done in [maximum-likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation).

Gaussian processes are different because they produce estimates over functions instead of parameters.
That is, instead of estimating the most likely $\theta$ given the data(by finding $p(\theta|\mathbf{X}), \mathbf{Y}$), we estimate the most likely *function* given the data by finding $p(f|\mathbf{X}, \mathbf{Y})$.
To do so, a GP assumes the outputs are part of a joint Gaussian distribution such that
$$
    
$$

## Resources

- [Murphy's Probabilistic ML](https://probml.github.io/pml-book/)