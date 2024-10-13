---
title: Deriving the Loss of a Dynamical VAE
tags: [ml]
publishDate: 2024-10-13
draft: false
enableToc: true
---

[Dynamical variational autoencoders (DVAEs)](https://dynamicalvae.github.io/) are a generalization of [variational autoencoders (VAEs)](https://en.wikipedia.org/wiki/Variational_autoencoder) for the case where the data is correlated through a hidden dynamical system. 
DVAEs use neural networks to learn how to represent the evolution of a latent state (aka hidden state) which evolves as a function of inputs and time. 

In this post I go over the derivation of the loss function used to train a DVAE.
I'll start by deriving the loss for a regular VAE, introduce state-space models as a way of describing dynamical systems, and then derive the loss for a VAE.

--- 
## Variational Autoencoders (VAEs)

Given a set $\mathcal{D}$ of observations $\mathbf{x}$, e.g., images assume the following: 
- Each observation $\mathbf{x} \in \mathcal{D}$ is independently sampled from an unknown distribution $p(\mathbf{x})$.
- Each observation has a probabilistic dependence on an unobserved variable (a.k.a. latent or hidden variable) $\mathbf{z} \sim p(\mathbf{z} | \mathbf{x})$ that contains information about the underlying structure of $\mathbf{x}$. Therefore, we can also write $\mathbf{x} \sim p(\mathbf{x} | \mathbf{z})$.

We wish to learn an approximation of the underlying process that generated $\mathcal{D}$, i.e., $p_\theta(\mathbf{x}) \approx p(\mathbf{x})$.
Why? 
A probabilistic model of observed natural and artificial phenomena is useful for decision-making tasks. 
And in cases where the data's dimension is very high, having a model that captures its underlying structure in a lower-dimensional representation may simplify controller design.
For example, consider a dynamical system one wishes to control, such as a pendulum.
A controller that takes in images of the pendulum will be much more complex than one whose input is the pendulum's state (position and velocity). 
However, this requires a model that captures the dependency between the images and the state, e.g., $p(\mathbf{x}| \mathbf{z})$ and $p(\mathbf{z} |\mathbf{x})$.

One way of learning $p_\theta(\mathbf{x})$ is by finding parameters $\theta$ that maximize the likelihood of observing the data in $\mathcal{D}$. 
If the data is independent, this is equivalent to maximizing the log-likelihood
$$
    \log p_\theta (\mathcal{D}) = \sum_{\mathbf{x} \in \mathcal{D}} \log p_\theta (\mathbf{x}) \tag{1}.
$$
Unfortunately, directly maximizing Equation 1 is typically intractable. 
To see why consider the likelihood of a single datum $\mathbf{x} \in \mathcal{D}$:
$$
    p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}, \mathbf{z}) d\mathbf{z}
    \tag{2}
$$
Computing $p_\theta(\mathbf{x})$ requires marginalizing over the latent variable $\mathbf{z}$, which is usually an intractable integral.

## The Evidence Lower Bound (ELBO) Loss

Fortunately, we can get around this computation by instead maximizing the lower bound on the data likelihood. 
We begin by noting that if $p_\theta(\mathbf{x})$ is intractable, so is the posterior $p_\theta(\mathbf{z}|\mathbf{x})$ since $p_\theta(\mathbf{z}|\mathbf{x}) = \frac{p_\theta(\mathbf{x}, \mathbf{z})}{p_\theta(\mathbf{x})}$.
Therefore, let $q_\phi(\mathbf{z}|\mathbf{x}) \approx p(\mathbf{z}|\mathbf{x})$ be a model with parameters $\phi$ which approximates the posterior and captures the dependency between the latent variable and the observed data. 
Then, we can write
$$ 
\begin{align}
    \log p_\theta(\mathbf{x}) &= \log \int p_\theta(\mathbf{x}, \mathbf{z}) dz\\
    &= \log \int \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} | \mathbf{x})}q_\phi(\mathbf{z} | \mathbf{x}) d\mathbf{z}\\
    &= \log \mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})}\left[\frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} | \mathbf{x})}\right]\\
    &\geq \mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})}\left[\log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} | \mathbf{x})}\right] \text{(by Jensen's inequality)}\\
    &= \mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})}\left[\log p_\theta(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z} | \mathbf{x})\right]\\
    &= \mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})}\left[\log p_\theta(\mathbf{x}|\mathbf{z}) + \log p_\theta(\mathbf{z}) - \log q_\phi(\mathbf{z} | \mathbf{x})\right]\\
    &= \mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})}\left[\log p_\theta(\mathbf{x}|\mathbf{z})\right] - \mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})}\left[\log \frac{q_\phi(\mathbf{z} | \mathbf{x})}{p_\theta(\mathbf{z})}\right]\\
    &= \mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})}\left[\log p_\theta(\mathbf{x}|\mathbf{z})\right] - D_{KL}(q_\phi(\mathbf{z} | \mathbf{x}) || p_\theta(\mathbf{z})) \tag{3},
\end{align}
$$
where the second term in Equation 3 is the Kullback-Leibler (KL) divergence between the approximated posterior and the prior on $\mathbf{z}$.
Equation 3 is also known as the Evidence Lower Bound (ELBO). 

In the literature on VAEs, the posterior distribution $q_\phi(\mathbf{z}|\mathbf{x})$ is also known as the encoder, inference distribution, or recognition model. 
On the other hand, the conditioned likelihood $p_\theta(\mathbf{x}|\mathbf{z})$ is known as the decoder or generative model.

To understand the explicit relationship between the ELBO and $\log p_\theta(\mathbf{x})$ first note that $\log p_\theta(\mathbf{x})$ is constant with respect to $\mathbf{z}$. Then
$$
\begin{align}
    \log p_\theta(\mathbf{x}) &= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [\log p_\theta(\mathbf{x})]\\
    &= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[ \log\frac{p_\theta(\mathbf{x}, \mathbf{z})}{p_\theta(\mathbf{z} | \mathbf{x})}\right]\\
    &= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[ \log\frac{p_\theta(\mathbf{x}, \mathbf{z}) q_\phi (\mathbf{z}|\mathbf{x})}{q_\phi(\mathbf{z} | \mathbf{x}) p_\theta(\mathbf{z} | \mathbf{x})}\right]\\
    &= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[ \log\frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} | \mathbf{x})}\right] + \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[ \log\frac{q_\phi (\mathbf{z}|\mathbf{x})}{p_\theta(\mathbf{z} | \mathbf{x})}\right]\\
    &= \text{ELBO} + D_{KL}(q_\phi(\mathbf{z} | \mathbf{x}) || p_\theta(\mathbf{z} | \mathbf{x})) \tag{4}. 
\end{align}
$$
By re-arranging Equation 4, and noting that the second term is non-negative and only zero when the approximated posterior equals the true posterior, we can see that maximizing the ELBO implies maximizing log-likelihood and minimizing the KL divergence between the approximated and true posteriors. 
$$
    \text{ELBO} = \log p_\theta(\mathbf{x}) - D_{KL}(q_\phi(\mathbf{z} | \mathbf{x}) || p_\theta(\mathbf{z} | \mathbf{x}))
$$
Therefore, the ELBO is an appropriate objective over the intractable $\log p_\theta(\mathbf{x})$.

## State-Space Models (SSMs)

So far we've talked about how to represent an input in a lower-dimensional space. 
If we have a set of images, like in the MNIST case, we can learn to represent features in the digits in a lower dimensional space. 
This makes sense: they're all handwritten digits. 
But what if we know that the observations result from a dynamical system, such as a pendulum, some kind of flow field,  or even a videogame? 
In this case, it might be useful to bake in the observation's dynamic nature into the latent space we're trying to learn.
Therefore, we need to modify the vanilla VAE, since it's designed for uncorrelated data.

To get started, it is useful to review [state-space models (SSMs)](https://en.wikipedia.org/wiki/State-space_representation), a mathematical model of physical dynamical systems used in time-series analysis, control theory, signal processing, neuroscience, and many other fields. 
We focus on the discrete-time, continuous-valued SSMs defined by the following equations: 
$$
    \begin{align}
        (\boldsymbol{\mu}^\mathbf{z}_{t}, \boldsymbol{\sigma}^\mathbf{z}_{t}) &= f(\mathbf{z}_{t-1}, \mathbf{u}_{t-1}) \tag{5}\\
        p_{\theta_\mathbf{z}}(\mathbf{z}_t | \mathbf{z}_{t-1}, \mathbf{u}_{t-1}) &= \mathcal{N}(\mathbf{z}_t; \boldsymbol{\mu}^\mathbf{z}_{t}, \boldsymbol{\sigma}^\mathbf{z}_{t}) \tag{6}\\
        (\boldsymbol{\mu}^\mathbf{x}_{t}, \boldsymbol{\sigma}^\mathbf{x}_{t}) &= g(\mathbf{z}_{t}) \tag{7}\\
        p_{\theta_\mathbf{x}}(\mathbf{x}_t | \mathbf{z}_{t}) &= \mathcal{N}(\mathbf{x}_t; \boldsymbol{\mu}^\mathbf{x}_{t}, \boldsymbol{\sigma}^\mathbf{x}_{t}) \tag{8}
    \end{align}
$$
Where Equations 5 and 6 define state transition dynamics, and Equations 7 and 8 define the observation model.
The distributions in Equations 6 and 8 are parameterized by $\theta_\mathbf{z}$ and $\theta_\mathbf{x}$, respectively.

Regarding notation: subscripts index time, and superscripts denote the corresponding variable.
In the following derivation we assume access to $\mathbf{z}_0$ (we can sample this from an assumed prior distribution) and the complete sequence of control inputs $\mathbf{u}_{0:T-1}$, where $T$ is the control horizon.

## The ELBO for Dynamical VAEs

Now let's approximate the distribution that generated the entire data sequence, not just individual data points. 
This is because the data was generated from a dynamical system and is correlated over time, e.g., if the data $\mathcal{D}$ is a set of images of a pendulum in motion, the content of one image depends on the content of the previous image. This is not the case for a set of handwritten digits.
Formally, we wish to learn a distribution $p_\theta(\mathbf{x}_{1:T}|\mathbf{u}_{0:T-1})$ that approximates the distribution which generated $\mathbf{x}_{1:T}$ given the known control sequence $\mathbf{u}_{0:T-1}$, i.e., $p_\theta(\mathbf{x}_{1:T-1}) \approx p(\mathbf{x}_{1:T-1})$.

As one would expect, we run into the same issue as before: maximizing the likelihood $p_\theta(\mathbf{x}_{1:T}|\mathbf{u}_{0:T-1})$ requires marginalizing over the latent space which is an intractable operation:  
$$
    p_\theta(\mathbf{x}_{1:T}|\mathbf{u}_{0:T-1}) = \int p_\theta(\mathbf{x}_{1:T}, \mathbf{z}_{1:T}|\mathbf{u}_{0:T-1})d\mathbf{z}_{1:T}.
    \tag{9}
$$
Fortunately, we can get around this with the same approach as before: learn a conditioned likelihood $p_\theta(\mathbf{x}_{1:T}|\mathbf{z}_{1:T}, \mathbf{u}_{0:T-1})$ and an approximate posterior $q_\phi(\mathbf{z}_{1:T}|\mathbf{x}_{1:T}, \mathbf{u}_{0:T-1})$ that maximize a version of the ELBO adapted for the correlated data.

We derive this new ELBO by first noting that the dynamical system defined in Equation 9 is [causal](https://en.wikipedia.org/wiki/Causal_system) and [Markovian](https://en.wikipedia.org/wiki/Markov_model).
It is a causal system because the distributions for observations and latent variables only depend on their values at previous time steps. 
And it is a Markovian system because the transition dynamics only depend on the previous state.
Therefore, $\mathbf{z}_{t}$ only depends on $\mathbf{z}_{t-1}$ and $\mathbf{u}_{t-1}$, and the approximate posterior can be re-written as: 
$$
    q_\phi(\mathbf{z}_{1:T}|\mathbf{x}_{1:T}, \mathbf{u}_{0:T-1}) = \prod_{t=1}^T q_\phi(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{u}_{t-1}).
    \tag{10}
$$
We can also factorize the joint distribution of the observed and latent sequences as a product of conditional distributions for every timestep, i.e., 
$$
    p_\theta(\mathbf{x}_{1:T}, \mathbf{z}_{1:T}|\mathbf{u}_{0:T-1}) = \prod_{t=1}^{T}p_\theta(\mathbf{x}_t|\mathbf{z}_{t})q_\phi(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{u}_{t-1}).
    \tag{11}
$$

Next, we write the ELBO in terms of the observation, latent, and control sequences, and then substitute the factorizations in Equations 10 and 11:

$$
    \begin{align}
        \text{ELBO} &= \mathbb{E}_{q_\phi(\mathbf{z}_{1:T}|\mathbf{x}_{1:T},\mathbf{u}_{0:T-1})}\left[ \log\frac{p_\theta(\mathbf{x}_{1:T}, \mathbf{z}_{1:T}|\mathbf{u}_{0:T-1})}{q_\phi(\mathbf{z}_{1:T} | \mathbf{x}_{1:T}, \mathbf{u}_{0:T-1})}\right]\\
        &= \mathbb{E}_{q_\phi(\mathbf{z}_{1:T}|\mathbf{x}_{1:T},\mathbf{u}_{0:T-1})}\left[ \log \frac{\prod_{t=1}^{T}p_{\theta_\mathbf{x}}(\mathbf{x}_t|\mathbf{z}_{t})p_{\theta_\mathbf{z}}(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{u}_{t-1})}{\prod_{t=1}^T q_\phi(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{u}_{t-1})}\right]\\
        &= \mathbb{E}_{q_\phi(\mathbf{z}_{1:T}|\mathbf{x}_{1:T},\mathbf{u}_{0:T-1})}\left[ \sum_{t=1}^T \log \left(p_{\theta_\mathbf{x}}(\mathbf{x}_t|\mathbf{z}_{t})\right) p_{\theta_\mathbf{z}}(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{u}_{t-1}) \right. \notag\\ 
        & \hspace{6cm} \left. - \log q_\phi(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{u}_{t-1}) \vphantom{\sum_{t=1}^T} \right]\\
        &= \mathbb{E}_{q_\phi(\mathbf{z}_{1:T}|\mathbf{x}_{1:T},\mathbf{u}_{0:T-1})}\left[ \sum_{t=1}^T \log p_{\theta_\mathbf{x}}(\mathbf{x}_t|\mathbf{z}_{t}) - \log \frac{q_\phi(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{u}_{t-1})}{p_{\theta_\mathbf{z}}(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{u}_{t-1})}\right].
    \end{align}
$$
Now, if we let $\Psi(\mathbf{z}_{1:T}) = \sum_{t=1}^T \log p_{\theta_\mathbf{x}}(\mathbf{x}_t|\mathbf{z}_{t}) - \log \frac{q_\phi(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{u}_{t-1})}{p_{\theta_\mathbf{z}}(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{u}_{t-1})}$ and write the expectation in terms of the factorized posterior in Equation 10 we have
$$
    \begin{align}
        \text{ELBO} &= \int_{\mathbf{z}_1} \dots \int_{\mathbf{z}_T} \Psi(\mathbf{z}_{1:T}) \prod_{t=1}^T q_\phi(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{u}_{t-1})d\mathbf{z}_{T} \dots d\mathbf{z}_{1}\\
        &= \mathbb{E}_{q_\phi(\mathbf{z}_{1}|\mathbf{z}_{0},\mathbf{u}_{0})}\left[\mathbb{E}_{q_\phi(\mathbf{z}_{2}|\mathbf{z}_{1},\mathbf{u}_{1})}\left[\dots\mathbb{E}_{q_\phi(\mathbf{z}_{T}|\mathbf{z}_{T-1},\mathbf{u}_{T-1})}\left[\Psi(\mathbf{z}_{1:T})\right]\dots \right]\right]\\
        &=\sum_{t=1}^T\mathbb{E}_{q_\phi(\mathbf{z}_{t}|\mathbf{z}_{t-1},\mathbf{u}_{t-1})}\left[\log p_{\theta_\mathbf{x}}(\mathbf{x}_t|\mathbf{z}_{t})\right] \\ 
        &\quad \, \, -\sum_{t=2}^T\mathbb{E}_{q_\phi(\mathbf{z}_{t-1}|\mathbf{z}_{t-2}, \mathbf{u}_{t-2})}\left[D_{KL}(q_\phi(\mathbf{z}_t |\mathbf{z}_{t-1}, \mathbf{u}_{t-1}||p_{\theta_\mathbf{z}}(\mathbf{z}_t|\mathbf{z}_{t-1}, \mathbf{u}_{t-1}))\right].
    \end{align}
$$


## Resources

- [Original VAE paper](https://arxiv.org/abs/1312.6114) by Kingma et al.
- [Tutorial on VAEs](https://arxiv.org/abs/1906.02691) by Kingma and Welling.
- [A review of DVAEs](https://arxiv.org/abs/2008.12595) by Girin et al. Super useful and includes a much more general exposition of DVAEs. For example, for the case of non-Markovian and non-causal DVAEs. I followed much of their development for this post.
- [Danijar Hafner](https://danijar.com/)'s [super cool work on world models](https://danijar.com/project/dreamerv3/) is what originally took me down this rabbit hole. 