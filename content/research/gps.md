---
title: (WIP) Gaussian Processes and Uncertainty in World Models
tags: [ml]
publishDate: 2024-10-20
draft: false
---

WIP

I've been looking into ways of incorporating [epistemic uncertainty](https://en.wikipedia.org/wiki/Uncertainty_quantification#Aleatoric_and_epistemic) into learned [world models](https://danijar.com/project/dreamerv3/).
David suggested I looking into using Gaussian processes (GPs) like James Harrison in his work on [Variational Bayesian Last Layers](https://arxiv.org/abs/2404.11599) or [ALPaCA](https://arxiv.org/abs/1807.08912). 
James' work looks super interesting, but once I started digging into it I realized I didn't really understand how GPs REALLY worked (even though I implemented one [here](research/gp.md) lol), so in this post I'll derive them.

---

# Gaussian Processes (GPs)

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

## Predicting from noise-free data

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


# Meta Learning with GPs

> [!info] Notation
> I don't have infinite time so in this section I'll mostly follow the notation James Harrison used in [Meta-Learning Priors for Efficient Online Bayesian Regression (ALPaCA)](https://arxiv.org/abs/1807.08912).
> Read everything carefully because there may be differences in notation and definitions from the previous sections.

## Formulation

Consider a function $f$ with unknown latent parameters $\theta$. 
Let's assume we can observe samples $y$ of $f$ corrupted by additive Gaussian noise. 
That is, we observe a sequence of samples (x, y) where $y = f(x;\theta) + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \Sigma_\epsilon)$.
Therefore, the likelihood of the data is given by
$$
    p(y|x, \theta) = \mathcal{N}(y|f(x;\theta), \Sigma_\epsilon).
$$

Given a prior over the latent parameters, i.e., $p(\theta)$ the posterior predictive density of $\tau$ data points $\mathcal{D}^*_\tau = \{(x_t,y_t)\}_{t=1}^\tau$ generated from $f(x;\theta^*)$ is given by 
$$
    p(y|x, \mathcal{D_\tau^*}) = \int p(y|x, \theta) p(\theta|\mathcal{D_\tau^*}) d\theta
$$
Unfortunately, this integral is intractable because we don't have analytic expressions for $f(x;\theta)$ and $p(\theta)$, and even if we did, computing $p(\theta|\mathcal{D_\tau^*})$ over all possible $\theta$ is likely too computationally expensive.

Instead, let's use a surrogate model $q_\xi(y|x,D_\tau^*)$, parameterized by $\xi$, to approximate the true posterior predictive density $p(y|x,D^*\tau)$, and then let's optimize this model so that it's as close to the true posterior predictive density **for all likely $\theta^*$**.
The bolded part is where the meta-learning is happening: we're learning a model that will work well for all possible $\theta^*$.

We consider a scenario in which the data comes in as a stream: at each timestep, the agent is given a new input $x_t$, and after estimating the output $\hat{y}_t$, the true output $y_t$ is revealed. 
An example of this is a Markovian dynamical system, where the agent wishes to predict the distributions of the next state given the current state.

In this setting, the problem of learning the surrogate model can be formulated as
$$
\min_\xi D_{KL}(p(y_{t+1}|x_{t+1}, \mathcal{D}_t^*) || q_\xi(y_{t+1}|x{t+1}, \mathcal{D}_t^*)). 
$$
Note that we don't know $t$ (dataset size), $x_{t+1}$, or $D^*_t$ ahead of time, so the best we can do is minimizethe objective in *expectation* (I'll elaborate on this later).
This implies that whatever $\xi$ we choose will be optimal for all possible datasets.
Very cool.

Unfortunately, this expected objective is intractable because need access to $p(\theta)$ and $p(D_t^*|\theta^*)$ which are unknown.
Instead, we assume we have access to various datasets $\mathcal{D}_t^j$ generated from iid samples of $\theta^j \sim p(\theta)$, $x_t \sim p(x)$, and $y_t \sim p(y|x, \theta^j)$.
Each dataset can be thought of as trajectories of the system generated by different latent parameters $\theta^j$.
The full dataset is defined as $\mathcal{D} := \{\mathcal{D}_t^j\}_{j=1}^M$.

## Bayesian Regression

ALPaCA uses Bayesian linear regression to compute $q_\xi(y|x, \mathcal{D}_t^*)$.
If we consider a set of basis functions $\phi$, the regression problem can be written as finding $K$ such that
$$
    y_t^\top = \phi^\top(x_t) K + \epsilon_t,
$$
where $K$ is a coefficient matrix and $\epsilon_t \sim \mathcal{N}(0, \Sigma_\epsilon)$.
Let $Y^\top = [y_1, \dots, y_\tau], \Phi^\top = [\phi(x_1), \dots, \phi(x_\tau)]$, and E = $[\epsilon_1, \dots, \epsilon_\tau]$, then we can re-write the regression problem as
$$
    Y = \Phi K + E.
$$
Therefore, the likelihood of the data is given by
$$
\begin{equation}
    p(Y|\Phi, K, \Sigma_\epsilon) = \mathcal{N}(Y|\Phi K, \Sigma_\epsilon).
\end{equation}
$$
Now let's select the prior for $K$ as $p(K) = \mathcal{MN}(K|\bar{K}_0, \Lambda_0^{-1}, \Sigma_\epsilon)$, where $\mathcal{MN}$ denotes the [matrix normal distribution](https://en.wikipedia.org/wiki/Matrix_normal_distribution), and $\Lambda_0$ is a precision matrix.
Then, the posterior distribution of $K$, conditioned on $Y$ and $\Phi$, is given by
$$
    \begin{align}
        p(K|Y, \Phi) &= \mathcal{MN}(K|\bar{K}_\tau, \Lambda_\tau^{-1}, \Sigma_\epsilon) \\
        \bar{\Lambda}_\tau &= \Phi^\top \Phi + \Lambda_0 \\
        \bar{K}_\tau &= \bar{\Lambda}_\tau^{-1} (\Phi^\top Y + \Lambda_0 \bar{K}_0).
    \end{align}
$$
The posterior distribution is then given by 
$$
    p(y_{\tau + 1} | \phi(x_{\tau + 1}), \Phi, Y) = \mathcal{N}(\bar{K}_\tau^\top \phi(x_{\tau + 1}), \Sigma_{\tau + 1})
$$
where
$$
    \Sigma_{\tau + 1} = (1 + \phi^\top(x_{\tau + 1}) \Lambda^{-1}_\tau \phi(x_{\tau + 1})) \Sigma_\epsilon.
$$
Now we have a posterior over $y_{\tau + 1}$ given the value of the basis function at $x_{\tau + 1}$, i.e., $\phi(x_{\tau + 1})$, the value of the basis functions for all previous data points, i.e., $\Phi$, and the observed outputs, i.e., $Y$.
The paper goes over details on how this was computed.

## ALPaCA

In ALPaCA, the basis functions are outputs of a neural network and we do a Bayesian regression on a linear transformation applied to the final output of the network.
Then, we have two phases: 
- Phase 1 (offline): learn the basis functions (the neural network weights $w$) and the prior parameters $K_0$ and $\Lambda_0$ using a sample of datasets. 
- Phase 2 (online): Update the posterior parameters $\bar{K}_\tau$ and $\Lambda_\tau$ as new data comes in.
This approach allows for fast adaptation to new tasks without having to retrain the neural network.
Additionally, we also get live, calibrated uncertainty estimates for the predictions.
Below are the algorithms for the two phases, taken from the paper.

<figure style="text-align: center;">
  <img src="media/gps_phase_1.png" alt="" style="width:65%">
</figure>

<figure style="text-align: center;">
  <img src="media/gps_phase_2.png" alt="" style="width:65%">
</figure>

## Questions
- Connection between ALPaCA and GPs?

# Resources

- Murphy's [Probabilistic ML](https://probml.github.io/pml-book/)
- Rasmussen's [Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/)

[^1]: For the development of GPs, we will refer to functions as a finite-dimensional vector of function values at a set of input points. Technically, functions are infinitely-dimensional objects that require an infinite number of (input,output) pairs to be fully described (unless we have an explicit functional form for a function, e.g., $f(x) = x^2$). However, when working with GPs we are able to describe functions with a finite number of points because we assume that function values at different points are **correlated**. This correlation is defined according to a selected kernel function. However, the underlying function is still infinite-dimensional. Disclaimer: I'm still trying to wrap my head around these details. But I think this is the gist of it.