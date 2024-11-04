---
title: WIP Why is Dual Control Hard? 
publishDate: 2024-11-03
draft: False
tags: [ct]
---

The past 6 months I've been exploring how take an active role in reducing the uncertainty of a dynamical system. 
We applied this idea in a game-theoretic setting and wrote ["Smooth Information Gathering in Two-Player Noncooperative Games"](smooth.md).
After that was done, I did some digging into how I could apply this idea to the case of a dynamical system like a robot. 
I ran into the ideal of [dual control](https://en.wikipedia.org/wiki/Dual_control_theory). 
Unfortunately, most of the literature seems to agree that it's very hard.
In this blog post, I outline exactly why it's hard, and then explore what can be done about it. 

<figure style="text-align: center;">
  <img src="media/dual_hard.png" alt="" style="width:100%">
  <figcaption style="max-width: 95%; margin: auto;">
    <em>Footnote from <a href="https://meta-learn.github.io/2018/papers/metalearn2018_paper58.pdf" target="_blank">"Control Adaptation via Meta-Learning Dynamics"</a>.</em>
  </figcaption>
</figure>

---

For this development I'll be following Klenske et al. in [Dual Control for Approximate Bayesian Reinforcement Learning](https://arxiv.org/abs/1510.03591)

# Toy Problem 

Let state be defined as $\mathbf{x}$, controls $\mathbf{u}$, and process noise $\mathbf{\xi} \sim \mathcal{N}(0, \mathbf{Q})$.
Now, let's start with the simplest possible problem: we have a linear, scalar system with noiseless observations and we're trying to find a control sequence that minimizes a cost that's a function of state and controls.
This is a standard problem in optimal control and reinforcement learning. 
Let the cost be given by the quadratic function
$$
  \mathcal{L}(\mathbf{x}, \mathbf{u}) = \sum_{k=0}^T(\mathbf{x}_k - \mathbf{r}_k)^\top\mathbf{W}(\mathbf{x}_k - \mathbf{r}_k) + \sum_{k=0}^{T-1}(\mathbf{u}_k^\top \mathbf{U} \mathbf{u}_k),
$$
where $\mathbf{r}_k$ denotes a target trajectory and $\mathbf{W}$, $\mathbf{U}$ are matrices define the state and control costs. 

If the dynamical system is
$$
x_{k+1} = a x_{k} + b u_{k} + \mathbf{\xi}_k, \, a,b \in \mathbb{R}.
$$
Then, assuming $a$ and $b$ are known and $\mathbf{r}_k = \mathbf{0}$, the optimal control $\mathbf{u}_k^*$ is given by
$$
\mathbf{u}_k^* = \frac{ab\mathbf{x}_k}{U + b^2}.
$$
This can be easily found by substituting the dynamics into the expected cost (because we have process noise), computing the cost's derivative with respect to $\mathbf{u}_k$, setting it to zero, and solving for $\mathbf{u}_k$.

Now let's consider the case where we are uncertain about $b$.  
So now, we have a given by a belief $p(b) = \mathcal{N}(b|\mathbf{\mu}_k, \mathbf{\sigma}_k^2)$.
A naïve solution is to replace $b$ with our current estimate of the mean ($\mathbf{\mu}_k$).
This approach is known as *certainty equivalence*. 
Unfortunately, this does not account for the uncertainty in the belief, i.e., how big $\mathbf{\sigma}_k^2$ is and often leads to poor performance.

Alternatively, we could minimize the expected cost 
$\mathbb{E}_{b\sim\mathcal{N}(b|\mathbf{\mu}_k, \mathbf{\sigma}_k^2)}[\mathbf{x}_{k+1}^2 + U \mathbf{u}_k^2]$, resulting in the optimal control 
$$
\mathbf{u}_k^* = -\frac{a\mathbf{\mu}_k\mathbf{x}_k}{U + \mathbf{\mu}_k^2 + \mathbf{\sigma}_k^2}.
$$