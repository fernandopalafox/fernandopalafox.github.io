---
title: Why is Dual Control Hard? 
publishDate: 2024-11-06
draft: False
tags: [ct]
---

<figure style="text-align: center;">
  <img src="media/dual_hard.png" alt="" style="width:100%">
  <figcaption style="max-width: 95%; margin: auto;">
    <em>Footnote from <a href="https://meta-learn.github.io/2018/papers/metalearn2018_paper58.pdf" target="_blank">"Control Adaptation via Meta-Learning Dynamics"</a>.</em>
  </figcaption>
</figure>

Recently I've been exploring how to take actions that actively reduce uncertainty. 
In  ["Smooth Information Gathering in Two-Player Noncooperative Games"](smooth.md) we explored this idea in a game-theoretic setting. 
And [over the summer](writing/summer.md), I worked on applying this idea to autonomous vehicles.

As I dug into the theoretical foundations of active uncertainty reduction in dynamical systems (like robots),  
I ran into the field of [dual control](https://en.wikipedia.org/wiki/Dual_control_theory). 
Unfortunately, most of the literature seems to agree that it's very hard.
In this blog post I explain exactly why it's hard, and then write about what this means for my research. 

---
# The Intractability of Dual Control

## The Dynamical System

We seek to solve the following problem: drive a dynamical system to a state of interest while minimizing cost.
Let's start by defining state as $\mathbf{x}$ and controls $\mathbf{u}$.
As is standard in optimal control and reinforcement learning, we define a cost given by the quadratic function
$$
  \mathcal{L}(\mathbf{x}, \mathbf{u}) = \sum_{k=0}^T(\mathbf{x}_k - \mathbf{r}_k)^\top\mathbf{W}(\mathbf{x}_k - \mathbf{r}_k) + \sum_{k=0}^{T-1}(\mathbf{u}_k^\top \mathbf{U} \mathbf{u}_k),
$$
where $\mathbf{r}_k$ denotes a target or reference trajectory, and $\mathbf{W}$, $\mathbf{U}$ are matrices that define the state and control costs. 

Let's start with the simplest possible dynamical system: a linear, scalar system with noiseless observations given by
$$
\mathbf{x}_{k+1} = \mathbf{x}_{k} + b \mathbf{u}_{k} + \mathbf{\xi}_k, \, a,b \in \mathbb{R}, 
$$
where $\mathbf{\xi} \sim \mathcal{N}(0, \mathbf{Q})$ is process noise.
Assuming $a$ and $b$ are known, and $\mathbf{r}_k = \mathbf{0}$, the optimal control $\mathbf{u}_k^*$ that drives $\mathbf{x}$ to zero **in one step** is given by
$$
\mathbf{u}_k^* = \frac{b\mathbf{x}_k}{\mathbf{U} + b^2}.
$$
This can be easily found by substituting the dynamics into the expected cost $\mathbb{E}_{\xi_k}[\mathbf{x}_{k+1}^2 + \mathbf{U} \mathbf{u}_k^2]$[^1], computing the cost's derivative with respect to $\mathbf{u}_k$, setting it to zero, and solving for $\mathbf{u}_k$.

## Adding Uncertainty

Now let's consider the case where we are uncertain about $b$.
Given a belief $b_k \coloneqq \mathcal{N}(b|\mathbf{\mu}_k, \mathbf{\sigma}_k^2)$ at time $k$, a naïve solution is to replace $b$ with our current estimate of its mean $\mathbf{\mu}_k$.
This approach is known as *certainty equivalence*. 
Unfortunately, this does not account for the uncertainty in the belief, i.e., how big $\mathbf{\sigma}_k^2$ is and often leads to poor performance.

Alternatively, we could minimize the expected cost 
$\mathbb{E}_{b}[\mathbf{x}_{k+1}^2 + \mathbf{U} \mathbf{u}_k^2|b_k]$. 
This results in the optimal control 
$$
\mathbf{u}_k^* = -\frac{\mathbf{\mu}_k\mathbf{x}_k}{\mathbf{U} + \mathbf{\mu}_k^2 + \mathbf{\sigma}_k^2}.
$$
This is also known as "cautious" control, since it's inversely proportional to the uncertainty given by $\mathbf{\sigma}^2_k$.
Although this approach accounts for uncertainty, it can prevent learning (a future reduction in the uncertainty of $b$) if the uncertainty is too high. 

To see how this happens, we compute the posterior on $b$ after observing $\mathbf{x}_{k+1}$, i.e., $b_{k+1} \coloneqq p(b|\mathbf{x}_{k+1})$.
We do so by invoking Bayes' rule, noting that $b_{k+1} \propto p(\mathbf{x}_{k+1}|\mathbf{x}_k,\mathbf{u}_{k}, b)b_k$, and then doing a bunch of algebra. 
The result is the following Gaussian
$$
b_{k+1} = \mathcal{N}(b|\mathbf{\mu}_{k+1}, \mathbf{\sigma}_{k+1}^2) = \mathcal{N}\left(b\middle|\frac{\mathbf{\sigma}_k^2\mathbf{u}_k(b\mathbf{u}_k + \mathbf{\xi}_k + \mathbf{\mu}_k \mathbf{Q})}{\mathbf{u}_k^2\mathbf{\sigma}_k^2 + \mathbf{Q}}, \frac{\mathbf{\sigma}_k^2\mathbf{Q}}{\mathbf{u}_k^2\mathbf{\sigma}_k^2 + \mathbf{Q}}\right). 
$$
Notice that if $\mathbf{\sigma}_k$ is large then $\mathbf{u}_k^* \rightarrow 0$ implying $\mathbf{\sigma}_{k+1}^2 \rightarrow \mathbf{\sigma}_{k}^2$, and no learning happens. 
Also note how even for a simple scalar, linear system, the posterior update is already pretty gnarly.

The fact that the controller doesn't really account for how uncertainty will be reduced at future timesteps should not come as a surprise: we selected a controller that minimizes cost for a **single** step. 
This is known as a myopic controller because it doesn't consider the consequences of its actions in horizons longer than $T = 1$.

## Dual Control

Dual control was pioneered by [Alexander Feldbaum in the 60's](https://www.sciencedirect.com/science/article/pii/S1474667017696873). 
A controller in an uncertain system is said to have a dual control effect if it takes actions to 1) simultaneously reduce the uncertainty and 2) drive the system towards the reference or goal trajectory.

As I described in the previous section, if we want a controller to have this effect, we must consider control over trajectories of horizon $T\geq2$.
So now, let's revisit the cost function for longer trajectories. 
At every timestep $k<T$ we can define the expected cost of the remaining trajectory in the following recursive manner
$$
J_k(\mathbf{u}_{k:T-1}, p(\mathbf{x}_k)) = \mathbb{E}_{x_k}\left[(\mathbf{x}_k - \mathbf{r}_k)^\top\mathbf{W}(\mathbf{x}_k - \mathbf{r}_k) + \mathbf{u}_k^\top \mathbf{U} \mathbf{u}_k + J_{k+1}(\mathbf{u}_{k+1:T-1}, p(\mathbf{x}_{k+1})) | p(\mathbf{x}_k)\right], 
$$
where the final equation is defined by 
$$
J_T(p(x_T)) = \mathbb{E}_{\mathbf{x}_T}\left[(\mathbf{x}_T - \mathbf{r}_T)^\top\mathbf{W}(\mathbf{x}_T - \mathbf{r}_T)|p(\mathbf{x}_T)\right].
$$
The optimal control sequence can be found by solving the following minimization with dynamic programming:
$$
J^*_k(p(\mathbf{x}_k)) = \min_{\mathbf{u}_k}\mathbb{E}_{x_k}\left[(\mathbf{x}_k - \mathbf{r}_k)^\top\mathbf{W}(\mathbf{x}_k - \mathbf{r}_k) + \mathbf{u}_k^\top \mathbf{U} \mathbf{u}_k + J_{k+1}^*(p(\mathbf{x}_{k+1})) | p(\mathbf{x}_k)\right]
$$
For more details on how to solve problems of this form, see Section 8.3.1 [here](https://underactuated.mit.edu/lqr.html).

If we substitute our dynamical system for a horizon of $T=2$ then $J^*(\mathbf{x}_0)$ is the nested expectation
$$
J^*(p(\mathbf{x}_0)) = \min_{\mathbf{u_0}}\mathbb{E}_{\mathbf{x_0}}\left[\mathbf{W}\mathbf{x}_0^2 + \mathbf{U}\mathbf{u}_0^2 + \min_{\mathbf{u_1}}\mathbb{E}_{\mathbf{x_1}}\left[\mathbf{W}\mathbf{x}_1^2 + \mathbf{U}\mathbf{u}_1^2 + \mathbb{E}_{\mathbf{x}_2}\left[\mathbf{W}\mathbf{x}_2^2\right]\right]\right].
$$
Substituting $\mathbf{u}_1$ with the myopic controller $\mathbf{u}_1^*$ and writing the expectations a bit more explicitly we get
$$
  J^*(p(\mathbf{x}_0)) = \min_{\mathbf{u}_0}\left[\mathbf{W}\mathbf{x_0}^2 + \mathbf{U}\mathbf{u}_0^2 + \mathbb{E}_{\xi_0,b}\left[\mathbf{W}\mathbf{x}_1^2 + \mathbf{U}\mathbf{u}_1^{*2} + \mathbb{E}_{\xi_1, b}\left[\mathbf{W}(\mathbf{x}_1 + b\mathbf{u}_1^* + \xi_1)^2|b_1\right]|b_0\right]\right].
$$

The solution to this problem has a dual control effect because solving for $u_0$ requires minimizing a cost that drives the system towards the reference trajectory while accounting for changes in future beliefs (expressed through the nested expectations).

## Intractability

The nested expectations already hint that solving for a control sequence with a dual effect may be tricky, but let's understand exactly why this is hard.
Let's consider the innermost expectation: 
$$
  \mathbb{E}_{\xi_1, b}\left[W(\mathbf{x}_1 + b\mathbf{u}_1^* + \xi_1)^2|b_1\right] = \mathbb{E}_{\xi_1, b}[\Phi(\xi_1,b)]
$$
For brevity, let 
$$
  \Phi(\xi_1,b) = \mathbf{W}(\mathbf{x}_1 + b\mathbf{u}_1^* + \xi_1)^2 = \mathbf{W}(\mathbf{x}_0 + b\mathbf{u}_0 + \xi_0 + b\mathbf{u}_1^* + \xi_1)^2
$$
This is a joint expectation over two independent Gaussian distributions $\mathcal{N}(0, \mathbf{Q})$ and $b_1$. 
And even though $b_1$ has nasty expressions for the mean and variance (as shown by the posterior update above) it's still a Gaussian, and depending on the argument of the expectation $\Phi(\xi_1,b)$, computing it should be relatively easy.

Unfortunately, $\Phi(\xi_1,b)$ is indeed very nasty.
Recall that $\mathbf{u}_{k+1}^*$ is a nonlinear function of the belief parameters at time $k+1$, i.e., $\mu_{k+1}$ and $\sigma_{k+1}$. 
However, as shown in the belief update above, both parameters are functions (also nonlinear) of the belief parameters in the previous timestep, i.e., $\mu_{k}$ and $\sigma_{k}$.
This means that $\mathbf{u}_{1}^*$ is a *very* nonlinear function of $\mu_{0}$ and $\sigma_{0}$ with a form that implies $\mathbb{E}_{\xi_1, b}[\Phi(\xi_1,b)]$ is analytically intractable[^2].
Therefore, the best we can do is to numerically approximate the expectation, e.g., by sampling.
This approach works, but it scales poorly to higher dimensions, and it's still an approximation.

In conclusion, dual control is hard because even in simple dynamical systems, the belief dynamics are nonlinear (courtesy of Bayes' rule) resulting in intractable nested expectations. 

# Research
I just described dual control in the context of a dynamical system where uncertainty is represented as a Gaussian belief $b_k \coloneqq \mathcal{N}(b|\mathbf{\mu}_k, \mathbf{\sigma}_k^2)$ over an unknown parameter $b$ in a linear system.
In this case the transition function is $p(\mathbf{x}_{k+1}|\mathbf{x}_k,\mathbf{u}_{k}, b) = \mathcal{N}(\mathbf{x}_{k+1}|\mathbf{x}_k + b\mathbf{u}_k, \mathbf{Q})$
However, as I explained in my previous post, [Gaussian Processes and Meta-Learning Priors](gps.md), I'm interested in leveraging more expressive representations of dynamical systems such as neural networks trained to imitate lots of example trajectories, i.e., the transition function is $p(\mathbf{x_{k+1}}|\mathbf{x}_k,\mathbf{u}_k) = f(\mathbf{x}_k,\mathbf{u}_k, \theta)$, where $f$ is a deep neural network parameterized by weights and biases $\theta$.

Let's talk about how to represent uncertainty when the transition function is a neural network.
In the linear dynamics model we have uncertainty over process noise $\xi_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$ and model parameter values $b\sim b_k$. 
Practically speaking, this means that when minimizing a cost we must compute its joint expectation over $\xi_k$ and $b$, i.e., $\mathbb{E}_{\xi_k, b_k}[\cdot]$.
On the other hand, when using a neural network to represent dynamics, there are many ways to reason about the uncertainty of the next step $\mathbf{x}_{k+1}$ (e.g., [neural network ensembles](https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf), [Monte Carlo Dropout](https://medium.com/@ciaranbench/monte-carlo-dropout-a-practical-guide-4b4dc18014b5), [conformal mapping](https://people.eecs.berkeley.edu/~angelopoulos/publications/downloads/gentle_intro_conformal_dfuq.pdf), etc.), and also the model parameters $\theta$ (e.g., [Bayesian neural networks](https://www.cs.toronto.edu/~duvenaud/distill_bayes_net/public/)) 

Bayesian neural networks are probably the closest to what we're trying to do since they define a posterior distribution over model parameters, i.e., instead of a point estimate for the model weights, we get a distribution conditioned on the training data.
Unfortunately, as far as I understand, Bayesian neural networks are hard to scale, and adding dual control, which, as I illustrated here, is already very hard, is unlikely to make the problem easier. 

Therefore, if I want to combine dual control with expressive dynamics models such as neural networks, I'm going to have to reason about how to get the best of both worlds: expressiveness and tractability of weight updates.
ALPaCA kinda does this: it's like a Bayesian neural network with uncertainty only over the last, linear layer. 
It also uses a recursive Bayesian update based on least squares that's supposed to be more efficient.
Now the question is how this translates to a dual control setting:
How can such an update translate to a dual control setting?
Are we subject to the same pitfalls? 

Another direction I'll be exploring is the connection between the expected weight updates in belief-space planning ([see this paper by Platt et al](https://groups.csail.mit.edu/robotics-center/public_papers/Platt10.pdf)) and those in the ALPaCA.
[David](https://www.ae.utexas.edu/people/faculty/faculty-directory/fridovich-keil) had a hunch that there might be an equivalence, or at the very least an interesting connection, between the ALPaCA weight updates and the EKF parameter updates in Platt's paper.

Thank you for reading :)

# Resources
- Much of this development followed Klenske et al. in [Dual Control for Approximate Bayesian Reinforcement Learning](https://arxiv.org/abs/1510.03591).
- Feldbaum's pioneering work on [Dual Control Theory Problems](https://www.sciencedirect.com/science/article/pii/S1474667017696873).

[^1]: You'll notice that this cost looks different from the one I defined before. This is because we're driving the cost to zero in a **single** step, so the expression simplifies to $\mathcal{L}(\mathbf{x}, \mathbf{u}) = \mathbf{x}_{k+1}^2 + U \mathbf{u}_k^2$. 
[^2]: It turns out that  $\Phi(\xi_1,b)$ can be written as a rational function $\Phi(\xi_1,b) = \frac{Q_1(\xi_1,b)}{Q_1(\xi_1,b)}$, where $Q_1$ and $Q_2$ are polynomials. This means that the expected value $\mathbb{E}_{\xi_1, b}[\Phi(\xi_1,b)]$ will most likely not have a closed form solution. I'm aware that there are some cases where it is possible to compute the expectation of such rational functions, but as far as I'm aware, the results are limited to very specific polynomials. [Here's an example](https://math.stackexchange.com/questions/1394268/multivariate-gaussian-integral-of-ratio-of-quadratic-forms). 
[^4]: Online "belief" updates of the entire neural network are what's done by Finn et al. in [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400).