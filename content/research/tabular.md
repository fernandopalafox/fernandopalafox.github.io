---
title: Simple Implementation of Tabular Q-Learning 
tags: [ml]
publishDate: 2024-10-30
draft: false
---

Today I made a very simple implementation of Q-learning algorithm that learns a tabular function using first-visit, constant alpha updates, and Monte Carlo rollouts based on an epsilon-greedy policy. 
I applied this to the [FrozenLake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) environment.
Here's a plot showing the environment the optimal policy and the full Q table. 

<figure style="text-align: center;">
  <img src="media/tabular_Q8.png" alt="" style="width:100%">
  <figcaption style="max-width: 95%; margin: auto;">
</figure>

Find the code [here](https://github.com/fernandopalafox/discrete-rl). 