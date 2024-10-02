---
title: Simple Gaussian Process 
tags: [ml]
publishDate: 2024-10-01
draft: false
---

Today I made a simple implementation of a [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process) (GP) and used it to approximate a [Van Der Pol oscillator](https://en.wikipedia.org/wiki/Van_der_Pol_oscillator). 
Here's a sample plot: 

<figure style="text-align: center;">
  <img src="media/gp_10p.jpg" alt="" style="width:65%">
  <figcaption style="max-width: 95%; margin: auto;"><em>The GP's predictions for position given 10 training points. The ribbon plot shows the predicted uncertainty.</em></figcaption>
</figure>

Shout out to Jie Wang and his [tutorial on GP regression](https://arxiv.org/abs/2009.10862v5), 
it had everything I needed to implement this.

Find the code [here](https://github.com/fernandopalafox/gp).



