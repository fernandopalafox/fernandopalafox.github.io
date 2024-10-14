---
title: "Parallelizing Stochastic Games with ADMM"
tags: [paper, gt]
publishDate: 2023-04-04
draft: false
---

<figure style="text-align: center;">
  <img src="media/admm.jpg" alt="" style="width:75%">
</figure>

[Here's a paper](https://arxiv.org/abs/2304.01945) where we used [ADMM](https://stanford.edu/~boyd/admm.html) to parallelize (and greatly speed up) the computation of solutions for a class of [stochastic games](https://en.wikipedia.org/wiki/Stochastic_game).

This work was done in collaboration with [Jingqi Li](https://sites.google.com/view/jingqi-li/), [Frank Chiu](https://chihyuanchiu.github.io/), [Somayeh Sojoudi](https://www2.eecs.berkeley.edu/Faculty/Homepages/sojoudi.html)1, [Claire Tomlin](https://people.eecs.berkeley.edu/~tomlin/) at UC Berkeley, [Lasse Peters](https://lasse-peters.net/) and [Javier Alonso-Mora](https://autonomousrobots.nl/people/) at TU Delft, and my colleague [Mustafa Karabag](https://scholar.google.com/citations?user=PbKuWIwAAAAJ&hl=en) and advisor [David Fridovich-Keil](https://www.ae.utexas.edu/people/faculty/faculty-directory/fridovich-keil) at UT Austin.

We presented this work at the [62nd IEEE Conference on Decision and Control (CDC 2023)](https://cdc2023.ieeecss.org/index.html) in Singapore.

Find the code [here](https://github.com/CLeARoboticsLab/ScenarioControl.jl).