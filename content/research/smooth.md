---
title: Smooth Information Gathering in Two-Player Noncooperative Games
tags: [gt, ct]
publishDate: 2024-04-31
draft: false
---

In this [paper](https://arxiv.org/abs/2404.00733) we define a mathematical framework for modeling two-player noncooperative games in which one player is uncertain of the other player’s costs but can preemptively allocate information-gathering resources to reduce this uncertainty. We also provide a gradient-based algorithm to solve a two-stage game, and identify conditions under which the gradient of the first stage's cost wrt the information-gathering resources, is well-defined.
Links: [paper](https://arxiv.org/abs/2404.00733), [code](https://github.com/CLeARoboticsLab/GamesVoI.jl/). [^1]

<figure style="text-align: center;">
  <img src="media/smooth_pic.png" alt="" style="width:85%">
  <figcaption style="max-width: 95%; margin: auto;"><em>Framework overview.</em></figcaption>
</figure>

<figure style="text-align: center;">
  <img src="media/smooth_alg.png" alt="" style="width:65%">
  <figcaption style="max-width: 95%; margin: auto;"><em>Proposed algorithm</em></figcaption>
</figure>

<figure style="text-align: center;">
  <img src="media/smooth_derivative.png" alt="" style="width:65%">
  <figcaption style="max-width: 95%; margin: auto;"><em>Conditions for existence of derivative of game's solution wrt. decision variables</em></figcaption>
</figure>

Links: [paper](https://arxiv.org/abs/2404.00733), [code](https://github.com/CLeARoboticsLab/GamesVoI.jl/).

[^1]: Arxiv paper is outdated but I'll upload the most recent version soon!