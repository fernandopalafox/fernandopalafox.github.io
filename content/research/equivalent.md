---
title: WIP Belief Space Planning and Deep Dynamics Uncertainty
publishDate: 2024-11-18
draft: False
tags: [ct, ml]
---

In belief space planning we typically have uncertainty about the state.
But what if we want to think about dynamics uncertainty too?
As I've explored in previous posts (LINK THOSE HERE), I'm interested in ways of making minimal assumptions about the dynamics while still being able to quantify its uncertainty.
In the case of ALPaCA, we talk about uncertainty over the last layer of neural network trained to imitate the state. 
In this post I'll explore how to combine belief space planning with dynamics uncertainty. 
In particular, I'll show the following: 
- ALPaCA-like updating is equivalent to an EKF update
- Therefore the same update can be plugged into the belief space algorithm (LINK)

