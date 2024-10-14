---
title: Learning Hyperplanes for Collision Avoidance in Space 
tags: [paper, gt, ct]
publishDate: 2023-11-25
draft: false
---
 
Collision-free trajectory generation for non-cooperative multi-agent systems using rotating hyperplanes constraints learned from expert trajectories by solving an inverse game. [[Paper]](https://arxiv.org/abs/2311.09439) [[Code]](https://github.com/CLeARoboticsLab/InverseHyperplanes.jl)

An example using the Hill-Clohessy-Wiltshire equations for relative orbital motion:

<table>
  <tr>
    <td style="height: 10px;">1. Noisy expert data</td>
    <td style="height: 10px;">2. Inferred hyperplanes</td>
    <td style="height: 10px;">3. Collision-free trajectory</td>
  </tr>
  <tr>
    <td valign="top"><img src="media/noisy.gif"  height="150"></td>
    <td valign="top"><img src="media/hyperplanes.gif" height="150"></td>
    <td valign="top"><img src="media/3D.gif"      height="150"></td>
  </tr>
 </table>

## Abstract 
A core challenge of multi-robot interactions is collision avoidance among robots with potentially conflicting objectives. We propose a game-theoretic method for collision avoidance based on rotating hyperplane constraints. These constraints ensure collision avoidance by defining separating hyperplanes that rotate around a keep-out zone centered on certain robots. Since it is challenging to select the parameters that define a hyperplane without introducing infeasibilities, we propose to learn them from an expert trajectory i.e., one collected by recording human operators. To do so, we solve for the parameters whose corresponding equilibrium trajectory best matches the expert trajectory.

You can read the full paper [here](https://arxiv.org/abs/2311.09439).