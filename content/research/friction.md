---
title: "Correcting Robot Joint Friction w/ Differentiable Simulators"
tags: [wh, ml]
publishDate: 2024-09-15
draft: false
---

Here I present method, based on differentiable simulators, to compensate for joint friction (or other unmodelled forces).
This post is only a proof of concept where I used Google's [Brax](https://github.com/google/brax), a differentiable physics simulator based on [Jax](https://github.com/google/jax), to correct for unmodelled joint friction in a simple pendulum.

**Table of Contents**
- [Problem](#problem)
- [Challenges](#challenges)
- [Approach](#approach)
- [Data collection](#data-collection)
- [Training the neural network](#training-the-neural-network)
- [Results](#results)
  - [Before](#before)
  - [After](#after)
- [Conclusion](#conclusion)
- [Code](#code)

# Problem 
[Force controllers](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-5-force-control/) are helpful in situations when you want your robot to be compliant.
For example, when you're interested in contact-rich tasks or you are operating around humans.
Unfortunately, they're very sensitive to unmodelled forces, such as joint friction. 
The symptoms of the problem are that trajectories are trajectory tracking performance is degraded. 
This is because there's no way for the controller to tell whether it's running into a wall (something it should be compliant to) or joint friction.

# Challenges
This problem is challenging because joint friction depends on a ton of variables such as: 
- Temperature
- Angles
- Velocity
- Loads
- Joint degradation
One way of solving this problem is to come up with a high-fidelity simulator of the robot you're using. 
This approach works kinda well, but it's typically bespoke and limited to a specific robot.
We want a more general method.

# Approach
Three steps
1. Run a real robot and a simulated robot with no joint friction.
2. Send the same commands to both robots and check the differences in states
3. Train a neural network to output a torque that minimizes the difference between the states 

Here's a diagram to get some intuition for how this works:
<figure style="text-align: center;">
  <img src="media/diffsim_plan.png" alt="" style="width:65%">
</figure>

In this blog post I'll present a proof of concept where both simulators are in simulation, but one of them has an added friction force I came up with.

# Data collection 

<figure style="text-align: center;">
  <img src="media/diffsim_data.png" alt="" style="width:65%">
</figure>

We collect data with the a simulator with an added friction force meant to represent a real robot with friction. 
We give the robot a random torque command, add friction to each of the joints, and then save the initial state, the final state, and the applied control torque. 
For this proof of concept, the robot is a single pendulum.

# Training the neural network 

Once data is collected, I trained a neural network to correct for the data. 
I used a multi-layer perceptron with 4 hidden layers and 256 neurons each.  
The input layer of the neural network takes in normalized joint positions and velocities from the initial state. [^1]
Training is done as follows: 
1. Sample a tuple from the dataset. Each data tuple contains an initial state, control torque, and a final state.  
2. Using the initial state, initialize a robot in a simulator (with no friction). 
3. Apply the control torque and the neural network torque to the robot. 
4. Compute the loss function by comparing the difference between the final state (from the data) and the resulting state in the simulation

Reducing loss by adjusting the network parameters essentially means that the networks is learning to imitate the effects of friction. 
So, once the network is trained, if we apply the *negative* of the learned torque, we should counteract the effects of friction. 

In the diagram below, I also note why it's important that we use a differentiable simulator. 
In order to propagate loss into changes of the neural network parameters we need to be able to differentiate the simulated state with respect to the neural network parameters. 
This can only be done with a differentiable simulator, as noted in the diagram below. 
 
<figure style="text-align: center;">
  <img src="media/diffsim_train.png" alt="" style="width:65%">
</figure>

# Results 

## Before

Here's a plot showing position error for the end effectors.
I use a simple PD controller since the dynamical system is very simple.
Using the same controller, the orange and blue lines show trajectories with and without friction, respectively.
As expected, the controller without friction has less error.

<figure style="text-align: center;">
  <img src="media/diffsim_before.png" alt="" style="width:65%">
</figure>

## After 

Below are the trajectories I got when combining the PD controller and the neural network. 
Green denotes the corrected trajectory. 
Notice how it does such a good job that it almost exactly cancels out the effects of friction, leaving us with a trajectory that almost perfectly overlaps the no-friction trajectory (blue line).

<figure style="text-align: center;">
  <img src="media/diffsim_after.png" alt="" style="width:65%">
</figure>

# Conclusion

This thing looks promising but I still need to figure out the following: 

**Coverage.** A pendulum is a very simple dynamical system, so covering the sample space was very straightforward. 
I just sampled random initial positions for the pendulum and random inputs, and since the robot just has a single joint, I quickly covered the entire state space.
However, getting good coverage of the state space will be way harder for robots with more joints.  

**Architecture.** As the dynamical system gets more complex, we will likely need a more complex model to better.

**Sim2real.** It remains to be shown whether this will transfer to a real robot. From my research I know that friction depends on temperature, and temperature changes with time of operation. 
This time-dependence may interfere with the data collection and/or learning process. 

**Safety.** Adding an extra torque that's "blind" to contact may violate safety limitations and remove some of the benefits of force control.

# Code
[Link to the GitHub repo](https://github.com/CLeARoboticsLab/friction-estimator). Take a look at the scripts and/or contact me if you have any questions. 

[^1]: Normalization is done according using averages and standard deviations from the collected dataset. 

