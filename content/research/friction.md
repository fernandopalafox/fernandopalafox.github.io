---
title: "Correcting Robot Joint Friction w/ Differentiable Simulators"
tags: [wh, ml]
publishDate: 2024-09-15
draft: false
---

How to correct for unmodelled dynamics in your robot by taking using differentiable simulators. 
This post is a proof of concept where I'll show you how I used Google's [Brax](https://github.com/google/brax), a differentiable physics simulator based on [Jax](https://github.com/google/jax), to correct for unmodelled joint friction in a simple pendulum 

**Table of Contents**
- [Problem](#problem)
- [Challenges](#challenges)
- [Approach](#approach)
- [Data collection](#data-collection)
- [Training the neural network](#training-the-neural-network)

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
The input layer of the neural network takes in normalized joint positions and velocities. [^1]
Training is done as follows: 
1. Sample a from the dataset. Each data tuple contains an initial state, control torque, and a final state.  
2. Using the initial state, initialize a robot in a simulator (with no friction). 
3. Apply the control torque and the neural network torque to the robot. 
4. Compute the loss function by comparing the difference between the final state (from the data) and the resulting state in the simulation

Reducing loss by adjusting the network parameters essentially means that the networks is learning to imitate the effects of friction. 
So, once the network is trained, if we apply the *negative* of the learned torque, we should counteract the effects of friction. 

In the diagram below, I also note why it's important that we use a differentiable simulator. 
In order to backpropagate from loss to neural network, we need to be able to differentiate the simulated state with respect to the neural network parameters. 
This can only be done with a differentiable simulator, and I note this in the diagram below. 
 
<figure style="text-align: center;">
  <img src="media/diffsim_train.png" alt="" style="width:65%">
</figure>

[^1] Normalization is done according to averages and standard deviations from states in the collected dataset. 
