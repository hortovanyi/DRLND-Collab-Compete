# DRLND Collaboration Compete Project
---
This is the thrid project in the Udacity Deep Reinforcement Learning Nanodegree. 

This project works with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

[![Trained Agent](https://github.com/hortovanyi/DRLND-Collab-Compete/blob/master/output/tennis.gif?raw=true)](https://www.youtube.com/watch?v=DH_ArATXSnE)

## Project Details
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

## Getting Started
It is recommended to follow the Udacity DRL ND dependencies [instructions here](https://github.com/udacity/deep-reinforcement-learning#dependencies) 

This project utilises [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md), [NumPy](http://www.numpy.org/) and [PyTorch](https://pytorch.org/) 

A prebuilt simulator is required in be installed. You need only select the environment that matches your operating system:

### Tennis Unity Environment 
Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)


The file needs to placed in the root directory of the repository and unzipped.

Next, before starting the environment utilising the corresponding prebuilt app from Udacity  **Before running the code cell in the notebook**, change the `file_name` parameter to match the location of the Unity environment that you downloaded.

## Instructions
Then run the [`Tennis.ipynb`](https://github.com/hortovanyi/DRLND-Collab-Compete/blob/master/Tennis.ipynb) notebook using the drlnd kernel to train the DDPG agent.

Once trained the model weights will be saved in the same directory in the files `checkpoint1_actor0.pth`, `checkpoint1_actor1.pth` and `checkpoint1_critic.pth`.

The model weights are used by the [`Trained Agent.ipynb` ](https://github.com/hortovanyi/DRLND-Collab-Compete/blob/master/Trained%20Agent.ipynb) notebook against the simulator. 

[Simulator Output Video](https://www.youtube.com/watch?v=DH_ArATXSnE)
