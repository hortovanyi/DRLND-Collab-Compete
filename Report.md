# Report
---
This project utilised the DDPG (Deep Deterministic Policy Gradient) architecture outlined in the previous project for [continuous control](https://github.com/hortovanyi/DRLND-Continuous-Control). That project in turn was based on the [DDPG-Bipedal Udacity project repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal).

This project used a shared critic with agents having a local actor. Was unable to successfully train agents each with their own actor and critic, sharing state and actions. However both sets of code are in the repository.

## State and Action Spaces
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

## Learning Algorithm

The agent training utilised the `ddpg` function in the [Tennis notebook](https://github.com/hortovanyi/DRLND-Collab-Compete/blob/master/Tennis.ipynb).

It continues episodical training via the the ddpg agent until `n_episodes` is reached or until the environment is solved. The  environment is considered solved when the average reward (over the last 100 episodes) is at least +0.5. Note the maximum reward for both of the agents is used for each time step as the reward. 

Each episode continues until `max_t` time-steps is reached or until the environment says it's done.

As above, a reward of +0.1 is provided if an agent hits the ball over the net. If an agent lets the ball hit the ground or hits the ball out of bounds, it receives, a reward of -0.01.

The DDPG agent is contained in [`ddpg_agent.py`](https://github.com/hortovanyi/DRLND-Collab-Compete/blob/master/ddpg_agent.py)

For each time step and agent the Agent acts upon the state utilising a shared (at class level) `replay_buffer`, `critic_local`, `criticl_target` and `critic_optimizer` networks  with local `actor_local`, `actor_target`, `actor_optimizer` networks.

### DDPG Hyper Parameters
- n_episodes (int): maximum number of training episodes
- max_t (int): maximum number of timesteps per episode
- num_agents: number of agents in the environment

Where
`n_episodes=300`, `max_t=1000`

Upon running this numerous times it became apparent that the environment would return done at 1000 timesteps. Higher values were irrelevant.

### DDPG Agent Hyper Parameters

- BUFFER_SIZE (int): replay buffer size
- BATCH_SIZE (int): mini batch size
- GAMMA (float): discount factor
- TAU (float): for soft update of target parameters
- LR_ACTOR (float): learning rate for optimizer
- LR_CRITIC (float): learning rate for optimizer
- WEIGHT_DECAY (float): L2 weight decay
- N_LEARN_UPDATES (int): number of learning updates
- N_TIME_STEPS (int): every n time step do update


Where 
`BUFFER_SIZE = int(1e5)`, `BATCH_SIZE = 256`, `GAMMA = 0.99`, `TAU = 1e-2`, `LR_ACTOR = 1e-3`, `LR_CRITIC = 1e-3`, `WEIGHT_DECAY = 0.0`, `N_LEARN_UPDATES = 6` and `N_TIME_STEPS = 6`

In addition the Ornstein-Uhlenbeck OUNoise `scale` was defaulted to `0.1`. Amplitude of OU Noise occurred starting at `2` reducing by `0.9999` each time step. 

### Neural Networks

Actor and Critic network models were defined in [`ddpg_model.py`](https://github.com/hortovanyi/DRLND-Collab-Compete/blob/master/ddpg_model.py).

The Actor networks utilised two fully connected layers with 256 and 128 units with relu activation and tanh activation for the action space. The network has an initial dimension the same as the state size.

The Critic networks utilised two fully connected layers with 256 and 128 units with leaky_relu activation. The critic network has  an initial dimension the size of the state size plus action size.

## Plot of rewards
![Reward Plot](https://github.com/hortovanyi/DRLND-Collab-Compete/blob/master/output/result.png?raw=true)

```
Episode 100	Average Score: 0.05
Episode 200	Average Score: 0.33
Episode 242	Timestep 869	Noise 0.16	Rewards [-0.01  0.00]	Scores: [ 2.19  2.30]
Environment solved in 142 episodes!	Average Score: 0.50
```

## Ideas for Future Work
Experimentation also occurred with agents having their own actor and critic networks sharing state & actions. Was unable to successfully train this network. As such further work could be done to include Tensorboard output to better understand the affect of hyper parameter changes and their impact to `Q_targets`, `Q_expected` and `critic_loss`.

In addition Proximal Policy Optimization (PPO) and Distributed Distributional Deterministic Policy Gradients (D4PG) methods could be explored.