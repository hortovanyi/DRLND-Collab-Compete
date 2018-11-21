import numpy as np
import random
import copy
from collections import namedtuple, deque

from agent_tools import OUNoise, ReplayBuffer
from maddpg_model import Actor, Critic


import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 4e-2              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay

N_LEARN_UPDATES = 4     # number of learning updates
N_TIME_STEPS = 2        # every n time step do update

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    memory = None

    def __init__(self, num, state_size, action_size, random_seed):
        """Initialize an Agent object.

        Params
        ======
            num (int): number of this agent
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.num = num
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size*2+action_size*2, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size*2+action_size*2, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        # Noise process
        self.noise = OUNoise(action_size, random_seed, scale = .1)


    def act(self, state, add_noise=True, noise_amplitude=0.0):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample() * noise_amplitude
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()


class MultiAgent():
    """Interacts with and learns from the environment using multiple agents and a shared critic."""
    
    
    def __init__(self, num_agents, state_size, action_size, random_seed):
        """Initialize an Agent object.

        Params
        ======
            num_agents (int): number of agents to create
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.agents = [Agent(num, state_size, action_size, random_seed) for num in range(num_agents)]
        
        # Replay memory - only intitialise once 
        if Agent.memory is None:
            print("Initialising ReplayBuffer")
            Agent.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
    
    def step(self, time_step, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        Agent.memory.add(states, actions, rewards, next_states, dones)

        # only learn every n_time_steps
        if time_step % N_TIME_STEPS != 0:
            return

        # Learn, if enough samples are available in memory
        if len(Agent.memory) > BATCH_SIZE:
            for i in range(N_LEARN_UPDATES):
                for agent_num in range (self.num_agents):
                    experiences = Agent.memory.ma_sample()
                    self.learn(agent_num, experiences, GAMMA)
    
    
    def learn(self, agent_num, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        
        num_agents = self.num_agents
        #print("num_agents: {}".format(num_agents))
        
        #print("next_states.size() {}".format(next_states.size()))
        
        states_agent = torch.split(states, self.state_size, dim=1)
        #print("states_agent[0].size() {}".format(states_agent[0].size()))
        
        actions_agent = torch.split(actions, self.action_size, dim=1)
        
        rewards_agent = torch.split(rewards, 1, dim=1)
#         print("rewards.size() {}".format(rewards.size()))
#         print("rewards_agent[{}].size() {}".format(agent_num, rewards_agent[agent_num].size()))
        
        next_states_agent = torch.split(next_states, self.state_size, dim=1)
        #print("next_states_agent[0].size() {}".format(next_states_agent[0].size()))
        
        dones_agent = torch.split(dones, 1, dim=1)
#         print("dones.size() {}".format(dones.size()))
#         print("dones_agent[{}].size() {}".format(agent_num, dones_agent[agent_num].size()))
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = torch.cat([self.agents[i].actor_target(next_states_agent[i]) for i in range(num_agents)],dim=1).to(device)
        #print("actions_next.size(), {}".format(actions_next.size(), actions_next))
        
        critic_states_next = torch.cat([next_states, actions_next],dim=1).to(device)
#         critic_states_next = next_states
        #print("critic_state_next.size() {}".format(critic_states_next.size()))
        actions_next_agent = torch.split(actions_next, self.action_size, dim=1)
        critic_actions_next = actions_next_agent[agent_num]
        #print("critic_action_next.size() {}".format(critic_actions_next.size()))
        Q_targets_next = self.agents[agent_num].critic_target(critic_states_next,critic_actions_next)
        #print("Q_targets_next.size() {}".format(Q_targets_next.size()))
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards_agent[agent_num] + (gamma * Q_targets_next * (1 - dones_agent[agent_num]))

        # Compute critic loss
        critic_states = torch.cat([states, actions],dim=1).to(device)

#         critic_states = states
        Q_expected = self.agents[agent_num].critic_local(critic_states,actions_agent[agent_num])
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.agents[agent_num].critic_optimizer.zero_grad()
        critic_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.agents[agent_num].critic_local.parameters(), 1)
        self.agents[agent_num].critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
#         actions_pred = torch.cat([self.agents[i].actor_local(states_agent[i]) for i in range(num_agents)],dim=1)
#         print("actions_pred.size() {}".format(actions_pred.size()))
#         actions_pred_agent = torch.split(actions_pred, self.action_size, dim=1)
#         print("actions_pred_agent[{}].size() {}".format(agent_num,actions_pred_agent[agent_num].size()))      
#         actor_loss = -self.agents[agent_num].critic_local(critic_states, actions_pred_agent[agent_num]).mean()
#         actor_loss = -self.agents[agent_num].critic_local(states, actions_pred_agent[agent_num]).mean()
             
        actions_pred = self.agents[agent_num].actor_local(states_agent[agent_num])
        actor_loss = -self.agents[agent_num].critic_local(critic_states, actions_pred).mean()
        
        # Minimize the loss
        self.agents[agent_num].actor_optimizer.zero_grad()
        actor_loss.backward()
        self.agents[agent_num].actor_optimizer.step()
        
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.agents[agent_num].critic_local, self.agents[agent_num].critic_target, TAU)
        self.soft_update(self.agents[agent_num].actor_local, self.agents[agent_num].actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def reset(self):
        for i in range(self.num_agents):
            self.agents[i].noise.reset()
    