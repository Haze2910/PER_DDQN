import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random


from memory import PrioritizedReplayBuffer, ReplayBuffer
from dqn import DQN, Dueling_DQN

class Agent(nn.Module):
    def __init__(self, 
                 env='CarRacing-v2', 
                 continuous=False, 
                 device=torch.device('cpu'),
                 n_episodes=100,
                 batch_size=32,
                 update_frequency=50,
                 replay_rate=10,
                 epsilon_decay=0.99,
                 epsilon_min=0.1,
                 gamma=0.99,
                 buffer_size=1000,
                 per=True,
                 norm_clip=1.0
                ):
        
        super(Policy, self).__init__()
        self.device = device
        self.env = gym.make(env, continuous=continuous)
        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n

        self.n_episodes = n_episodes
        self.batch_size = batch_size

        self.total_steps = 0
        self.episode_steps = 0
        self.update_frequency = update_frequency
        self.replay_rate = replay_rate

        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        self.buffer_size = buffer_size
        self.memory = PrioritizedReplayBuffer(self.buffer_size) if per else ReplayBuffer(self.buffer_size)

        self.network = DQN(self.action_dim).to(self.device)
        self.target_network = DQN(self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30, eta_min=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30, eta_min=1e-4)
        self.norm_clip = 1.0


    def act(self, state):
        Q = self.network(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        return Q.argmax(dim=1).item()

        
    def forward(self):
        batch, weights, tree_idxs = self.memory.sample(self.batch_size)
        #print(weights)
        states, actions, rewards, next_states, dones = batch

        Q = torch.gather(self.network(states), -1, actions)#self.network(states).gather(-1, actions.unsqueeze(1)).squeeze(1)#
        Q_next = torch.max(self.target_network(next_states), dim=-1).values.reshape(-1, 1)#self.target_network(next_states).max(dim=1).values
        Q_target = rewards + self.gamma * (1 - dones) * Q_next
        
        td_errors = torch.abs(Q - Q_target).detach().cpu()

        self.memory.update_priorities(tree_idxs, td_errors)

        loss = ((Q - Q_target) ** 2 * weights.to(self.device)).mean()#torch.mean((Q - Q_target) * weights.to(self.device))
        #loss = F.smooth_l1_loss(Q, Q_target, beta=(1 / weights)) # less sensible to outliers

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(self.network.parameters(), self.norm_clip)
        self.optimizer.step()

        # Update the target network with some frequency
        if self.episode_steps % self.update_frequency == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        return loss.item()
    
    def train(self):
        for episode in range(self.n_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            done = truncated = False

            episode_reward = 0
            self.episode_steps = 0

            while not (done or truncated):
                
                # Epsilon-greedy action policy
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = self.act(state)

                self.episode_steps += 1
                self.total_steps += 1

                # Perform action and collect reward
                next_state, reward, done, truncated, _ = self.env.step(action)
                episode_reward += reward
            
                next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
                
                # Compute td error
                Q = self.network(state.unsqueeze(0))[0][action]
                Q_next = torch.max(self.target_network(next_state.unsqueeze(0)))#self.target_network(next_state.unsqueeze(0)).max(dim=1).values
                
                td_error = torch.abs(reward + (1 - done) * self.gamma * Q_next - Q).detach().cpu()

                action = torch.tensor([action], dtype=torch.int64).to(self.device)
                reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
                done = torch.tensor([done], dtype=torch.float32).to(self.device)

                # Add experience into memory
                self.memory.push((state, action, reward, next_state, done), td_error)

                # Collect some experience in the memory before starting training
                if self.total_steps >= self.batch_size:

                    # Learn from memory every 10 steps
                    if self.episode_steps % self.replay_rate == 0:
                        self.forward()

                
                state = next_state

                # Exponential annealing for the epsilon
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                
            self.scheduler.step()
            # Log episode results
            print(f"episode {episode}: reward: {episode_reward:.2f}, epsilon: {self.epsilon:.2f}")

    
    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret


