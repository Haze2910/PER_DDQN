import torch
from torch import nn
from torch.nn import functional as F

# DQN
class DQN(nn.Module):
  def __init__(self, num_actions, fc_dim=64):
    super(DQN, self).__init__()
    self.num_actions = num_actions # size of action space
    self.channels = 3 # RGB channels

    # CNN
    self.conv = nn.Sequential(
      nn.Conv2d(self.channels, 16, kernel_size=5, stride=1), # (batch_size, 3, 84, 84) -> (batch_size, 16, 80, 80)
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, stride=2), # (batch_size, 16, 80, 80) -> (batch_size, 16, 39, 39)
      nn.ReLU(),
      nn.Conv2d(32, 32, kernel_size=3, stride=1), # (batch_size, 16, 39, 39) -> (batch_size, 32, 37, 37)
      nn.ReLU(),
      nn.Conv2d(32, 32, kernel_size=3, stride=1), # (batch_size, 32, 37, 37) -> (batch_size, 32, 35, 35)
      nn.ReLU(),
      nn.MaxPool2d(2, 2), # (batch_size, 32, 35, 35) -> (batch_size, 32, 17, 17)
      nn.Flatten()
    )

    # Linear
    self.linear = nn.Sequential(
      nn.Linear(32 * 17 * 17, fc_dim),
      nn.ReLU(),
      nn.Linear(fc_dim, num_actions)
    )

  def forward(self, state):
    state = self.preprocess(state)
    x = self.conv(state)
    return self.linear(x)

  def preprocess(self, state):
    # Remove borders
    state = state[:, :84, 6:90]

    # Normalize the RGB value of each pixel from (0, 255) to (0, 1)
    state = state / 255.0

    # Transpose the state so we have the RGB channels as first dimension for the CNNs
    return state.transpose(-1, -3)


# Dueling DQN
class Dueling_DQN(nn.Module):
  def __init__(self, num_actions, fc_dim=512):
    super(Dueling_DQN, self).__init__()
    self.num_actions = num_actions # size of action space
    self.channels = 3 # RGB channels

    # Shared layer
    self.shared_layer = nn.Sequential(
        nn.Conv2d(self.channels, 32, kernel_size=8, stride=4), # (batch_size, 3, 96, 96) -> (batch_size, 32, 23, 23)
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2), # (batch_size, 32, 23, 23) -> (batch_size, 64, 10, 10)
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1), # (batch_size, 64, 10, 10) -> (batch_size, 64, 8, 8)
        nn.ReLU()
    )

    # Advantage stream
    self.advantage_stream = nn.Sequential(
        nn.Linear(64 * 8 * 8, fc_dim),
        nn.ReLU(),
        nn.Linear(fc_dim, num_actions)
    )
    
    # Value stream
    self.value_stream = nn.Sequential(
        nn.Linear(64 * 8 * 8, fc_dim),
        nn.ReLU(),
        nn.Linear(fc_dim, 1)
    )

  def forward(self, state):

    state = self.preprocess(state)
    x = self.shared_layer(state)
    x = x.view(x.size(0), -1) # flatten into a vector

    values = self.value_stream(x)
    advantages = self.advantage_stream(x)
    q = values + (advantages - torch.mean(advantages, dim=1, keepdim=True))

    return q
  
  def preprocess(self, state):
      # Normalize the RGB value of each pixel from (0, 255) to (0, 1)
      state = state / 255.0

      # Transpose the state so we have the RGB channels as first dimension for the CNNs
      return state.transpose(1, 3)
