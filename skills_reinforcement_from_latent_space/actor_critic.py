import torch
from torch import nn

class Actor(nn.Module):
    def __init__(self, action_shape, robot_state_shape):
      super().__init__()
      self.vision_net = torch.nn.Sequential(
          # CNN Layers
          torch.nn.Conv2d(1, 32, kernel_size=8, stride=4),
          torch.nn.ReLU(),
          torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
          torch.nn.ReLU(),
          torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
          torch.nn.ReLU(),
          torch.nn.Flatten(),
      )

      # Fully Connected Layers
      # Assuming the flattened output from CNN is of size
      # We concatenate it with the action tensor, so the input size becomes  + action_shape
      self.fc_net = nn.Sequential(
          nn.Linear(1024 + robot_state_shape, 256),
          nn.ReLU(),
          nn.Linear(256, action_shape)
      )

    def forward(self, vision, robot_state):
        vision_out = self.vision_net(vision)
        robot_state = robot_state.unsqueeze(0)
        print(robot_state.size())
        combined = torch.cat([vision_out, robot_state], dim=1)
        action = self.fc_net(combined)
        return action

class Critic(nn.Module):
    def __init__(self, action_shape, robot_state_shape):
        super(Critic, self).__init__()

        # CNN Layers for processing the observation
        self.vision_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Fully Connected Layers
        # Assuming the flattened output from CNN is of size
        # We concatenate it with the action tensor, so the input size becomes  + action_shape
        self.fc_net = nn.Sequential(
            nn.Linear(1024 + action_shape+robot_state_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, vision, robot_state, act):
      if not isinstance(vision, torch.Tensor):
        vision = torch.tensor(vision, dtype=torch.float32)
      if not isinstance(robot_state, torch.Tensor):
        robot_state = torch.tensor(robot_state, dtype=torch.float32)
      if not isinstance(act, torch.Tensor):
        act = torch.tensor(act, dtype=torch.float32)

      obs_repr = self.vision_net(vision)

      # Concatenate the CNN output with the action tensor along dimension 1 (columns)
      combined = torch.cat([obs_repr, act, robot_state], dim=1)

      q_value = self.fc_net(combined)

      return q_value