import gym
import numpy as np

class actionNormalizer(gym.Wrapper):
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

class rewardFunc(gym.Wrapper):
  """ Recompute the reward function based on different goals """
  def __init__(self, env, theta_offset):
    super().__init__(env)
    self.theta_offset = theta_offset

  def step(self, action):
    obs, _, done, info = self.env.step(action)
    theta = np.arccos(obs[0])
    theta_dt = obs[2]
    torque = action
    new_reward =  -((theta-self.theta_offset)**2 + 0.1 * theta_dt**2 + 0.001 * torque**2)
    return obs, new_reward, done, info


