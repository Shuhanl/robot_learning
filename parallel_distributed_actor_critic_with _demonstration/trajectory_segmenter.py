from typing import List, Tuple, Dict
import numpy as np

class TrajectorySegmenter:
    def __init__(self, n_agent, obs_dim, total_size, batch_size=32, goal_reward=1):
        self.n_agent = n_agent
        self.total_size = total_size
        self.goal_reward = goal_reward
        segment_size = total_size // self.n_agent
        if segment_size == 0:
            raise ValueError("Number of segments is larger than the trajectory data size.")
        self.segment_size = segment_size

        self.obs_buf = np.zeros([n_agent, segment_size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([n_agent, segment_size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([n_agent, segment_size], dtype=np.float32)
        self.rews_buf = np.zeros([n_agent, segment_size], dtype=np.float32)
        self.done_buf = np.zeros([n_agent, segment_size], dtype=np.float32)

        self.batch_size = batch_size
        self.ptr, self.size, = 0, 0

    def segment(self, demo):
        for i in range(self.n_agent):
          start = i * self.segment_size
          end = (i + 1) * self.segment_size if i < self.n_agent - 1 else self.total_size
          demo_slice = demo[start: end+1]
          self.extend(i, demo_slice)

    def setGoal(self):
      for i in range(self.n_agent):
        self.rews_buf[i][-1] = self.goal_reward

    def store(
        self,
        ageng_id: int,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Store the transition in buffer."""
        self.obs_buf[ageng_id][self.ptr] = obs
        self.next_obs_buf[ageng_id][self.ptr] = next_obs
        self.acts_buf[ageng_id][self.ptr] = act
        self.rews_buf[ageng_id][self.ptr] = rew
        self.done_buf[ageng_id][self.ptr] = done

        self.ptr = (self.ptr + 1) % self.segment_size
        self.size = min(self.size + 1, self.segment_size)

    def extend(
        self,
        ageng_id: int,
        trajectory: List[Tuple],
    ):
        """Store the trajectory in buffer."""
        for trajectory in trajectory:
            self.store(ageng_id, *trajectory)

    def sample_batch(self, ageng_id: int) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of data from segment."""
        segment_idxs = np.random.choice(self.segment_size, size=self.batch_size, replace=False)

        return dict(obs=self.obs_buf[ageng_id][segment_idxs],
                    next_obs=self.next_obs_buf[ageng_id][segment_idxs],
                    acts=self.acts_buf[ageng_id][segment_idxs],
                    rews=self.rews_buf[ageng_id][segment_idxs],
                    done=self.done_buf[ageng_id][segment_idxs])