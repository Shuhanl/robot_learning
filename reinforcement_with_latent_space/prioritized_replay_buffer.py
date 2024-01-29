
import random
import numpy as np
from typing import Dict, List
from segment_tree import MinSegmentTree, SumSegmentTree
import parameters as params

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self):
        """Initialization."""
        self.vision_buf = np.zeros([params.memory_size, *params.vision_dim], dtype=np.float32)
        self.proprioception_buf = np.zeros([params.memory_size, params.proprioception_dim], dtype=np.float32)

        self.next_vision_buf = np.zeros([params.memory_size, *params.vision_dim], dtype=np.float32)
        self.next_proprioception_buf = np.zeros([params.memory_size, params.proprioception_dim], dtype=np.float32)

        self.action_buf = np.zeros([params.memory_size, params.action_dim], dtype=np.float32)
        self.reward_buf = np.zeros([params.memory_size], dtype=np.float32)
        self.done_buf = np.zeros([params.memory_size], dtype=np.float32)
        self.max_size = params.memory_size
        self.ptr, self.size = 0, 0

    def store(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray, next_obs: np.ndarray, done: np.ndarray):
        vision, pos, quat = obs['frontview_image'], obs['robot0_eef_pos'], obs['robot0_eef_quat']
        next_vision, next_pos, next_quat = next_obs['frontview_image'], next_obs['robot0_eef_pos'], next_obs['robot0_eef_quat']

        self.vision_buf[self.ptr] = vision
        self.proprioception_buf = np.concatenate([pos, quat], axis=-1)

        self.next_vision_buf[self.ptr] = next_vision
        self.next_proprioception_buf = np.concatenate([next_pos, next_quat], axis=-1)
        
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        indices = np.random.choice(self.size, size=params.batch_size, replace=False)
        return dict(
            vision = self.vision_buf[indices],
            proprioception = self.proprioception_buf[indices],
            next_vision = self.next_vision_buf[indices],
            next_proprioception = self.next_proprioception_buf[indices],
            action = self.action_buf[indices],
            reward = self.reward_buf[indices],
            done = self.done_buf[indices],
            indices=indices
        )

    def __len__(self) -> int:
        return self.size
    
class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(
        self, 
        alpha: float = 0.6,
        beta: float = 0.4
    ):
        """Initialization."""
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer, self).__init__()
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        self.beta = beta
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(
        self, 
        obs: np.ndarray, 
        act: int, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool
    ):
        """Store experience and priority."""
        super().store(obs, act, rew, next_obs, done)
        
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= params.batch_size
        assert self.beta > 0
        
        indices = self._sample_proportional()
        
        return dict(
            vision = self.vision_buf[indices],
            proprioception = self.proprioception_buf[indices],
            next_vision = self.next_vision_buf[indices],
            next_proprioception = self.next_proprioception_buf[indices],
            action = self.action_buf[indices],
            reward = self.reward_buf[indices],
            done = self.done_buf[indices],
            weights=np.array([self._calculate_weight(i, self.beta) for i in indices]),
            indices=indices
        )
        
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / params.batch_size
        
        for i in range(params.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight