import torch
import numpy as np
import torch.optim as optim
from actor_critic_nn import Actor, Critic
from typing import List, Tuple
import gym
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.display import clear_output
from env import actionNormalizer, rewardFunc
from trajectory_segmenter import TrajectorySegmenter
from noise import OUNoise


class MutiAgent:
    """MutiAgent interacting with environment.

    Attribute:
        actor (nn.Module): target actor model to select actions
        actor_target (nn.Module): actor model to predict next actions
        actor_optimizer (Optimizer): optimizer for training actor
        critic (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        critic_optimizer (Optimizer): optimizer for training critic
        demo (TrajectorySegmenter): demonstration data
        n_segments (int): num of segments
        gamma (float): discount factor
        tau (float): parameter for soft target update
        initial_random_steps (int): initial random action steps
        lambda1 (float): weight for policy gradient loss
        lambda2 (float): weight for behavior cloning loss
        noise (OUNoise): noise generator for exploration
        device (torch.device): cpu / gpu
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
    """
    def __init__(
        self,
        env_id,
        memory_size: int,
        n_agent: int,
        batch_size: int,
        demo_batch_size: int,
        ou_noise_theta: float,
        ou_noise_sigma: float,
        demo: list,
        gamma: float = 0.99,
        tau: float = 5e-3,
        initial_random_steps: int = 1e4,
        # loss parameters
        lambda1: float = 1e-3,
        lambda2: int = 1.0
    ):

        env = gym.make(env_id)
        self.global_env = actionNormalizer(env)

        """Initialize."""
        obs_dim = self.global_env.observation_space.shape[0]
        action_dim = self.global_env.action_space.shape[0]

        self.n_agent = n_agent
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps

        # loss parameters
        self.lambda1 = lambda1
        self.lambda2 = lambda2 / demo_batch_size

        self.actor_losses = [[] for _ in range(n_agent)]
        self.critic_losses = [[] for _ in range(n_agent)]
        self.scores = [[] for _ in range(n_agent)]

        # demo segmentation
        self.demo_segmentation = TrajectorySegmenter(n_agent, obs_dim, len(demo), demo_batch_size)
        self.demo_segmentation.segment(demo)

        # initial state of each segment
        self.initial_states = [self.demo_segmentation.obs_buf[i][0] for i in range(n_agent)]

        self.local_env = [gym.make(env_id) for _ in range(n_agent)]
        for i in range(n_agent):
          self.local_env[i] = actionNormalizer(self.local_env[i])
          theta_offset = np.arccos(self.initial_states[i][0])
          print(theta_offset)
          self.local_env[i] = rewardFunc(self.local_env[i], theta_offset)

        # replay segmentation
        self.replay_segmentation = TrajectorySegmenter(n_agent, obs_dim, memory_size, batch_size)

        # noise
        self.noise = OUNoise(
            action_dim,
            theta=ou_noise_theta,
            sigma=ou_noise_sigma,
        )

        # device: cpu / gpu
        self.device = [torch.device("cuda" if torch.cuda.is_available() else "cpu") for _ in range(n_agent)]
        print(self.device)

        # global networks
        self.global_actor = Actor(obs_dim, action_dim).to(self.device[0])
        self.global_critic = Critic(obs_dim + action_dim).to(self.device[0])
        # local networks
        self.local_actor = [Actor(obs_dim, action_dim).to(self.device[i]) for i in range(n_agent)]
        self.local_critic = [Critic(obs_dim + action_dim).to(self.device[i]) for i in range(n_agent)]

        for i in range(n_agent):
            self.local_actor[i].load_state_dict(self.global_actor.state_dict())
            self.local_critic[i].load_state_dict(self.global_critic.state_dict())

        # optimizer
        for i in range(n_agent):
          self.local_actor_optimizer = [optim.Adam(self.local_actor[i].parameters(), lr=3e-4) for i in range(n_agent)]
          self.local_critic_optimizer = [optim.Adam(self.local_critic[i].parameters(), lr=1e-3) for i in range(n_agent)]

        # transition to store in memory
        self.transitions = [list() for _ in range(n_agent)]

        # total steps count
        self.total_step = 0

        # max test steps
        self.max_step = 5000

    def select_action(self, states: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted
        selected_actions = [0 for _ in range(self.n_agent)]
        for i in range(self.n_agent):
          if self.total_step < self.initial_random_steps:
              selected_actions[i] = self.local_env[i].action_space.sample()
          else:
              selected_actions[i] = self.local_actor[i](
                  torch.FloatTensor(states[i]).to(self.device[i])
              ).detach().cpu().numpy()

          # add noise for exploration during training
          noise = self.noise.sample()
          selected_actions[i] = np.clip(selected_actions[i] + noise, -1.0, 1.0)

          self.transitions[i] = [states[i], selected_actions[i]]

        return selected_actions

    def step(self, actions) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_states, rewards, dones = [[] for _ in range(self.n_agent)], [0 for _ in range(self.n_agent)], [0 for _ in range(self.n_agent)]
        for i in range(self.n_agent):
          next_states[i], rewards[i], dones[i], _ = self.local_env[i].step(actions[i])
          self.transitions[i] += [rewards[i], next_states[i], dones[i]]
          self.replay_segmentation.store(i, *self.transitions[i])

        return next_states, rewards, dones

    def update_model(self, agent_id) -> np.ndarray:
        """Update the model by gradient descent."""
        device = self.device[agent_id]

        # sample batch from replay segments
        samples = self.replay_segmentation.sample_batch(agent_id)
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # sample batch from demo segments
        d_samples = self.demo_segmentation.sample_batch(agent_id)
        d_state = torch.FloatTensor(d_samples["obs"]).to(device)
        d_next_state = torch.FloatTensor(d_samples["next_obs"]).to(device)
        d_action = torch.FloatTensor(d_samples["acts"].reshape(-1, 1)).to(device)
        d_reward = torch.FloatTensor(d_samples["rews"].reshape(-1, 1)).to(device)
        d_done = torch.FloatTensor(d_samples["done"].reshape(-1, 1)).to(device)

        masks = 1 - done
        next_action = self.global_actor(next_state)
        next_value = self.global_critic(next_state, next_action)
        curr_return = reward + self.gamma * next_value * masks
        curr_return = curr_return.to(device).detach()

        # train local critic
        values = self.local_critic[agent_id](state, action)
        local_critic_loss = F.mse_loss(values, curr_return)

        self.local_critic_optimizer[agent_id].zero_grad()
        local_critic_loss.backward()
        self.local_critic_optimizer[agent_id].step()

        # train local actor
        # PG loss
        pg_loss = -self.local_critic[agent_id](state, self.local_actor[agent_id](state)).mean()

        # BC loss with Q filter
        pred_action = self.local_actor[agent_id](d_state)
        qf_mask = torch.gt(
            self.local_critic[agent_id](d_state, d_action),
            self.local_critic[agent_id](d_state, pred_action),
        ).to(device)
        qf_mask = qf_mask.float()
        n_qf_mask = int(qf_mask.sum().item())

        if n_qf_mask == 0:
            bc_loss = torch.zeros(1, device=device)
        else:
            bc_loss = (
                torch.mul(pred_action, qf_mask) - torch.mul(d_action, qf_mask)
            ).pow(2).sum() / n_qf_mask

        local_actor_loss = self.lambda1 * pg_loss + self.lambda2 * bc_loss

        self.local_actor_optimizer[agent_id].zero_grad()
        local_actor_loss.backward()
        self.local_actor_optimizer[agent_id].step()


        # To-Do: Update the gobal agent every x episode: Refer to TD3
        self.global_soft_update(agent_id)

        return local_actor_loss.data.detach().cpu().numpy(), local_critic_loss.data.detach().cpu().numpy()

    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        states = [[] for _ in range(self.n_agent)]
        for i in range(self.n_agent):
          states[i] = self.local_env[i].reset(options={'theta': np.arccos(self.initial_states[i][0]), 'theta_dot': self.initial_states[i][2]}, seed=0)
          # states[i] = self.local_env[i].reset(seed=0)

        score = [0 for _ in range(self.n_agent)]

        for self.total_step in range(1, num_frames + 1):
            actions = self.select_action(states)
            next_states, rewards, dones = self.step(actions)
            states = next_states

            for i in range(self.n_agent):
              score[i] += rewards[i]
              # if episode ends
              if dones[i]:
                states[i] = self.local_env[i].reset(seed=0)
                self.scores[i].append(score[i])
                score[i] = 0

              # if training is ready
              if self.total_step > self.initial_random_steps:
                  actor_loss, critic_loss = self.update_model(i)
                  self.actor_losses[i].append(actor_loss)
                  self.critic_losses[i].append(critic_loss)

            # plotting
            if self.total_step % plotting_interval == 0:
                self.plot_train()

        for i in range(self.n_agent):
          self.local_env[i].close()

    def test(self):
        """Test the global agent."""
        device = self.device[0]
        state = self.global_env.reset(seed=0)
        done = False
        score = 0
        test_step = 0

        while not done or self.max_step > test_step:
            action = self.global_actor(
                torch.FloatTensor(state).to(device)).detach().cpu().numpy()

            next_state, reward, done, _ = self.global_env.step(action)

            state = next_state
            score += reward
            test_step += 1

        print("score: ", score)
        self.global_env.close()

    def global_soft_update(self, agent_id):
        """Soft-update: global = tau*local + (1-tau)*global."""
        tau = self.tau

        for g_param, l_param in zip(
            self.global_actor.parameters(), self.local_actor[agent_id].parameters()
        ):
            g_param.data.copy_(tau * l_param.data + (1.0 - tau) * g_param.data)

        for g_param, l_param in zip(
            self.global_critic.parameters(), self.local_critic[agent_id].parameters()
        ):
            g_param.data.copy_(tau * l_param.data + (1.0 - tau) * g_param.data)

    def plot_train(self):
        """Plot the training progresses."""
        def subplot(loc: int, title: str, values: List[float]):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(values)
            # adjust the spacing between subplots
            plt.subplots_adjust(hspace=0.2)

        # loop over the agents
        subplot_params = []
        for i in range(self.n_agent):
            # calculate the subplot location based on the row and column index
            loc = 100*self.n_agent + 30 + 3*i

            # plot the score, actor_loss and critic_loss for each agent
            subplot_params.extend([(loc+1, f"frame {self.total_step}. score: {np.mean(self.scores[i][-10:])}", self.scores[i]),
            (loc + 2, "actor_loss", self.actor_losses[i]),
            (loc + 3, "critic_loss", self.critic_losses[i])])

        clear_output(True)
        # create a figure with self.n_agent rows and 3 columns of subplots
        plt.figure(figsize=(20, 5*self.n_agent))
        for loc, title, values in subplot_params:
          subplot(loc, title, values)
        plt.show()