import copy
import torch
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from actor_critic_nn import Actor, Critic
from replay_buffer import PrioritizedReplayBuffer
from noise import OrnsteinUhlenbeckProcess

class Learner:
    def __init__(self, action_shape, robot_state_shape, num_agent, gamma=0.95,lr=0.001,batch_size=1024,memory_size=int(1e6),tau=0.01,grad_norm_clipping = 0.5):
        self.action_shape = action_shape
        self.robot_state_shape = robot_state_shape
        self.gamma = gamma
        self.actor = Actor(self.action_shape, robot_state_shape)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=lr)
        self.critic = Critic(self.action_shape, robot_state_shape)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # Wrap your models with DataParallel
        if torch.cuda.device_count() > 1:
          print("Using", torch.cuda.device_count(), "GPUs!")
          self.actor = torch.nn.DataParallel(self.actor)
          self.target_actor = torch.nn.DataParallel(self.target_actor)
          self.critic = torch.nn.DataParallel(self.critic)
          self.target_critic = torch.nn.DataParallel(self.target_critic)
        else:
          self.actor = self.actor.to(self.device)
          self.target_actor = self.target_actor.to(self.device)
          self.critic = self.critic.to(self.device)
          self.target_critic = self.target_critic.to(self.device)

        self.pri_buffer = PrioritizedReplayBuffer(memory_size, alpha=0.6, beta=0.4)
        self.loss_fn = torch.nn.MSELoss()
        self.batch_size = batch_size
        self.is_gpu = torch.cuda.is_available
        self.noise = OrnsteinUhlenbeckProcess(size=self.action_shape)
        self.grad_norm_clipping = grad_norm_clipping
        self.tau = tau
        self.num_agent = num_agent

    @torch.no_grad()
    def td_targeti(self, reward, vision, next_vision, robot_state, next_robot_state, done):
        next_action = torch.tanh(self.target_actor(vision, robot_state))
        next_q = self.target_critic(next_vision, next_robot_state, next_action)
        td_targeti = reward.unsqueeze(1) + self.gamma * next_q*(1.-done.unsqueeze(1))
        return td_targeti.float()

    def update(self):
      indice = self.pri_buffer.sample_indices(self.batch_size)
      sample = self.pri_buffer.__getitem__(indice)
      obs, action, reward, next_obs, done = sample['obs'], sample['act'], sample['rew'], sample['obs_next'], sample['terminated']

      robot_state = [
          np.array(obs['robot0_eef_pos'], dtype=np.float32).flatten(),
          np.array(obs['robot0_eef_quat'], dtype=np.float32).flatten()
      ]

      vision = np.array(obs['vision'], dtype=np.float32)
      next_robot_state = [
          np.array(next_obs['robot0_eef_pos'], dtype=np.float32).flatten(),
          np.array(next_obs['robot0_eef_quat'], dtype=np.float32).flatten()
      ]
      next_vision = np.array(next_obs['vision'], dtype=np.float32)

      robot_state = np.concatenate(robot_state)
      next_robot_state = np.concatenate(next_robot_state)
      robot_state = torch.tensor(robot_state, dtype=torch.float32)

      next_robot_state = torch.tensor(next_robot_state, dtype=torch.float32)
      vision = torch.tensor(vision, dtype=torch.float32)
      next_vision = torch.tensor(next_vision, dtype=torch.float32)

      action = action.to(self.device)

      reward = torch.FloatTensor(reward).to(self.device)
      done = np.array(done)
      done = torch.IntTensor(done).to(self.device)

      td_targeti = self.td_targeti(reward, vision, next_vision, robot_state, next_robot_state, done)
      current_q = self.critic(vision, robot_state, action)

      critic_loss = self.loss_fn(current_q,td_targeti)
      """ Update priorities based on TD errors """
      td_errors = (td_targeti - current_q).t()          # Calculate the TD Errors
      self.pri_buffer.update_weight(indice, td_errors.data.detach().cpu().numpy())

      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      clip_grad_norm_(self.critic.parameters(),max_norm=self.grad_norm_clipping)
      self.critic_optimizer.step()
      ac_up = self.actor(obs)
      ac = torch.tanh(ac_up)
      pr = -self.critic(obs,ac).mean()
      pg = (ac.pow(2)).mean()
      actor_loss = pr + pg*1e-3
      self.actor_optimizer.zero_grad()
      clip_grad_norm_(self.actor.parameters(),max_norm=self.grad_norm_clipping)
      actor_loss.backward()
      self.actor_optimizer.step()

      for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
        target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
      for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
        target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def get_action(self, vision, robot_state, greedy=False):
        vision = vision.to(self.device)
        robot_state = robot_state.to(self.device)
        action = torch.tanh(self.actor(vision, robot_state))
        if not greedy:
            action += torch.tensor(self.noise.sample(),dtype=torch.float).cuda()
        return np.clip(action.detach().cpu().numpy(),-1.0,1.0)

    def load_checkpoint(self, filename):
      checkpoint = torch.load(filename)

      self.actor.load_state_dict(checkpoint['actor_state_dict'])
      self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
      self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

      self.critic.load_state_dict(checkpoint['critic_state_dict'])
      self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
      self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])