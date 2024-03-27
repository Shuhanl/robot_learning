import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from model import Actor, Critic
from prioritized_replay_buffer import PrioritizedReplayBuffer
from noise import OrnsteinUhlenbeckProcess
import parameters as params
  
class AgentTrainer():
  def __init__(self, gamma=0.95):
    self.gamma = gamma
    self.lr = params.lr
    self.device = params.device
    self.action_dim = params.action_dim
    self.batch_size = params.batch_size
    self.grad_norm_clipping = params.grad_norm_clipping
    self.goal = None

    print(self.device)

    self.actor = Actor()
    self.target_actor = Actor()
    self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=self.lr)

    self.critic = Critic()
    self.target_critic = Critic()
    self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=self.lr)

    self.pri_buffer = PrioritizedReplayBuffer(alpha=0.6, beta=0.4)
    self.noise = OrnsteinUhlenbeckProcess(size=self.action_dim)
    self.mse_loss = torch.nn.MSELoss()
    self.beta = params.beta

    """ Wrap your models with DataParallel """
    if torch.cuda.device_count() > 1:
      print("Using", torch.cuda.device_count(), "GPUs!")
      self.actor = torch.nn.DataParallel(self.actor)
      self.critic = torch.nn.DataParallel(self.critic)
      self.target_actor = torch.nn.DataParallel(self.target_actor)
      self.target_critic = torch.nn.DataParallel(self.target_critic)

    else:
      print("Using single GPU")
      self.actor = self.actor.to(self.device)
      self.critic = self.critic.to(self.device)
      self.target_actor = self.target_actor.to(self.device)
      self.target_critic = self.target_critic.to(self.device)


  def get_action(self, vision, proprioception, greedy=True):

    with torch.no_grad():
      proprioception = torch.FloatTensor(proprioception).unsqueeze(0).to(params.device)
      vision = torch.FloatTensor(vision).unsqueeze(0).to(params.device)

      goal_embeded = self.vision_network(self.goal)
      vision_embeded = self.vision_network(vision)

      proposal_dist = self.plan_proposal(vision_embeded, proprioception, goal_embeded)
      proposal_latent = proposal_dist.sample()
      action = self.actor(vision_embeded, proprioception, proposal_latent, goal_embeded)

      if not greedy:
          action += torch.tensor(self.noise.sample(),dtype=torch.float).to(self.device)

      action = action.detach().cpu().numpy()
      return action[0]
  
  def _get_next_action(self, goal, vision, proprioception, greedy=True):

    with torch.no_grad():
      goal_embeded = self.vision_network(goal)
      vision_embeded = self.vision_network(vision)

      proposal_dist = self.plan_proposal(vision_embeded, proprioception, goal_embeded)
      proposal_latent = proposal_dist.sample()
      next_action = self.target_actor(vision_embeded, proprioception, proposal_latent, goal_embeded)

      if not greedy:
          next_action += torch.tensor(self.noise.sample(),dtype=torch.float).to(self.device)
      return next_action
  
  def _td_target(self, goal, vision, next_vision, proprioception, next_proprioception, action, reward, done):

    vision_embeded = self.vision_network(vision)
    current_q = self.critic(vision_embeded, proprioception, action)

    next_action = self._get_next_action(goal, vision, proprioception, greedy=True)
    next_vision_embeded = self.vision_network(next_vision)
    next_q = self.target_critic(next_vision_embeded, next_proprioception, next_action)
    target_q = reward.unsqueeze(1) + self.gamma * next_q*(1.-done.unsqueeze(1))

    critic_loss = self.mse_loss(current_q, target_q)
    td_errors = torch.abs((target_q - current_q))          # Calculate the TD Errors

    return td_errors, critic_loss

  def init_buffer(self, vision, proprioception, action, 
              reward, next_vision, next_proprioception, done):
    
    self.pri_buffer.store(vision, proprioception, action, reward, next_vision, next_proprioception, done)


  def fine_tune(self):

    buffer = self.pri_buffer.sample_batch()
    vision, proprioception, next_vision, next_proprioception, action, reward, done, weights, indices = buffer['vision'], \
      buffer['proprioception'], buffer['next_vision'], buffer['next_proprioception'], buffer['action'], buffer['reward'], \
        buffer['done'], buffer['weights'], buffer['indices']

    vision = torch.FloatTensor(vision).to(self.device)
    proprioception = torch.FloatTensor(proprioception).to(self.device)
    next_vision = torch.FloatTensor(next_vision).to(self.device)
    next_proprioception = torch.FloatTensor(next_proprioception).to(self.device)
    action = torch.FloatTensor(action).to(self.device)
    reward = torch.FloatTensor(reward).to(self.device)
    done = torch.FloatTensor(done).to(self.device)
    goal = self.goal.repeat(vision.shape[0], 1, 1, 1)

    td_errors, critic_loss = self._td_target(goal, vision, next_vision, proprioception, next_proprioception, action, reward, done)

    """ Update priorities based on TD errors """
    self.pri_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
      
    """ Update critic """
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm_clipping)
    self.critic_optimizer.step()

    vision_embeded = self.vision_network(vision)
    pr = -self.critic(vision_embeded, proprioception, action).mean()
    pg = (action.pow(2)).mean()
    actor_loss = pr + pg*1e-3

    """ Update actor """
    self.actor_optimizer.zero_grad()
    clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm_clipping)
    actor_loss.backward()
    self.actor_optimizer.step()

    """ Soft update target networks """
    for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
      target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
      target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


  def save_checkpoint(self, filename):
      
    # Create a checkpoint dictionary containing the state dictionaries of all components
    checkpoint = {
        'vision_network_state_dict': self.vision_network.state_dict(),
        'actor_state_dict': self.actor.state_dict(),
        'target_actor_state_dict': self.target_actor.state_dict(),
        'critic_state_dict': self.critic.state_dict(),
        'target_critic_state_dict': self.target_critic.state_dict(),
    }
    
    # Use torch.save to serialize and save the checkpoint dictionary
    torch.save(checkpoint, filename)
    print('Model saved')

  def load_checkpoint(self, filename):
      checkpoint = torch.load(filename)

      self.vision_network.load_state_dict(checkpoint['vision_network_state_dict'])

      self.actor.load_state_dict(checkpoint['actor_state_dict'])
      self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])

      self.critic.load_state_dict(checkpoint['critic_state_dict'])
      self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])

      print('Model loaded')