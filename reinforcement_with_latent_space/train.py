import copy
import torch
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from model import VisionNetwork, PlanRecognition, PlanProposal, Actor, Critic
from prioritized_replay_buffer import PrioritizedReplayBuffer
from noise import OrnsteinUhlenbeckProcess
from utils import compute_loss, compute_regularisation_loss
import parameters as params
  
class AgentTrainer():
  def __init__(self, gamma=0.95, tau=0.01,):
    self.gamma = gamma

    print(params.device)

    self.vision_network = VisionNetwork()
    self.plan_recognition = PlanRecognition()
    self.plan_proposal = PlanProposal()
    self.vision_network_optimizer = optim.Adam(self.vision_network.parameters(), lr=params.lr)
    self.plan_recognition_optimizer = optim.Adam(self.plan_recognition.parameters(), lr=params.lr)  
    self.plan_proposal_optimizer = optim.Adam(self.plan_proposal.parameters(), lr=params.lr)  

    self.actor = Actor()
    self.target_actor = copy.deepcopy(self.actor)
    self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=params.lr)
    self.critic = Critic()
    self.target_critic = copy.deepcopy(self.critic)
    self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=params.lr)

    self.pri_buffer = PrioritizedReplayBuffer(alpha=0.6, beta=0.4)
    self.noise = OrnsteinUhlenbeckProcess(size=params.action_dim)
    self.tau = tau

    # Wrap your models with DataParallel
    if torch.cuda.device_count() > 1:
      print("Using", torch.cuda.device_count(), "GPUs!")
      self.vision_network = torch.nn.DataParallel(self.vision_network)
      self.plan_recognition = torch.nn.DataParallel(self.plan_recognition)
      self.plan_proposal = torch.nn.DataParallel(self.plan_proposal)

      self.actor = torch.nn.DataParallel(self.actor)
      self.critic = torch.nn.DataParallel(self.critic)

    else:
      print("Using single GPU")
      self.vision_network = self.vision_network.to(params.device)
      self.plan_recognition = self.plan_recognition.to(params.device)
      self.plan_proposal = self.plan_proposal.to(params.device)

      self.actor = self.actor.to(params.device)
      self.critic = self.critic.to(params.device)


  def get_action(self, goal, vision, proprioception, greedy=False):

    goal_embeded = self.vision_network(goal)
    vision_embeded = self.vision_network(vision)

    proposal_dist = self.plan_proposal(vision_embeded, proprioception, goal_embeded)
    proposal_latent = proposal_dist.resample()
    action = self.actor(vision_embeded, proprioception, proposal_latent, goal_embeded)

    if not greedy:
        action += torch.tensor(self.noise.sample(),dtype=torch.float).cuda()
    return np.clip(action.detach().cpu().numpy(),-1.0,1.0)
  
  def pre_train(self, goal, vision, actions_label, video, proprioception):

    goal_embeded = self.vision_network(goal)
    vision_embeded = self.vision_network(vision)
    video_embeded = torch.stack([self.vision_network(video[:, i]) for i in range(params.sequence_length)], dim=1)
    # Combine CNN output with proprioception data
    combined = torch.cat([video_embeded, proprioception, actions_label], dim=-1)
  
    recognition_dist = self.plan_recognition(combined)
    proposal_dist = self.plan_proposal(vision_embeded, proprioception, goal_embeded)

    kl_loss = compute_regularisation_loss(recognition_dist, proposal_dist)
    normal_kl_loss = torch.mean(-0.5 * torch.sum(1 + proposal_dist.scale**2 - 
                                                 proposal_dist.loc**2 - torch.exp(proposal_dist.scale**2), dim=1), dim=0)

    proposal_latent = proposal_dist.resample()
    actions = self.actor(vision_embeded, proposal_latent, goal_embeded)

    recon_loss = compute_loss(actions_label, actions, params.sequence_length)

    loss = kl_loss + normal_kl_loss + recon_loss

    # Backpropagation
    self.vision_network.zero_grad()  # Clear existing gradients
    loss.backward()        # Calculate gradients
    clip_grad_norm_(self.vision_network.parameters(),max_norm=params.grad_norm_clipping)
    self.vision_network.step()  # Update parameters

    self.plan_recognition.zero_grad()  # Clear existing gradients
    loss.backward()        # Calculate gradients
    clip_grad_norm_(self.plan_recognition.parameters(),max_norm=params.grad_norm_clipping)
    self.plan_recognition.step()  # Update parameters

    self.plan_proposal.zero_grad()  # Clear existing gradients
    loss.backward()        # Calculate gradients
    clip_grad_norm_(self.plan_proposal.parameters(),max_norm=params.grad_norm_clipping)
    self.plan_proposal.step()  # Update parameters

    self.actor_optimizer.zero_grad()  # Clear existing gradients
    loss.backward()        # Calculate gradients
    clip_grad_norm_(self.actor.parameters(),max_norm=params.grad_norm_clipping)
    self.actor_optimizer.step()  # Update parameters

    return loss.item()
  
  @torch.no_grad()
  def td_target(self, goal, reward, vision, next_vision, next_proprioception, done):

    next_action = self.get_action(goal, vision)

    next_vision_embeded = self.vision_network(next_vision)
    next_q = self.target_critic(next_vision_embeded, next_proprioception, next_action)
    target = reward.unsqueeze(1) + self.gamma * next_q*(1.-done.unsqueeze(1))

    return target.float()

  def init_buffer(self, vision, proprioception, action, 
              reward, next_vision, next_proprioception, done):
    
    self.pri_buffer.store(vision, proprioception, action, reward, next_vision, next_proprioception, done)


  def fine_tune(self, goal):

    vision, proprioception, next_vision, next_proprioception, action, reward, done, weights, indices = self.pri_buffer.sample_batch()
   
    target_q = self.td_target(goal, reward, vision, next_vision, next_proprioception, done)

    vision_embeded = self.vision_network(vision)
    current_q = self.critic(vision_embeded, proprioception, action)

    critic_loss = torch.nn.MSELoss(current_q, target_q)
    """ Update priorities based on TD errors """
    td_errors = (target_q - current_q).t()          # Calculate the TD Errors

    self.pri_buffer.update_priorities(indices, td_errors.data.detach().cpu().numpy())

    action = self.get_action(goal, vision, greedy=False)
    pr = -self.critic(vision_embeded, proprioception, action).mean()
    pg = (action.pow(2)).mean()
    actor_loss = pr + pg*1e-3

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    clip_grad_norm_(self.critic.parameters(), max_norm=params.grad_norm_clipping)
    self.critic_optimizer.step()

    self.actor_optimizer.zero_grad()
    clip_grad_norm_(self.actor.parameters(), max_norm=params.grad_norm_clipping)
    actor_loss.backward()
    self.actor_optimizer.step()
  
    for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
      target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
      target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

  def save_checkpoint(self, filename):
      
      checkpoint = torch.load(filename)

      self.vision_network._save_to_state_dict(checkpoint['vision_network_state_dict'])
      self.plan_recognition._save_to_state_dict(checkpoint['plan_recognition_state_dict'])
      self.plan_proposal._save_to_state_dict(checkpoint['plan_proposal_state_dict'])
      
      self.actor._save_to_state_dict(checkpoint['actor_state_dict'])
      self.target_actor._save_to_state_dict(checkpoint['target_actor_state_dict'])

      self.critic._save_to_state_dict(checkpoint['critic_state_dict'])
      self.target_critic._save_to_state_dict(checkpoint['target_critic_state_dict'])

      print('Model saved')

  def load_checkpoint(self, filename):
      checkpoint = torch.load(filename)

      self.vision_network._load_from_state_dict(checkpoint['vision_network_state_dict'])
      self.plan_recognition._load_from_state_dict(checkpoint['plan_recognition_state_dict'])
      self.plan_proposal._load_from_state_dict(checkpoint['plan_proposal_state_dict'])

      self.actor.load_state_dict(checkpoint['actor_state_dict'])
      self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])

      self.critic.load_state_dict(checkpoint['critic_state_dict'])
      self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])

      print('Model loaded')