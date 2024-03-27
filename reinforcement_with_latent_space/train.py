import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from model import VisionNetwork, PlanRecognitionTransformer, PlanProposal, DirectActorTransformer, Critic
from prioritized_replay_buffer import PrioritizedReplayBuffer
from noise import OrnsteinUhlenbeckProcess
from utils import compute_regularisation_loss
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

    self.vision_network = VisionNetwork()
    self.plan_recognition = PlanRecognitionTransformer()
    self.plan_proposal = PlanProposal()
    self.vision_network_optimizer = optim.Adam(self.vision_network.parameters(), lr=self.lr)
    self.plan_recognition_optimizer = optim.Adam(self.plan_recognition.parameters(), lr=self.lr)  
    self.plan_proposal_optimizer = optim.Adam(self.plan_proposal.parameters(), lr=self.lr)  

    self.actor = DirectActorTransformer()
    self.target_actor = DirectActorTransformer()
    self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=self.lr)
    self.critic = Critic()
    self.target_critic = Critic()
    self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=self.lr)

    self.pri_buffer = PrioritizedReplayBuffer(alpha=0.6, beta=0.4)
    self.noise = OrnsteinUhlenbeckProcess(size=self.action_dim)
    self.mse_loss = torch.nn.MSELoss()
    self.tau = params.tau
    self.beta = params.beta

    """ Wrap your models with DataParallel """
    if torch.cuda.device_count() > 1:
      print("Using", torch.cuda.device_count(), "GPUs!")
      self.vision_network = torch.nn.DataParallel(self.vision_network)
      self.plan_recognition = torch.nn.DataParallel(self.plan_recognition)
      self.plan_proposal = torch.nn.DataParallel(self.plan_proposal)

      self.actor = torch.nn.DataParallel(self.actor)
      self.critic = torch.nn.DataParallel(self.critic)
      self.target_actor = torch.nn.DataParallel(self.target_actor)
      self.target_critic = torch.nn.DataParallel(self.target_critic)

    else:
      print("Using single GPU")
      self.vision_network = self.vision_network.to(self.device)
      self.plan_recognition = self.plan_recognition.to(self.device)
      self.plan_proposal = self.plan_proposal.to(self.device)

      self.actor = self.actor.to(self.device)
      self.critic = self.critic.to(self.device)
      self.target_actor = self.target_actor.to(self.device)
      self.target_critic = self.target_critic.to(self.device)


  def set_goal(self, goal):
    self.goal = torch.FloatTensor(goal).to(self.device)
    self.goal = self.goal.unsqueeze(0)

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
  
  def pre_train(self, action_labels, video, proprioceptions):

    action_labels = torch.FloatTensor(action_labels).to(self.device)
    video = torch.FloatTensor(video).to(self.device)
    proprioceptions = torch.FloatTensor(proprioceptions).to(self.device)

    sequence_length = video.shape[1]
    video_embeded = torch.stack([self.vision_network(video[:, i, :, :, :]) for i in range(sequence_length)], dim=1)
    goal_embeded = video_embeded[:, -1, :]

    """ Combine CNN output with proprioception data """
    combined = torch.cat([video_embeded, proprioceptions], dim=-1)
    recognition_dist = self.plan_recognition(combined)

    """ Compute the loss for batches sequence of data """
    kl_loss, normal_kl_loss, recon_loss = 0, 0, 0
    for i in range(sequence_length):
      vision_embeded = video_embeded[:, i, :]
      proprioception = proprioceptions[:, i, :]
      action_label = action_labels[:, i, :]
      proposal_dist = self.plan_proposal(vision_embeded, proprioception, goal_embeded)

      kl_loss += compute_regularisation_loss(recognition_dist, proposal_dist)
      
      normal_kl_loss += torch.mean(-0.5 * torch.sum(1 + proposal_dist.scale**2 - 
                                                 proposal_dist.loc**2 - torch.exp(proposal_dist.scale**2), dim=1), dim=0)

      proposal_latent = proposal_dist.sample()
      """ Prepend the goal to let the network attend to it """
      # pred_action = torch.zeros_like(action_label)
      pred_action = self.actor(vision_embeded, proprioception, proposal_latent, goal_embeded)

      recon_loss += self.mse_loss(action_label, pred_action)

    # Compute the batch loss
    loss = self.beta*(kl_loss + normal_kl_loss) + recon_loss / sequence_length

    # Assuming the loss applies to all model components and they're all connected in the computational graph.
    self.vision_network.zero_grad()
    self.plan_recognition.zero_grad()
    self.plan_proposal.zero_grad()
    self.actor.zero_grad()

    # Only need to call backward once if all parts are connected and contribute to the loss.
    loss.backward()

    clip_grad_norm_(self.vision_network.parameters(), max_norm=self.grad_norm_clipping)
    clip_grad_norm_(self.plan_recognition.parameters(), max_norm=self.grad_norm_clipping)
    clip_grad_norm_(self.plan_proposal.parameters(), max_norm=self.grad_norm_clipping)
    clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm_clipping)

    # Then step each optimizer
    self.vision_network_optimizer.step()
    self.plan_recognition_optimizer.step()
    self.plan_proposal_optimizer.step()
    self.actor_optimizer.step()

    return loss.item()
  
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
  
    return critic_loss.item(), actor_loss.item()

  def save_checkpoint(self, filename):
      
    # Create a checkpoint dictionary containing the state dictionaries of all components
    checkpoint = {
        'vision_network_state_dict': self.vision_network.state_dict(),
        'plan_recognition_state_dict': self.plan_recognition.state_dict(),
        'plan_proposal_state_dict': self.plan_proposal.state_dict(),
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
      self.plan_recognition.load_state_dict(checkpoint['plan_recognition_state_dict'])
      self.plan_proposal.load_state_dict(checkpoint['plan_proposal_state_dict'])

      self.actor.load_state_dict(checkpoint['actor_state_dict'])
      self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])

      self.critic.load_state_dict(checkpoint['critic_state_dict'])
      self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])

      print('Model loaded')