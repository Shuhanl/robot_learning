import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from transformer_model import EmbeddingNetwork, PlanRecognition, PlanProposal, Actor, Critic
from rl import PPO, TargetRL
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
    self.sequence_length = params.sequence_length

    print(self.device)

    self.plan_recognition = PlanRecognition()
    self.plan_proposal = PlanProposal()
    self.embedding = EmbeddingNetwork()
    self.embedding_optimizer = optim.Adam(self.embedding.parameters(), lr=self.lr)
    self.plan_recognition_optimizer = optim.Adam(self.plan_recognition.parameters(), lr=self.lr)  
    self.plan_proposal_optimizer = optim.Adam(self.plan_proposal.parameters(), lr=self.lr)  

    self.actor = Actor()
    self.target_actor = Actor()
    self.target_actor.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=self.lr)
    self.critic = Critic()
    self.target_critic = Critic()
    self.target_critic.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=self.lr)

    self.noise = OrnsteinUhlenbeckProcess(size=self.action_dim)
    self.mse_loss = torch.nn.MSELoss()
    self.tau = params.tau
    self.beta = params.beta
    self.d_model = params.d_model

    # Initialize buffers as tensors
    self.vision_buffer = torch.empty((1, self.sequence_length, self.d_model)).to(self.device)
    self.proprioception_buffer = torch.empty((1, self.sequence_length, self.d_model)).to(self.device)
    self.action_buffer = torch.empty(1, self.sequence_length, self.d_model).to(self.device)

    # self.target_rl = TargetRL(embedding=self.embedding, plan_proposal=self.plan_proposal, 
    #           actor=self.actor,
    #           critic=self.critic,
    #           target_actor=self.target_actor,
    #           target_critic=self.target_critic,
    #           actor_optimizer=self.actor_optimizer,
    #           critic_optimizer=self.critic_optimizer, gamma = params.target_gamma, target_tau=params.target_tau)

    self.ppo = PPO(embedding=self.embedding, plan_proposal=self.plan_proposal,
        actor=self.actor,
        critic=self.critic,
        actor_optimizer=self.actor_optimizer,
        critic_optimizer=self.critic_optimizer,
        gamma = params.gamma,
        tau = params.tau,
        epsilon = params.epsilon)

    """ Wrap your models with DataParallel """
    if torch.cuda.device_count() > 1:
      print("Using", torch.cuda.device_count(), "GPUs!")
      self.embedding = torch.nn.DataParallel(self.embedding)
      self.plan_recognition = torch.nn.DataParallel(self.plan_recognition)

      self.actor = torch.nn.DataParallel(self.actor)
      self.critic = torch.nn.DataParallel(self.critic)
      self.target_actor = torch.nn.DataParallel(self.target_actor)
      self.target_critic = torch.nn.DataParallel(self.target_critic)

    else:
      print("Using single GPU")
      self.embedding = self.embedding.to(self.device)
      self.plan_recognition = self.plan_recognition.to(self.device)
      self.plan_proposal = self.plan_proposal.to(self.device)

      self.actor = self.actor.to(self.device)
      self.critic = self.critic.to(self.device)
      self.target_actor = self.target_actor.to(self.device)
      self.target_critic = self.target_critic.to(self.device)

  def update_buffer(self, buffer, new_data):
      return torch.cat((buffer[:, 1:, :], new_data.unsqueeze(1) ), dim=1)
    
  def set_goal(self, goal):
    self.embedding.eval()
    goal = torch.FloatTensor(goal).to(self.device)
    self.goal = goal.unsqueeze(0)
    # self.goal_embedded = self.embedding.vision_embed(goal)
    
  def get_action(self, vision, proprioception, greedy=True):

    self.embedding.eval()
    self.plan_recognition.eval()
    self.plan_proposal.eval()
    self.actor.eval()

    with torch.no_grad():
      proprioception = torch.FloatTensor(proprioception).to(self.device)
      vision = torch.FloatTensor(vision).to(self.device)
     
      vision_embedded = self.embedding.vision_embed(vision)
      proprioception_embedded = self.embedding.proprioception_embed(proprioception)
      goal_embedded = self.embedding.vision_embed(self.goal)

      self.vision_buffer = self.update_buffer(self.vision_buffer, vision_embedded)
      self.proprioception_buffer = self.update_buffer(self.proprioception_buffer, proprioception_embedded)

      latent = self.plan_proposal(vision_embedded, proprioception_embedded, goal_embedded).sample()

      action, _ = self.actor.get_action(self.vision_buffer, self.proprioception_buffer, latent, goal_embedded, self.action_buffer)
      action_embedded= self.embedding.action_embed(action)
      self.action_buffer = self.update_buffer(self.action_buffer, action_embedded)

      action = action.detach().cpu().numpy()
      if not greedy:
        action += self.noise.sample()

      return action[0]
    
  def pre_train(self, action_labels, video, proprioception):

    self.embedding.train()
    self.plan_recognition.train()
    self.plan_proposal.train()
    self.actor.train()

    action_labels = action_labels.to(self.device)
    video = torch.FloatTensor(video).to(self.device)
    proprioception = torch.FloatTensor(proprioception).to(self.device)

    sequence_length = video.shape[1]
    vision_embedded = torch.stack([self.embedding.vision_embed(video[:, i, :, :, :]) for i in range(sequence_length)], dim=1)
    proprioception_embedded = self.embedding.proprioception_embed(proprioception)
    goal_embedded = vision_embedded[:, -1, :]

    action_buffer = torch.empty((self.batch_size, self.sequence_length, params.d_model)).to(self.device)

    """ Combine CNN output with proprioception data """
    recognition_dist = self.plan_recognition(vision_embedded, proprioception_embedded)
    
    """ Compute the loss for batches sequence of data """
    kl_loss, normal_kl_loss, recon_loss = 0, 0, 0
    for i in range(sequence_length):
      proposal_dist = self.plan_proposal(vision_embedded[:, i, :], proprioception_embedded[:, i, :], goal_embedded)

      kl_loss += compute_regularisation_loss(recognition_dist, proposal_dist)
      
      normal_kl_loss += torch.mean(-0.5 * torch.sum(1 + proposal_dist.scale**2 - proposal_dist.loc**2 - torch.exp(proposal_dist.scale**2), dim=1), dim=0)

      latent = proposal_dist.sample()
      """ Prepend the goal to let the network attend to it """
      
      action, _ = self.actor.get_action(vision_embedded[:, :i, :], proprioception_embedded[:, :i, :], latent, goal_embedded, action_buffer)
      action_embedded= self.embedding.action_embed(action)
      action_buffer = self.update_buffer(action_buffer, action_embedded)

      recon_loss += self.mse_loss(action_labels[:, i, :], action)

    # Compute the batch loss
    loss = self.beta*(kl_loss + normal_kl_loss) + recon_loss / sequence_length

    # Assuming the loss applies to all model components and they're all connected in the computational graph.
    self.embedding_optimizer.zero_grad()
    self.plan_recognition_optimizer.zero_grad()
    self.plan_proposal_optimizer.zero_grad()
    self.actor_optimizer.zero_grad()

    # Only need to call backward once if all parts are connected and contribute to the loss.
    loss.backward()

    clip_grad_norm_(self.embedding.parameters(), max_norm=self.grad_norm_clipping)
    clip_grad_norm_(self.plan_recognition.parameters(), max_norm=self.grad_norm_clipping)
    clip_grad_norm_(self.plan_proposal.parameters(), max_norm=self.grad_norm_clipping)
    clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm_clipping)

    # Then step each optimizer
    self.embedding_optimizer.step()
    self.plan_recognition_optimizer.step()
    self.plan_proposal_optimizer.step()
    self.actor_optimizer.step()

    return loss.item()
    
  def fine_tune(self):

    # critic_loss, actor_loss = self.target_rl.update_model(self.goal)
    # self.target_rl.update_target()

    critic_loss, actor_loss = self.ppo.update_model(self.goal)

    return critic_loss, actor_loss


  def save_checkpoint(self, filename):
      
    # Create a checkpoint dictionary containing the state dictionaries of all components
    checkpoint = {
        'embedding_state_dict': self.embedding.state_dict(),
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

      self.embedding.load_state_dict(checkpoint['embedding_state_dict'])
      self.plan_recognition.load_state_dict(checkpoint['plan_recognition_state_dict'])
      self.plan_proposal.load_state_dict(checkpoint['plan_proposal_state_dict'])

      self.actor.load_state_dict(checkpoint['actor_state_dict'])
      self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])

      self.critic.load_state_dict(checkpoint['critic_state_dict'])
      self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])

      print('Model loaded')