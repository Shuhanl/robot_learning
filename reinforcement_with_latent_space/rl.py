import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from prioritized_replay_buffer import PrioritizedReplayBuffer
from noise import OrnsteinUhlenbeckProcess
import parameters as params
from utils import convert_observation


class PPO:
    """PPO Agent.
    Attributes:
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        batch_size (int): batch size for sampling
        epsilon (float): amount of clipping surrogate objective
        epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        transition (list): temporory storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)        
    """

    def __init__(
        self,
        embedding: nn.Module,
        plan_proposal: nn.Module,
        actor: nn.Module,
        critic: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        gamma: float,
        tau: float,
        epsilon: float,
    ):
        """Initialize."""
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.env = None

        # networks
        self.plan_proposal = plan_proposal
        self.embedding = embedding
        self.actor = actor
        self.critic = critic

        # optimizer
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.mse_loss = torch.nn.MSELoss()
        self.grad_norm_clipping = params.grad_norm_clipping

        self.sequence_length = params.sequence_length
        self.rollout_length = params.rollout_length
        self.device = params.device

        # Initialize sequential buffers as tensors
        self.vision_buffer = torch.empty((1, self.sequence_length, params.d_model)).to(self.device)
        self.proprioception_buffer = torch.empty((1, self.sequence_length, params.d_model)).to(self.device)
        self.action_buffer = torch.empty((1, self.sequence_length, params.d_model)).to(self.device)

    def set_env(self, env):
        self.env = env

    def update_seq_buffer(self, buffer, new_data):
        return torch.cat((buffer[:, 1:, :], new_data.unsqueeze(1) ), dim=1)
    
    def clear_seq_buffer(self):
        self.vision_buffer = torch.empty((1, self.sequence_length, params.d_model)).to(self.device)
        self.proprioception_buffer = torch.empty((1, self.sequence_length, params.d_model)).to(self.device)
        self.action_buffer = torch.empty((1, self.sequence_length, params.d_model)).to(self.device)

    def compute_rtgs(self, reward_batch):
        rtgs_batch = torch.zeros_like(reward_batch)
        # Start from the last time step
        rtgs_batch[:, -1] = reward_batch[:, -1]
        # Calculate rewards-to-go by iterating backwards from second last to first timestep
        for t in reversed(range(reward_batch.size(1) - 1)):  # reward_batch.size(1) is rollout_len
            rtgs_batch[:, t] = reward_batch[:, t] + self.gamma * rtgs_batch[:, t + 1]

        return rtgs_batch
    
    def rollout_storage(self, goal):

        # roll-out storage
        vision_batch = torch.zeros([1, self.rollout_length, *params.vision_dim], dtype=torch.float32, device=self.device)
        proprioception_batch = torch.zeros([1, self.rollout_length, params.proprioception_dim], dtype=torch.float32, device=self.device)
        action_batch = torch.zeros([1, self.rollout_length, params.action_dim], dtype=torch.float32, device=self.device)
        action_log_prob_batch = torch.zeros([1, self.rollout_length], dtype=torch.float32, device=self.device)
        reward_batch = torch.zeros([1, self.rollout_length], dtype=torch.float32, device=self.device)
        done_batch = torch.zeros([1, self.rollout_length], dtype=torch.float32, device=self.device)

        observation, _ = self.env.reset()
        vision, proprioception = convert_observation(observation)

        # Add batch dimension
        proprioception = torch.FloatTensor(proprioception).to(self.device)
        vision = torch.FloatTensor(vision).to(self.device)

        self.clear_seq_buffer()
        for i in range(self.rollout_length):
    
            vision_embedded = self.embedding.vision_embed(vision)
            proprioception_embedded = self.embedding.proprioception_embed(proprioception)
            goal_embedded = self.embedding.vision_embed(goal)

            self.vision_buffer = self.update_seq_buffer(self.vision_buffer, vision_embedded)
            self.proprioception_buffer = self.update_seq_buffer(self.proprioception_buffer, proprioception_embedded)

            latent = self.plan_proposal(vision_embedded, proprioception_embedded, goal_embedded).sample()
            action, _ = self.actor.get_action(self.vision_buffer, self.proprioception_buffer, latent, goal_embedded, self.action_buffer)

            observation, reward, done, truncated, info = self.env.step(action[0].detach().cpu().numpy())
            action_embedded = self.embedding.action_embed(action)
            self.action_buffer = self.update_seq_buffer(self.action_buffer, action_embedded)
            
            vision, proprioception = convert_observation(observation)

            # Add batch dimension
            proprioception = torch.FloatTensor(proprioception).to(self.device)
            vision = torch.FloatTensor(vision).to(self.device)

            vision_batch[0, i] = vision
            proprioception_batch[0, i] = proprioception
            action_batch[0, i] = action
            reward_batch[0, i] = reward
            done_batch[0, i] = done

            if done:
                break
        
        rtgs_batch = self.compute_rtgs(reward_batch)

        return vision_batch, proprioception_batch, action_batch, action_log_prob_batch, reward_batch, rtgs_batch, done_batch


    def evaluate(self, vision: torch.Tensor, proprioception: torch.Tensor, action: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:

        vision_embedded = self.embedding.vision_embed(vision)
        proprioception_embedded = self.embedding.proprioception_embed(proprioception)
        action_embedded = self.embedding.action_embed(action)
        goal_embedded = self.embedding.vision_embed(goal)

        self.vision_buffer = self.update_seq_buffer(self.vision_buffer, vision_embedded)
        self.proprioception_buffer = self.update_seq_buffer(self.proprioception_buffer, proprioception_embedded)
        
        latent = self.plan_proposal(vision_embedded, proprioception_embedded, goal_embedded).sample()
        action, action_log_prob = self.actor.get_action(self.vision_buffer, self.proprioception_buffer, latent, goal_embedded, self.action_buffer)
        action_embedded = self.embedding.action_embed(action)
        self.action_buffer = self.update_seq_buffer(self.action_buffer, action_embedded)

        value = self.critic(vision_embedded, proprioception_embedded, action_embedded)

        return value, action_log_prob

    def update_model(self, goal):
        vision_batch, proprioception_batch, action_batch, action_log_prob_batch, reward_batch, rtgs_batch, done_batch = self.rollout_storage(goal)

        goal = goal.repeat(vision_batch.shape[0], 1, 1, 1)  

        actor_losses, critic_losses = 0, 0

        self.clear_seq_buffer()
        for i in range(self.rollout_length):

            value, action_log_prob = self.evaluate(vision_batch[:, i, :, :, :], proprioception_batch[:, i, :], action_batch[:, i, :], goal)

            ratio = torch.exp(action_log_prob - action_log_prob_batch[:, i])
            advantage = rtgs_batch[:, i] - value

            # Calculate surrogate losses.
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
            actor_losses += -torch.min(surr1, surr2)
            critic_losses += self.mse_loss(value, rtgs_batch[:, i])

        actor_loss = actor_losses / self.rollout_length
        critic_loss = critic_losses / self.rollout_length

        print(actor_loss, critic_loss)

        # train critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm_clipping)
        self.critic_optimizer.step()

        # train actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm_clipping)
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

class TargetRL:
    def __init__(self, embedding: nn.Module, plan_proposal: nn.Module, actor: nn.Module, critic: nn.Module, target_actor: nn.Module, 
                 target_critic: nn.Module, actor_optimizer: torch.optim.Optimizer, 
                 critic_optimizer: torch.optim.Optimizer, gamma: float, target_tau: float):
        """Initialize."""
        self.embedding = embedding
        self.plan_proposal = plan_proposal
        self.actor = actor
        self.critic = critic
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer   
        self.pri_buffer = PrioritizedReplayBuffer(alpha=0.6, beta=0.4) 
        self.noise = OrnsteinUhlenbeckProcess(size=params.action_dim)
        self.mse_loss = torch.nn.MSELoss()
        self.target_tau = target_tau
        self.gamma = gamma
        self.grad_norm_clipping = params.grad_norm_clipping
        self.device = params.device
        self.sequence_length = params.sequence_length
        self.bacth_size = params.batch_size 

        # Initialize sequential buffers as tensors
        self.vision_buffer = torch.empty((self.bacth_size, self.sequence_length, params.d_model)).to(self.device)
        self.proprioception_buffer = torch.empty((self.bacth_size, self.sequence_length, params.d_model)).to(self.device)
        self.action_buffer = torch.empty((self.bacth_size, self.sequence_length, params.d_model)).to(self.device)

    def store_buffer(self, vision, proprioception, action, reward, next_vision, next_proprioception, done):
        self.pri_buffer.store(vision, proprioception, action, reward, next_vision, next_proprioception, done)


    def update_seq_buffer(self, buffer, new_data):
        return torch.cat((buffer[:, 1:, :], new_data.unsqueeze(1) ), dim=1)

    def get_next_action(self, vision_embedded, proprioception_embedded, goal_embedded, greedy=True):
    
        proposal_dist = self.plan_proposal(vision_embedded, proprioception_embedded, goal_embedded)
        proposal_latent = proposal_dist.sample()

        self.vision_buffer = self.update_seq_buffer(self.vision_buffer, vision_embedded)
        self.proprioception_buffer = self.update_seq_buffer(self.proprioception_buffer, proprioception_embedded)
        next_action, _ = self.target_actor.get_action(self.vision_buffer, self.proprioception_buffer, proposal_latent, goal_embedded, self.action_buffer)

        next_action_embedded = self.embedding.action_embed(next_action)
        self.action_buffer = self.update_seq_buffer(self.action_buffer, next_action_embedded)

        if not greedy:
            next_action += torch.tensor(self.noise.sample(),dtype=torch.float).to(self.device)
        return next_action
    
    def td_target(self, vision, next_vision, proprioception, 
                    next_proprioception, action, goal, reward, done):


        vision_embedded = self.embedding.vision_embed(vision)
        next_vision_embedded = self.embedding.vision_embed(next_vision)
        proprioception_embedded = self.embedding.proprioception_embed(proprioception)
        next_proprioception_embedded = self.embedding.proprioception_embed(next_proprioception)
        action_embedded = self.embedding.action_embed(action)
        goal_embedded = self.embedding.vision_embed(goal)

        current_q = self.critic(vision_embedded, proprioception_embedded, action_embedded)

        next_action = self.get_next_action(vision_embedded, proprioception_embedded, goal_embedded, greedy=True)
        next_action_embedded = self.embedding.action_embed(next_action)
        next_q = self.target_critic(next_vision_embedded, next_proprioception_embedded, next_action_embedded)
        target_q = reward.unsqueeze(1) + self.gamma * next_q*(1.-done.unsqueeze(1))

        critic_loss = self.mse_loss(current_q, target_q)
        td_errors = torch.abs((target_q - current_q))          # Calculate the TD Errors for Prioritized Experience Replay

        return td_errors, critic_loss

    def update_model(self, goal):

        self.embedding.eval()
        self.plan_proposal.eval()
        self.actor.train()
        self.critic.train()

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
        goal = goal.repeat(vision.shape[0], 1, 1, 1)  

        td_errors, critic_losses, actor_losses = 0, 0, 0
        for i in range(self.sequence_length):            
            td_error, critic_loss = self.td_target(vision[:, i, :, :, :], next_vision[:, i, :, :, :], proprioception[:, i, :],
                                                next_proprioception[:, i, :], action[:, i, :], goal, reward[:, i], done[:, i])
                      
            td_errors += td_error
            critic_losses += critic_loss       
        
        td_errors /= self.sequence_length
        critic_losses /= self.sequence_length

        """ Update priorities based on TD errors """
        self.pri_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        """ Update critic """
        self.critic_optimizer.zero_grad()
        critic_losses.backward()
        clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm_clipping)
        self.critic_optimizer.step()

        for i in range(self.sequence_length):
            vision_embedded = self.embedding.vision_embed(vision[:, i, :, :, :])
            proprioception_embedded = self.embedding.proprioception_embed(proprioception[:, i, :])
            action_embedded = self.embedding.action_embed(action[:, i, :])
            actor_loss = -self.critic(vision_embedded, proprioception_embedded, action_embedded).mean()
            actor_losses += actor_loss  
                    
        actor_losses /= self.sequence_length

        """ Update actor """
        self.actor_optimizer.zero_grad()
        actor_losses.backward()
        clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm_clipping)
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def update_target(self):
        with torch.no_grad():
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.target_tau  * param.data + (1.0 - self.target_tau ) * target_param.data)
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.target_tau  * param.data + (1.0 - self.target_tau ) * target_param.data)

    