import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from prioritized_replay_buffer import PrioritizedReplayBuffer
from noise import OrnsteinUhlenbeckProcess
import parameters as params


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
        plan_proposal: nn.Module,
        embedding: nn.Module,
        actor: nn.Module,
        critic: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        gamma: float,
        tau: float,
        epsilon: float,
        entropy_weight: float,
    ):
        """Initialize."""
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.entropy_weight = entropy_weight

        # networks
        self.plan_proposal = plan_proposal
        self.embedding = embedding
        self.actor = actor
        self.critic = critic

        # optimizer
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.pri_buffer = PrioritizedReplayBuffer(alpha=0.6, beta=0.4) 
        self.device = params.device

    def init_buffer(self, vision, proprioception, action, 
              reward, next_vision, next_proprioception, done):
    
        self.pri_buffer.store(vision, proprioception, action, reward, next_vision, next_proprioception, done)

    def compute_gae(self,
        value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gae."""

        delta = (
            reward
            + self.gamma * value * (1-done)
            - value
        )
        advantage = delta + self.gamma * self.tau * (1-done) * advantage + value

        return advantage

    def update_model(self, goal_embedded):

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
        goal_embedded = goal_embedded.repeat(vision.shape[0], 1)  

        vision_embedded = self.embedding.vision_embed(vision)
        next_vision_embedded = self.embedding.vision_embed(next_vision)
        proprioception_embedded = self.embedding.proprioception_embed(proprioception)
        next_proprioception_embedded = self.embedding.proprioception_embed(next_proprioception)
        action_embedded = self.embedding.action_embed(action)

        value = self.critic(vision_embedded, proprioception_embedded, action_embedded, goal_embedded)

        advantage = self.compute_gae(
            value,
            reward,
            done
        )

        # calculate ratios
        _, dist = self.actor(vision_embedded, proprioception_embedded, latent, goal_embedded)
        log_prob = dist.log_prob(action)
        ratio = (log_prob - old_log_prob).exp()

        # actor_loss
        surr_loss = ratio * advantage
        clipped_surr_loss = (
            torch.clamp(ratio, 1.0 - self.epsilon,
                        1.0 + self.epsilon) * advantage
        )

        # entropy
        entropy = dist.entropy().mean()

        actor_loss = (
            -torch.min(surr_loss, clipped_surr_loss).mean()
            - entropy * self.entropy_weight
        )

        # critic_loss
        critic_loss = (advantage - value).pow(2).mean()

        # train critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        # train actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss, critic_loss

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

        # Initialize buffers as tensors
        self.vision_buffer = torch.empty((self.bacth_size, self.sequence_length, params.d_model)).to(self.device)
        self.pproprioception_buffer = torch.empty((self.bacth_size, self.sequence_length, params.d_model)).to(self.device)

    def store_buffer(self, vision, proprioception, action, reward, next_vision, next_proprioception, done):
        self.pri_buffer.store(vision, proprioception, action, reward, next_vision, next_proprioception, done)


    def update_buffer(self, buffer, new_data):

        return torch.cat((buffer[:, 1:, :], new_data.unsqueeze(1) ), dim=1)

    def get_next_action(self, vision_embedded, proprioception_embedded, goal_embedded, greedy=True):
    
        proposal_dist = self.plan_proposal(vision_embedded, proprioception_embedded, goal_embedded)
        proposal_latent = proposal_dist.sample()

        self.vision_buffer = self.update_buffer(self.vision_buffer, vision_embedded)
        self.pproprioception_buffer = self.update_buffer(self.pproprioception_buffer, proprioception_embedded)
        next_action = self.target_actor.get_action(self.vision_buffer, self.pproprioception_buffer, proposal_latent, goal_embedded)

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
        td_errors = torch.abs((target_q - current_q))          # Calculate the TD Errors

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

        td_errors, critic_losses, pg = 0, 0, 0
        for i in range(self.sequence_length):
            vision_embedded = self.embedding.vision_embed(vision[:, i, :, :, :])
            # next_vision_embedded = self.embedding.vision_embed(next_vision)
            proprioception_embedded = self.embedding.proprioception_embed(proprioception[:, i, :])
            # next_proprioception_embedded = self.embedding.proprioception_embed(next_proprioception)
            action_embedded = self.embedding.action_embed(action[:, i, :])
            
            td_error, critic_loss = self.td_target(vision[:, i, :, :, :], next_vision[:, i, :, :, :], proprioception[:, i, :],
                                                next_proprioception[:, i, :], action[:, i, :], goal, reward[:, i], done[:, i])
                                            
            td_errors += td_error
            critic_losses += critic_loss
            pg += (action.pow(2)).mean()
            
        
        td_errors /= self.sequence_length
        critic_losses /= self.sequence_length
        pg /= self.sequence_length

        """ Update priorities based on TD errors """
        self.pri_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        # """ Update critic """
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm_clipping)
        self.critic_optimizer.step()

        pr = -self.critic(vision_embedded, proprioception_embedded, action_embedded).mean()
        actor_loss = pr + 0.1*pg

        """ Update actor """
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm_clipping)
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def update_target(self):
        with torch.no_grad():
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.target_tau  * param.data + (1.0 - self.target_tau ) * target_param.data)
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.target_tau  * param.data + (1.0 - self.target_tau ) * target_param.data)

    